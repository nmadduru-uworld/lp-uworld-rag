"""Retrieve -> resolve -> rank.

  * Retrieve -- per collection, hybrid dense (its own embed model) + BM25 -> RRF.
  * Resolve  -- for each hit, join its linked ids (controller_id, DB collections, feature siblings'
                technical hub, a controller's used_by) via a plain in-memory id lookup against the
                same collection's docstore -- no graph DB. Cite, don't inline: a linked node not
                independently retrieved comes back as {id, title, doc_type}, never full text.
  * Rank     -- merge collections' output, apply the doc_type quota (docs_technical only), optional
                cross-encoder rerank, truncate to top_k.

Heavy imports are lazy so the CLI's --help and config loading work without the ML deps installed.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from .chunker import decode_list_field
from .config import RagConfig

COLLECTIONS = ("functional", "technical")


def _abs(cfg: RagConfig, rel: str) -> str:
    return str((cfg.root_path / rel).resolve())


def _docstore_dir(cfg: RagConfig, collection: str) -> str:
    return _abs(cfg, os.path.join(cfg.persist_dir(collection), "docstore"))


def _resolve_device(cfg: RagConfig) -> str:
    if cfg.embed.device and cfg.embed.device != "auto":
        return cfg.embed.device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_embed_model(cfg: RagConfig, collection: str):
    """nomic HuggingFace embedding (or a per-collection override) with asymmetric task prefixes."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return HuggingFaceEmbedding(
        model_name=cfg.embed_model_name(collection),
        trust_remote_code=cfg.embed.trustRemoteCode,
        query_instruction=cfg.embed.queryPrefix,
        text_instruction=cfg.embed.textPrefix,
        device=_resolve_device(cfg),
        normalize=True,
    )


def get_chroma_client(cfg: RagConfig, collection: str):
    """Each collection gets its own PersistentClient -- two independent local databases, not two
    collections sharing one client (see StoreConfig / RagConfig.persist_dir)."""
    import chromadb
    return chromadb.PersistentClient(path=_abs(cfg, cfg.persist_dir(collection)))


def get_collection(cfg: RagConfig, client, collection: str):
    return client.get_or_create_collection(cfg.collection_name(collection))


def load_index(cfg: RagConfig, collection: str, embed_model=None):
    from llama_index.core import VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embed_model = embed_model or get_embed_model(cfg, collection)
    client = get_chroma_client(cfg, collection)
    vstore = ChromaVectorStore(chroma_collection=get_collection(cfg, client, collection))
    return VectorStoreIndex.from_vector_store(vstore, embed_model=embed_model)


def _load_docstore(cfg: RagConfig, collection: str):
    from llama_index.core.storage.docstore import SimpleDocumentStore

    path = os.path.join(_docstore_dir(cfg, collection), "docstore.json")
    if not os.path.exists(path):
        return None
    return SimpleDocumentStore.from_persist_path(path)


def _build_id_index(docstore) -> dict[str, list[str]]:
    """Map every stable id a node exposes (endpoint_id, controller_id, db_id, page_id, feature) to
    the underlying content-hash node id(s) that carry it -- the "no graph DB" join mechanism."""
    idx: dict[str, list[str]] = {}

    def add(key, node_id):
        if key:
            idx.setdefault(key, []).append(node_id)

    if docstore is None:
        return idx
    for node_id, doc in docstore.docs.items():
        md = doc.metadata or {}
        doc_type = md.get("doc_type")
        add(md.get("page_id"), node_id)
        # Index a stable id only where the node OWNS it, not everywhere it's merely referenced --
        # e.g. an endpoint node also carries `controller_id`, but as a pointer to its controller, not
        # as its own identity. Indexing that too would leak the endpoint itself into another
        # endpoint's controller citation whenever two endpoints share one controller.
        if doc_type == "endpoint":
            add(md.get("endpoint_id"), node_id)
        elif doc_type == "controller-context":
            add(md.get("controller_id"), node_id)
        elif doc_type == "db-collection":
            add(md.get("db_id"), node_id)
        elif doc_type == "technical-hub":
            for feat in decode_list_field(md.get("feature")):
                add(feat, node_id)
    return idx


def _build_children_index(docstore) -> dict[str, list[str]]:
    """WS4: map a section-split page's overview/parent node id -> its own leaf section node ids
    (the reverse of each leaf's own ``parent_id``), so a hit on the OVERVIEW can cite its sections
    ("there's more -- Contract, Flow, Gotchas exist, call expand() on the one you need") the same
    way a hit on a LEAF cites its overview. Built once per cache lifetime; a page with no sections
    (whole-page fallback, or any doc_type WS4 doesn't split) simply has no entry here."""
    children: dict[str, list[str]] = {}
    if docstore is None:
        return children
    for node_id, doc in docstore.docs.items():
        parent_id = (doc.metadata or {}).get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append(node_id)
    return children


class RetrieverCache:
    """The expensive, question-independent pieces of retrieval, built once and reused across many
    query()/expand() calls in one process (the MCP server's whole session lifetime)."""

    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        self.embed_models = {c: get_embed_model(cfg, c) for c in COLLECTIONS}
        self.vindexes = {c: load_index(cfg, c, embed_model=self.embed_models[c]) for c in COLLECTIONS}
        self.docstores = {c: _load_docstore(cfg, c) for c in COLLECTIONS}
        pool_size = max(cfg.retrieval.poolSize, cfg.retrieval.topK)
        self.bm25 = {}
        for c in COLLECTIONS:
            ds = self.docstores[c]
            if ds is not None:
                from llama_index.retrievers.bm25 import BM25Retriever
                self.bm25[c] = BM25Retriever.from_defaults(docstore=ds, similarity_top_k=pool_size)
            else:
                self.bm25[c] = None
        self.id_index = {c: _build_id_index(self.docstores[c]) for c in COLLECTIONS}
        self.children_of = {c: _build_children_index(self.docstores[c]) for c in COLLECTIONS}
        self.reranker = None
        if cfg.rerank.enabled:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            # top_n generous on purpose -- query() slices to the caller's actual top_k itself
            # afterward, so this only needs to not drop candidates before that slice happens.
            self.reranker = SentenceTransformerRerank(model=cfg.rerank.model, top_n=pool_size)


def build_retriever_cache(cfg: RagConfig) -> RetrieverCache:
    return RetrieverCache(cfg)


def _build_retriever(cfg: RagConfig, collection: str, top_k: int, cache: RetrieverCache):
    vector_retriever = cache.vindexes[collection].as_retriever(similarity_top_k=top_k)
    bm25 = cache.bm25[collection]
    if bm25 is None:
        return vector_retriever  # vector-only until an ingest has persisted this collection's docstore

    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever

    return QueryFusionRetriever(
        [vector_retriever, bm25],
        mode=cfg.retrieval.fusionMode,
        num_queries=cfg.retrieval.numQueries,  # 1 -> no query generation, no LLM call
        similarity_top_k=top_k,
        use_async=False,
        llm=MockLLM(),
    )


def _apply_quota(cfg: RagConfig, results: list, top_k: int) -> list:
    """Guarantee minimum representation per doc_type in docs_technical, then backfill by score.
    Same technique as any RRF-fused multi-category retrieval -- fitted here to this project's own
    doc_type set (endpoint/controller-context/db-collection), not borrowed from elsewhere."""
    quotas = cfg.retrieval.quotas
    pool = sorted(results, key=lambda r: (r.score if r.score is not None else 0.0), reverse=True)

    selected: list = []
    selected_ids: set = set()
    for doc_type, min_n in quotas.items():
        candidates = [r for r in pool
                      if (r.node.metadata or {}).get("doc_type") == doc_type and r.node.id_ not in selected_ids]
        for r in candidates[:min_n]:
            selected.append(r)
            selected_ids.add(r.node.id_)

    remaining = top_k - len(selected)
    if remaining > 0:
        for r in pool:
            if r.node.id_ not in selected_ids:
                selected.append(r)
                selected_ids.add(r.node.id_)
                remaining -= 1
                if remaining == 0:
                    break
    if len(selected) > top_k:
        selected = sorted(selected, key=lambda r: (r.score if r.score is not None else 0.0),
                          reverse=True)[:top_k]
    return selected


def _normalize_scores(results: list) -> list:
    """Min-max normalize a pool's scores to [0, 1] in place, in rank order.

    RRF scores are a function of rank-within-pool, not absolute relevance -- a 2-document pool's
    rank-1 hit and a 20-document pool's rank-1 hit both land near the top of their own RRF curve
    regardless of how well either actually matches the query, so comparing their raw scores directly
    (as a plain sort across pools would) lets a tiny pool's every result outrank a much larger pool's
    genuinely-better matches purely because the tiny pool has fewer competitors to be ranked against.
    Confirmed live: a 2-chunk docs_functional consistently outscored the correct docs_technical hit
    on unrelated queries until this was added. Normalizing within each pool first makes "best in a
    2-doc pool" and "best in a 20-doc pool" comparable on the same 0-1 scale before merging.
    """
    if not results:
        return results
    scores = [r.score if r.score is not None else 0.0 for r in results]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        for r in results:
            r.score = 1.0
        return results
    for r in results:
        s = r.score if r.score is not None else 0.0
        r.score = (s - lo) / (hi - lo)
    return results


def _maybe_rerank(cfg: RagConfig, question: str, results: list, cache: RetrieverCache | None = None) -> list:
    if not cfg.rerank.enabled or not results:
        return results
    from llama_index.core.schema import QueryBundle

    if cache is not None and cache.reranker is not None:
        reranker = cache.reranker
    else:
        # No cache (e.g. a one-off call) -- build one-shot rather than reuse a stale instance.
        from llama_index.core.postprocessor import SentenceTransformerRerank
        reranker = SentenceTransformerRerank(model=cfg.rerank.model, top_n=len(results))
    return reranker.postprocess_nodes(results, QueryBundle(query_str=question))


def _resolve_citations(node_md: dict, node_id: str, collection: str, cache: RetrieverCache,
                        exclude_ids: set) -> list[dict]:
    """A hit's linked-but-not-independently-retrieved nodes, as {id, title, doc_type} only -- never
    inlined full text (see the plan's Document model: cite, don't inline).

    WS4 adds the section small-to-big join, symmetric in both directions: a LEAF section hit (has
    its own ``parent_id``) cites its overview; an OVERVIEW hit cites its own leaf sections (via
    ``cache.children_of``) -- "there's more: Contract, Flow, Gotchas exist" -- so a caller sees a
    section-split page's other parts exist without them ever being inlined.
    """
    idx = cache.id_index[collection]
    docstore = cache.docstores[collection]
    if docstore is None:
        return []

    candidate_keys: list[str] = []
    doc_type = node_md.get("doc_type")
    if doc_type == "endpoint":
        if node_md.get("controller_id"):
            candidate_keys.append(node_md["controller_id"])
        # db_ids is the endpoint's own forward reference to the data stores it reads (dependency-
        # direction-correct: endpoint -> data store, not the historical reverse consumed_by on the
        # db-collection side) -- a direct read, no reverse-index lookup needed.
        candidate_keys.extend(decode_list_field(node_md.get("db_ids")))
        candidate_keys.extend(decode_list_field(node_md.get("feature")))  # -> each feature's technical-hub node
    elif doc_type == "controller-context":
        candidate_keys.extend(decode_list_field(node_md.get("used_by")))

    citations = []
    seen = set()
    for key in candidate_keys:
        for nid in idx.get(key, []):
            if nid in exclude_ids or nid in seen:
                continue
            seen.add(nid)
            doc = docstore.docs.get(nid)
            if doc is None:
                continue
            md = doc.metadata or {}
            citations.append({"id": key, "title": md.get("title"), "doc_type": md.get("doc_type")})

    # WS4: parent (leaf -> its overview) and children (overview -> its leaf sections) are looked up
    # by raw node id directly, not through the stable-id index above -- a section has no stable id
    # of its own distinct from its endpoint_id/db_id (it shares its page's), so "the id I'd look up"
    # and "the node I actually mean" aren't the same thing the way they are for controller/db/feature.
    related_node_ids: list[str] = []
    parent_id = node_md.get("parent_id")
    if parent_id:
        related_node_ids.append(parent_id)
    related_node_ids.extend(cache.children_of[collection].get(node_id, []))

    for nid in related_node_ids:
        if nid in exclude_ids or nid in seen:
            continue
        seen.add(nid)
        doc = docstore.docs.get(nid)
        if doc is None:
            continue
        md = doc.metadata or {}
        stable_id = md.get("endpoint_id") or md.get("db_id") or md.get("page_id")
        citations.append({"id": stable_id, "title": md.get("title"), "doc_type": md.get("doc_type")})

    return citations


def query(cfg: RagConfig, question: str, collection: str | None = None, top_k: int | None = None,
          include_siblings: bool = True, cache: RetrieverCache | None = None) -> dict:
    """Retrieve for a question, optionally scoped to one collection. ``top_k`` overrides the
    configured default so a caller can size retrieval to its own intent (narrow for a bugfix, wider
    for scoping a new feature) -- this tool makes no LLM call to infer that itself.
    ``include_siblings=False`` skips the feature/technical-hub resolve step for a caller that only
    wants the one hit (cheaper, narrower -- the bugfix case).
    """
    cache = cache or build_retriever_cache(cfg)
    top_k = top_k or cfg.retrieval.topK
    pool_size = max(cfg.retrieval.poolSize, top_k)
    collections = [collection] if collection else list(COLLECTIONS)

    pooled = []
    for c in collections:
        retriever = _build_retriever(cfg, c, pool_size, cache)
        for r in retriever.retrieve(question):
            r.node.metadata["_collection"] = c
            pooled.append(r)

    technical_pool = [r for r in pooled if r.node.metadata.get("_collection") == "technical"]
    functional_pool = [r for r in pooled if r.node.metadata.get("_collection") == "functional"]

    if cfg.rerank.enabled and (functional_pool or technical_pool):
        # Rerank the FULL merged candidate set before truncating, not after -- otherwise a naive
        # cross-pool score merge (see _normalize_scores) could already have cut a genuinely relevant
        # result before rerank ever gets a chance to rescue it. Quota still runs first (it's a
        # doc_type-diversity guarantee, not a relevance judgment), but against the wider pool_size,
        # not top_k, so it doesn't itself over-truncate ahead of rerank.
        quota_technical = _apply_quota(cfg, technical_pool, pool_size) if technical_pool else []
        merged = _maybe_rerank(cfg, question, functional_pool + quota_technical, cache=cache)[:top_k]
    else:
        quota_technical = _apply_quota(cfg, technical_pool, top_k) if technical_pool else []
        merged = sorted(_normalize_scores(functional_pool) + _normalize_scores(quota_technical),
                         key=lambda r: (r.score if r.score is not None else 0.0), reverse=True)[:top_k]

    hit_ids = {r.node.id_ for r in merged}
    chunks = []
    for r in merged:
        md = r.node.metadata or {}
        c = md.get("_collection")
        entry = {
            "id": md.get("endpoint_id") or md.get("controller_id") or md.get("db_id") or md.get("page_id"),
            "title": md.get("title"),
            "docType": md.get("doc_type"),
            "collection": c,
            "content": r.node.get_content(),
            "score": round(float(r.score), 6) if r.score is not None else None,
        }
        if include_siblings:
            entry["citations"] = _resolve_citations(md, r.node.id_, c, cache, hit_ids)
        chunks.append(entry)

    return {"chunks": chunks}


def expand(cfg: RagConfig, node_id_or_stable_id: str, cache: RetrieverCache | None = None) -> list[dict]:
    """Resolve a stable id (ep::..., ctrl::..., db::...) or a bare page_id to its full chunk
    content(s) -- the other half of "cite, don't inline": pull a citation's detail on demand."""
    cache = cache or build_retriever_cache(cfg)
    out = []
    for c in COLLECTIONS:
        docstore = cache.docstores[c]
        if docstore is None:
            continue
        node_ids = cache.id_index[c].get(node_id_or_stable_id, [])
        if not node_ids and node_id_or_stable_id in docstore.docs:
            node_ids = [node_id_or_stable_id]
        for nid in node_ids:
            doc = docstore.docs.get(nid)
            if doc is None:
                continue
            md = doc.metadata or {}
            out.append({
                "title": md.get("title"), "docType": md.get("doc_type"), "collection": c,
                "content": doc.get_content(),
            })
    return out


def status(cfg: RagConfig) -> str:
    """Chunk counts grouped by (collection, doc_type)."""
    lines = []
    total = 0
    for c in COLLECTIONS:
        client = get_chroma_client(cfg, c)
        coll = get_collection(cfg, client, c)
        n = coll.count()
        total += n
        if n == 0:
            lines.append(f"{c} | (empty)")
            continue
        got = coll.get(include=["metadatas"])
        counts = Counter(m.get("doc_type") or "?" for m in got.get("metadatas", []))
        for doc_type, cnt in sorted(counts.items()):
            lines.append(f"{c} | {doc_type} | {cnt} chunks")
    if total == 0:
        return "lp-uworld-rag: index empty -- run `python -m lp_uworld_rag ingest`."
    lines.append(f"total | | {total} chunks")
    return "\n".join(lines)
