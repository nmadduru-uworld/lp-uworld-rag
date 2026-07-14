"""Generalized direct-Chroma retrieval engine for the "index" mode of the Repo RAG Retrieval
Contract (v2, see ``docs/repo-rag-contract.md``) -- lets a repo plug into ``deep_query``/
``query_code`` by declaring embed/store/retrieval config in its manifest instead of running its
own MCP server. No subprocess, no IPC: the embedding model and Chroma/BM25 corpus load once,
in-process, kept alive for the orchestrator's whole lifetime (see :class:`RetrieverCache`).

Ported from -- and kept behaviorally identical to -- the hand-written pipeline this generalizes
over (``reports_rag/index.py``), with that pipeline's one piece of bespoke Python logic
(``_category``'s overview/entity special-casing) turned into declarative config
(``categoryMap``/``overviewHeadingSuffix``) so a second repo can reuse this engine without writing
Python. "Index" mode fixes the chunk metadata a repo's docstore must carry: ``heading``,
``source_type``, ``layer`` (optional -- absent on non-code chunks), ``file_path``, and optionally
``start_line``/``end_line``/``parent_id`` (with ``parent_summary`` living on the parent node's own
metadata). A repo whose chunk schema doesn't fit this should use "serve" mode instead.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

from . import retrieval_engine


class IndexEmbedConfig(BaseModel):
    model: str
    trustRemoteCode: bool = True
    queryPrefix: str = ""
    textPrefix: str = ""
    device: str = "auto"  # "auto" -> cuda if available else cpu; or force "cuda"/"cpu"


class IndexStoreConfig(BaseModel):
    persistDir: str  # resolved relative to the manifest's own directory, not repoRoot
    collection: str


class IndexRetrievalConfig(BaseModel):
    # WS1 (token-economy plan): lowered 8->5 -- default-topK retrieval was shown (token_usage_bench.py
    # baseline) to cost MORE tokens than reading the one relevant file in several scenarios. A repo's
    # manifest can still override this per its own needs.
    topK: int = 5
    poolSize: int = 24
    fusionMode: str = "reciprocal_rerank"
    numQueries: int = 1
    quotas: dict[str, int] = Field(default_factory=dict)
    priorityOrder: list[str] = Field(default_factory=list)
    minCategoryChars: int = 0
    minCategoryExtra: int = 0
    hintBoost: float = 1.0
    # Declarative generalization of the hardcoded _category() this engine is ported from: bucket a
    # chunk for quota purposes by its own `layer` value through this map (a value with no entry ->
    # "other"). A chunk whose heading ends with overviewHeadingSuffix is always "other" regardless
    # of layer -- an overview must never consume a guaranteed code/entity slot with just a class
    # signature, crowding out the sibling chunk that actually answers the question.
    categoryMap: dict[str, str] = Field(default_factory=dict)
    overviewHeadingSuffix: str | None = " (overview)"
    rerankEnabled: bool = False
    rerankModel: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class IndexConfig(BaseModel):
    embed: IndexEmbedConfig
    store: IndexStoreConfig
    retrieval: IndexRetrievalConfig = Field(default_factory=IndexRetrievalConfig)
    # Optional L1 retrieval override (see repo_registry.get_retriever): a dotted import path or a
    # ``.py`` file path to a repo-supplied factory returning an object with a
    # ``query(question, top_k, file_hints) -> dict`` method, for a repo whose retrieval is too
    # complex for this config alone. None -> use the built-in engine below (the default).
    retriever: str | None = None


def get_embed_model(cfg: IndexEmbedConfig):
    return retrieval_engine.build_embed_model(
        model=cfg.model,
        trust_remote_code=cfg.trustRemoteCode,
        query_prefix=cfg.queryPrefix,
        text_prefix=cfg.textPrefix,
        device=cfg.device,
    )


def _docstore_dir(persist_dir: Path) -> Path:
    return persist_dir / "docstore"


def _load_docstore(persist_dir: Path):
    return retrieval_engine.load_docstore(str(_docstore_dir(persist_dir) / "docstore.json"))


def load_index(persist_dir: Path, collection: str, embed_model):
    return retrieval_engine.load_vector_index(str(persist_dir), collection, embed_model)


class RetrieverCache:
    """The expensive, question-independent pieces of retrieval for one repo's "index" manifest --
    built once and reused for the orchestrator's whole process lifetime. Without this, every
    ``query()`` call would reload the embedding model and re-tokenize the entire BM25 corpus from
    scratch (~17-18s per call, per the pipeline this is ported from -- see
    ``reports_rag/index.py``'s own ``RetrieverCache`` docstring)."""

    def __init__(self, persist_dir: Path, index_cfg: IndexConfig):
        self.persist_dir = persist_dir
        self.cfg = index_cfg
        self.embed_model = get_embed_model(index_cfg.embed)
        self.vindex = load_index(persist_dir, index_cfg.store.collection, self.embed_model)
        self.docstore = _load_docstore(persist_dir)
        self.bm25 = None
        if self.docstore is not None:
            pool_size = retrieval_engine.pool_size(index_cfg.retrieval.poolSize, index_cfg.retrieval.topK)
            self.bm25 = retrieval_engine.build_bm25(self.docstore, pool_size)


def build_retriever_cache(persist_dir: Path, index_cfg: IndexConfig) -> RetrieverCache:
    return RetrieverCache(persist_dir, index_cfg)


def _build_retriever(retrieval_cfg: IndexRetrievalConfig, top_k: int, cache: RetrieverCache):
    vector_retriever = cache.vindex.as_retriever(similarity_top_k=top_k)
    # bm25 None here (no persisted docstore yet) -> build_fusion_retriever returns vector-only.
    return retrieval_engine.build_fusion_retriever(
        vector_retriever, cache.bm25,
        top_k=top_k, fusion_mode=retrieval_cfg.fusionMode, num_queries=retrieval_cfg.numQueries)


def _category(node_with_score, retrieval_cfg: IndexRetrievalConfig) -> str:
    md = node_with_score.node.metadata or {}
    heading = md.get("heading") or ""
    if retrieval_cfg.overviewHeadingSuffix and heading.endswith(retrieval_cfg.overviewHeadingSuffix):
        return "other"
    return retrieval_cfg.categoryMap.get(md.get("layer"), "other")


def _apply_quota(retrieval_cfg: IndexRetrievalConfig, results: list, top_k: int) -> list:
    """Guarantee minimum representation per category, then backfill remaining slots by score --
    and top up a thin category past a char-count floor, not just a chunk-count floor (a category's
    single highest-scoring chunk can be a near-empty fragment that satisfies a count minimum while
    giving nothing useful). Ported verbatim from the pipeline this generalizes."""
    quotas = retrieval_cfg.quotas
    min_chars = retrieval_cfg.minCategoryChars
    max_extra = retrieval_cfg.minCategoryExtra
    pool = sorted(results, key=lambda r: (r.score if r.score is not None else 0.0), reverse=True)

    selected: list = []
    selected_ids: set = set()
    for cat, min_n in quotas.items():
        cat_candidates = [r for r in pool if _category(r, retrieval_cfg) == cat and r.node.id_ not in selected_ids]
        taken = cat_candidates[:min_n]
        total_chars = sum(len(r.node.get_content()) for r in taken)
        extra = 0
        idx = min_n
        while total_chars < min_chars and extra < max_extra and idx < len(cat_candidates):
            r = cat_candidates[idx]
            taken.append(r)
            total_chars += len(r.node.get_content())
            extra += 1
            idx += 1
        selected.extend(taken)
        selected_ids.update(r.node.id_ for r in taken)

    remaining = top_k - len(selected)
    if remaining > 0:
        for r in pool:
            if r.node.id_ not in selected_ids:
                selected.append(r)
                selected_ids.add(r.node.id_)
                remaining -= 1
                if remaining == 0:
                    break
    if len(selected) > top_k:  # quotas summed above top_k -- keep the highest-scored
        selected = sorted(selected, key=lambda r: (r.score if r.score is not None else 0.0),
                           reverse=True)[:top_k]
    return selected


def _priority_reorder(retrieval_cfg: IndexRetrievalConfig, nodes: list) -> list:
    """Stable sort selected nodes by doc_type (else source_type) per configured priority. Runs
    AFTER _apply_quota -- decides presentation order, not which chunks made the cut."""
    order = {name: i for i, name in enumerate(retrieval_cfg.priorityOrder)}

    def rank(nws) -> int:
        md = nws.node.metadata or {}
        key = md.get("doc_type") or md.get("source_type") or ""
        return order.get(key, len(order))

    return sorted(nodes, key=rank)


def _apply_hint_boost(retrieval_cfg: IndexRetrievalConfig, results: list, file_hints: list[str]) -> list:
    """Multiply the fused score of chunks matching a caller hint (case-insensitive substring
    against file_path/heading). Runs BEFORE quota selection so a hinted chunk can make the cut,
    not merely reorder the already-selected set. Strictly a boost, never a filter (contract
    guardrail C1) -- a stale hint leaves every candidate's score untouched."""
    hints = [h.strip().lower() for h in file_hints if h and h.strip()]
    if not hints:
        return results
    for r in results:
        md = r.node.metadata or {}
        haystack = f"{md.get('file_path') or ''}|{md.get('heading') or ''}".lower()
        if any(h in haystack for h in hints):
            r.score = (r.score if r.score is not None else 0.0) * retrieval_cfg.hintBoost
    return results


def _maybe_rerank(retrieval_cfg: IndexRetrievalConfig, question: str, results: list) -> list:
    if not retrieval_cfg.rerankEnabled or not results:
        return results
    return retrieval_engine.rerank(question, results, model=retrieval_cfg.rerankModel)


def query(persist_dir: Path, index_cfg: IndexConfig, question: str, top_k: int | None = None,
          file_hints: list[str] | None = None, cache: RetrieverCache | None = None) -> dict:
    """Retrieve for a question against one repo's declared "index" config; return a dict matching
    the contract's ``query_rag`` result shape (the same shape a "serve" mode server returns)."""
    retrieval_cfg = index_cfg.retrieval
    cache = cache or build_retriever_cache(persist_dir, index_cfg)
    top_k = top_k or retrieval_cfg.topK
    pool_size = retrieval_engine.pool_size(retrieval_cfg.poolSize, top_k)

    retriever = _build_retriever(retrieval_cfg, pool_size, cache)
    results = retriever.retrieve(question)

    if file_hints:
        results = _apply_hint_boost(retrieval_cfg, results, file_hints)
    results = _maybe_rerank(retrieval_cfg, question, results)
    results = _apply_quota(retrieval_cfg, results, top_k)
    results = _priority_reorder(retrieval_cfg, results)

    docstore = cache.docstore
    selected_ids = {r.node.id_ for r in results}

    chunks = []
    for r in results:
        md = r.node.metadata or {}
        entry = {
            "heading": md.get("heading"),
            "sourceType": md.get("source_type"),
            "layer": md.get("layer"),  # None on non-code chunks (e.g. rules) -- contract allows null
            "filePath": md.get("file_path"),
            "content": r.node.get_content(),
            "score": retrieval_engine.round_score(r.score),
        }
        if md.get("start_line"):
            entry["startLine"] = md["start_line"]
            entry["endLine"] = md.get("end_line")
        parent_id = md.get("parent_id")
        if parent_id and parent_id not in selected_ids and docstore is not None:
            try:
                parent_summary = (docstore.get_document(parent_id).metadata or {}).get("parent_summary")
                if parent_summary:
                    entry["parentSummary"] = parent_summary
            except ValueError:
                pass  # parent not found in docstore (stale index) -- skip enrichment, not fatal
        chunks.append(entry)

    return {"chunks": chunks}


def status(persist_dir: Path, index_cfg: IndexConfig) -> str:
    """Chunk counts grouped by (layer, source_type) from the Chroma collection -- no cache
    required, this is a liveness/count check, not a query."""
    collection = retrieval_engine.open_chroma_collection(str(persist_dir), index_cfg.store.collection)
    total = collection.count()
    if total == 0:
        return f"{index_cfg.store.collection}: index empty"
    got = collection.get(include=["metadatas"])
    counts = Counter((m.get("layer") or "-", m.get("source_type") or "?") for m in got.get("metadatas", []))
    lines = [f"{layer} | {stype} | {n} chunks" for (layer, stype), n in sorted(counts.items())]
    lines.append(f"total | | {total} chunks")
    return "\n".join(lines)
