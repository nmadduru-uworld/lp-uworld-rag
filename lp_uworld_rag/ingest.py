"""Ingest pipeline -- crawl, chunk, and content-hash delta-upsert into docs_functional/docs_technical.

Each collection lives in its own Chroma database (see StoreConfig) and gets its own independent sync
cycle here: its own client, its own delta state file, its own embed/delete calls. Only discovery (the
crawl + chunk pass) is shared -- a page's resulting nodes land in exactly one collection, so splitting
the sync afterward doesn't need any cross-collection coordination.

Delta is driven by comparing each page's resulting node id(s) (a content hash -- see chunker.py)
against what was persisted last run, not by trusting Confluence's ``version`` number alone: that
catches a cross-referenced field changing (e.g. a controller's ``used_by`` shifting because an
endpoint elsewhere was added or reassigned) even when the controller's own page didn't change.
``version``/``parentId`` are still tracked in each collection's own state file for observability
(logging what changed), per the plan's "version-tracked delta" framing.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from . import confluence_reader
from .config import RagConfig
from .retrieval.docs_index import _abs, _docstore_dir, get_embed_model


def _collection_for(node) -> str:
    return "functional" if node.metadata.get("doc_type") == "feature-hub" else "technical"


def _state_path(cfg: RagConfig, collection: str) -> str:
    return _abs(cfg, os.path.join(cfg.persist_dir(collection), ".ingest_state.json"))


def _load_state(cfg: RagConfig, collection: str) -> dict:
    path = Path(_state_path(cfg, collection))
    if not path.exists():
        return {"pages": {}, "nodeIds": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(cfg: RagConfig, collection: str, state: dict) -> None:
    path = Path(_state_path(cfg, collection))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _embed_and_upsert(cfg: RagConfig, nodes: list, collection_obj, collection: str) -> None:
    if not nodes:
        return
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    embed = get_embed_model(cfg, collection)
    vstore = ChromaVectorStore(chroma_collection=collection_obj)
    storage = StorageContext.from_defaults(vector_store=vstore)
    VectorStoreIndex(nodes, storage_context=storage, embed_model=embed, show_progress=True)


def _sync_collection(cfg: RagConfig, collection: str, collection_pages: list[dict],
                      page_nodes: dict[str, list], full: bool) -> int:
    """Delta-sync one collection against its own Chroma client and its own state file -- entirely
    independent of the other collection's sync. Returns the collection's final chunk count. The
    delta mechanics (skip-unchanged/prune/docstore-rebuild) live in ``store_sync``, shared with the
    code ingest; only this collection's state-file schema (the extra ``pages`` version map) is local."""
    from .common import store_sync
    from .retrieval.docs_index import get_chroma_client, get_collection

    state = {"pages": {}, "nodeIds": {}} if full else _load_state(cfg, collection)
    prev_node_ids: dict[str, list[str]] = state.get("nodeIds", {})

    client = get_chroma_client(cfg, collection)
    if full:
        store_sync.drop_collection(client, cfg.collection_name(collection))
    coll = get_collection(cfg, client, collection)

    to_embed, new_node_ids, removed_page_ids = store_sync.delta_sync(
        coll, page_nodes, prev_node_ids, full)

    print(f"[ingest] {collection}: embedding {len(to_embed)} new/changed chunks "
          f"({len(removed_page_ids)} page(s) removed) …")
    _embed_and_upsert(cfg, to_embed, coll, collection)
    store_sync.persist_docstore([n for nodes in page_nodes.values() for n in nodes],
                                 Path(_docstore_dir(cfg, collection)))

    _save_state(cfg, collection, {
        "pages": {p["id"]: {"version": p["version"], "parentId": p.get("parentId")}
                  for p in collection_pages},
        "nodeIds": new_node_ids,
    })
    return coll.count()


def run_ingest(cfg: RagConfig, full: bool = False, strict: bool = False) -> bool:
    """Crawl/chunk/embed/upsert, then validate the result. Returns True unless ``strict`` is set
    and the post-ingest metadata-integrity check (the same one ``eval`` tier 1 runs) found an
    error-severity finding -- bad metadata (e.g. the Order(api) repo-naming/controller-field
    corruption this validator originally caught) used to only surface whenever someone happened to
    run ``eval``; gating ``ingest`` on it means a broken page can't reach the live index silently."""
    print("[ingest] crawling …")
    pages = confluence_reader.crawl(cfg)
    pages_by_id = {p["id"]: p for p in pages}
    print(f"[ingest] {len(pages)} pages crawled")

    print("[ingest] fetching bodies + chunking …")
    nodes = confluence_reader.read_all(cfg, pages=pages)

    by_page_by_collection: dict[str, dict[str, list]] = {"functional": defaultdict(list),
                                                          "technical": defaultdict(list)}
    for n in nodes:
        by_page_by_collection[_collection_for(n)][n.metadata["page_id"]].append(n)

    counts = {}
    for collection, page_nodes in by_page_by_collection.items():
        collection_pages = [pages_by_id[pid] for pid in page_nodes if pid in pages_by_id]
        counts[collection] = _sync_collection(cfg, collection, collection_pages, dict(page_nodes), full)

    print(f"[ingest] done. docs_functional={counts['functional']} docs_technical={counts['technical']} chunks.")

    from .eval import check_metadata
    findings = check_metadata(cfg)
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warn"]
    if findings:
        print(f"\n[ingest] metadata validation: {len(errors)} error(s), {len(warnings)} warning(s)")
        for f in findings:
            print(f"  [{f.severity.upper():5}] {f.where}: {f.message}")
    else:
        print("\n[ingest] metadata validation: no issues found")

    if strict and errors:
        print(f"\n[ingest] --strict: failing due to {len(errors)} error-severity finding(s) above")
        return False
    return True
