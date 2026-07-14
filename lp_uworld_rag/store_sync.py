"""Ingest-time Chroma sync helpers shared by the docs ingest (``ingest.py``) and the code ingest
(``repo_ingest/pipeline.py``). Both do the same thing to a Chroma collection: content-hash **delta**
against what was persisted last run (skip unchanged, re-embed changed, prune removed), a full-rebuild
of the BM25 docstore, and swallow-on-miss deletes. Only the *key* differs (a docs page_id vs a code
file_path) and the state-file *schema* differs (kept in each caller); the mechanics live here once.

Node ids are content hashes computed upstream in the chunkers, so nothing here is frozen -- this is
pure ingest plumbing, no effect on query output.
"""
from __future__ import annotations

from pathlib import Path


def drop_collection(client, name: str) -> None:
    """Delete a collection if it exists; a missing collection is not an error (fresh --full run)."""
    try:
        client.delete_collection(name)
    except Exception:
        pass


def safe_delete(coll, ids: list[str]) -> None:
    """Delete ids from a collection, tolerating ids that aren't present (already-pruned/stale)."""
    if not ids:
        return
    try:
        coll.delete(ids=ids)
    except Exception:
        pass


def delta_sync(coll, items_by_key: dict, prev_ids: dict[str, list[str]], full: bool):
    """Content-hash delta of ``items_by_key`` (key -> list of nodes) against ``prev_ids`` (key ->
    that key's node ids from last run), mutating ``coll`` in place.

    For each key: record its current sorted node ids; if not ``full`` and they equal last run's,
    skip it (no re-embed); otherwise delete the key's old ids and queue its nodes for embedding.
    Then prune every key that disappeared this run. Returns ``(to_embed, new_ids, removed)`` -- the
    caller embeds ``to_embed``, persists ``new_ids`` to its own state file, and may log ``removed``.
    """
    new_ids: dict[str, list[str]] = {}
    to_embed: list = []
    for key, nodes in items_by_key.items():
        cur = sorted(n.id_ for n in nodes)
        new_ids[key] = cur
        if not full and prev_ids.get(key) == cur:
            continue  # content unchanged -- skip re-embedding
        safe_delete(coll, prev_ids.get(key, []))
        to_embed.extend(nodes)

    removed = set(prev_ids) - set(new_ids)
    for key in removed:
        safe_delete(coll, prev_ids[key])
    return to_embed, new_ids, removed


def persist_docstore(nodes: list, docstore_dir: Path) -> None:
    """Full rebuild each run -- ``nodes`` is the complete current corpus, so BM25's docstore is
    simply the current truth (not patched incrementally). Writes ``<docstore_dir>/docstore.json``."""
    from llama_index.core.storage.docstore import SimpleDocumentStore

    ds = SimpleDocumentStore()
    ds.add_documents(nodes)
    docstore_dir.mkdir(parents=True, exist_ok=True)
    ds.persist(str(docstore_dir / "docstore.json"))
