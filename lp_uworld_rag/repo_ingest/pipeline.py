"""The CodeIngestor: read a repo checkout -> chunks -> TextNodes -> Chroma + docstore, with a
file-keyed delta-with-prune sync.

Node identity + shape are kept byte-identical to ``reports_rag/ingest.py`` so a repo migrated to this
engine produces the same chunk ids it did under its old in-tree tool (a copied Chroma dir stays valid
and a re-ingest is a no-op) -- the id is ``sha256(source_type|file_path|heading|content)[:24]`` over
the RAW chunk content (no title prefix), and all metadata is excluded from the embedding. The one
addition is a ``repo:<repoKey>`` stamp on every chunk's metadata (WS8-E); metadata isn't hashed or
embedded, so it changes neither ids nor vectors.

Delta is content-hash + prune keyed on ``file_path`` (mirroring the docs side's
``ingest._sync_collection`` keyed on page id): a file whose chunk-id set is unchanged is skipped; a
changed file's old ids are deleted before re-embedding; a file that disappeared has its ids pruned.
``full`` deletes and rebuilds the collection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..direct_index import IndexEmbedConfig
from .chunkers import get_chunker


@dataclass
class IngestJob:
    """Everything the pipeline needs for one repo, already resolved (checkout path made absolute,
    persistDir/collection derived from repoKey). Built by ``repo_ingest.spec.load_job``."""
    repo_key: str
    checkout_path: Path
    source_dirs: list[str]
    exclude: list[str]
    chunker_spec: str            # registered key ("csharp") or an L1 override path
    language: str                # passed to the generic factory via opts
    embed: IndexEmbedConfig
    persist_dir: Path
    collection: str
    rules_dir: str | None = None
    claude_md: str | None = None
    extra_opts: dict = field(default_factory=dict)


def _compute_id(source_type: str, file_path: str, heading: str, text: str) -> str:
    return hashlib.sha256(
        f"{source_type}|{file_path}|{heading}|{text}".encode("utf-8")
    ).hexdigest()[:24]


def _node(text: str, md: dict, source_type: str):
    from llama_index.core.schema import TextNode

    md = {k: ("" if v is None else v) for k, v in md.items()}
    md["source_type"] = source_type
    keys = list(md.keys())
    nid = _compute_id(source_type, md.get("file_path", ""), md.get("heading", ""), text)
    return TextNode(
        text=text, id_=nid, metadata=md,
        excluded_embed_metadata_keys=keys, excluded_llm_metadata_keys=keys,
    )


def _build_nodes(job: IngestJob) -> list:
    """Language chunker + always the markdown chunker -> finalized TextNodes carrying the contract
    metadata (heading/source_type/layer/file_path/[start_line/end_line]/[parent_id]/[parent_summary])
    plus the repo stamp."""
    opts = {"language": job.language, "rules_dir": job.rules_dir, "claude_md": job.claude_md,
            **job.extra_opts}
    code_chunker = get_chunker(job.chunker_spec, job.checkout_path, job.source_dirs, job.exclude, **opts)
    md_chunker = get_chunker("markdown", job.checkout_path, job.source_dirs, job.exclude, **opts)
    raw_chunks = list(code_chunker.read_all()) + list(md_chunker.read_all())

    # Precompute every chunk's id so a leaf's parent_heading resolves to the parent's id BEFORE nodes
    # are built (so parent_id is part of the excluded-from-embedding metadata from the start).
    id_by_key = {
        (c.file_path, c.heading): _compute_id(c.source_type, c.file_path, c.heading, c.content)
        for c in raw_chunks
    }

    nodes = []
    for c in raw_chunks:
        md = {"heading": c.heading, "file_path": c.file_path, "repo": job.repo_key}
        if c.layer is not None:
            md["layer"] = c.layer
        if c.start_line:
            md["start_line"] = c.start_line
            md["end_line"] = c.end_line
        if c.metadata.get("parent_summary"):
            md["parent_summary"] = c.metadata["parent_summary"]
        if c.parent_heading:
            parent_id = id_by_key.get((c.file_path, c.parent_heading))
            if parent_id:
                md["parent_id"] = parent_id
        nodes.append(_node(c.content, md, source_type=c.source_type))
    return nodes


# -- delta state ----------------------------------------------------------------

def _state_path(persist_dir: Path) -> Path:
    return persist_dir / ".ingest_state.json"


def _load_state(persist_dir: Path) -> dict:
    p = _state_path(persist_dir)
    if not p.exists():
        return {"files": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"files": {}}


def _save_state(persist_dir: Path, files: dict[str, list[str]]) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    _state_path(persist_dir).write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")


def run_code_ingest(job: IngestJob, full: bool = False) -> int:
    """Ingest one repo. Returns the collection's final chunk count."""
    from collections import Counter
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from .. import store_sync
    from ..direct_index import get_embed_model
    from ..retrieval_engine import open_chroma_client

    print(f"[ingest-code] {job.repo_key}: building nodes from {job.checkout_path} …")
    nodes = _build_nodes(job)
    by = Counter((n.metadata.get("layer") or "-", n.metadata["source_type"]) for n in nodes)
    print(f"[ingest-code] {len(nodes)} nodes: " + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))

    nodes_by_file: dict[str, list] = {}
    for n in nodes:
        nodes_by_file.setdefault(n.metadata.get("file_path", ""), []).append(n)

    job.persist_dir.mkdir(parents=True, exist_ok=True)
    client = open_chroma_client(str(job.persist_dir))
    if full:
        store_sync.drop_collection(client, job.collection)
    coll = client.get_or_create_collection(job.collection)

    prev = ({"files": {}} if full else _load_state(job.persist_dir)).get("files", {})
    to_embed, new_ids, removed = store_sync.delta_sync(coll, nodes_by_file, prev, full)

    print(f"[ingest-code] embedding {len(to_embed)} new/changed chunks "
          f"({len(removed)} file(s) removed) …")
    if to_embed:
        embed = get_embed_model(job.embed)
        vstore = ChromaVectorStore(chroma_collection=coll)
        storage = StorageContext.from_defaults(vector_store=vstore)
        VectorStoreIndex(to_embed, storage_context=storage, embed_model=embed, show_progress=True)

    store_sync.persist_docstore(nodes, job.persist_dir / "docstore")
    _save_state(job.persist_dir, new_ids)
    print(f"[ingest-code] done. collection '{job.collection}' now has {coll.count()} chunks.")
    return coll.count()
