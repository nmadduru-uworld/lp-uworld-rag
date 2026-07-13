"""Page/section -> one TextNode.

WS4 (token-economy plan): endpoint and db-collection pages -- the two doc_types with rich, multi-
``## heading`` bodies (Contract/Business rules/Flow/Gotchas; Document shape/Indexes/Constraints/
Sample document) -- are now split section-by-section (see confluence_reader._split_page_sections),
small-to-big like the code side's overview+per-method chunks: a short overview/Purpose parent plus
one leaf node per remaining section, linked via ``parent_id``. Every other doc_type (feature-hub,
technical-hub, controller-context, repo-registry) still gets one whole-page/whole-section node --
they're already thin and single-purpose by authoring convention, so splitting them further would
orphan content from its own context for no token benefit.

A defensive size ceiling still applies -- real pages rarely approach it, but nothing here enforces
thinness at ingest time, so a page that grows unexpectedly large gets truncated rather than blowing up
an embedding with unrelated tail content.
"""
from __future__ import annotations

import hashlib
import json

# 8000 (~2048 tokens, nomic-embed-text-v1.5's context window) -- raised from 3000, which was cutting
# real endpoint pages mid-Gotchas (confirmed live on ep::reports::GetFacultyLedGroupPerformance:
# its highest-value section, loosely-typed serialization warnings, was silently lost from the
# index while the Confluence page itself was fine). Still a defensive ceiling only, not a target --
# eval.py's oversize-chunk check flags any node approaching it so truncation stays visible instead
# of silent.
MAX_CHUNK_CHARS = 8000

# Chroma metadata values must be scalar (str/int/float/None) -- list-valued tags (`feature`,
# `consumed_by`, `used_by`, `db_ids`) get JSON-encoded to a string here and decoded back via
# `decode_list_field` wherever they're read from a persisted node (see index.py). `db_ids` is an
# endpoint's forward reference to the data stores it reads -- the dependency-direction-correct
# replacement for `db-collection`'s `consumed_by` (a reverse reference); `consumed_by` stays in
# this set for backward compatibility even after Confluence authoring drops it.
LIST_FIELDS = frozenset({"feature", "consumed_by", "used_by", "db_ids"})


def decode_list_field(value) -> list:
    """Inverse of the JSON-encoding this module applies to list-valued metadata fields."""
    if not value:
        return []
    if isinstance(value, list):
        return value
    return json.loads(value)


def _cap(text: str) -> str:
    if len(text) <= MAX_CHUNK_CHARS:
        return text
    return text[:MAX_CHUNK_CHARS].rstrip() + "\n\n_[... truncated]_"


def _compute_id(doc_type: str, key: str, text: str) -> str:
    return hashlib.sha256(f"{doc_type}|{key}|{text}".encode("utf-8")).hexdigest()[:24]


def compute_id(doc_type: str, key: str, title: str, text: str) -> str:
    """Public wrapper of the id scheme ``build_node`` uses internally (same ``_cap`` + hash), so a
    caller (the section-splitter) can pre-compute a parent/overview node's id BEFORE its full
    ``TextNode`` is built, to hand it to a child section as that child's ``parent_id`` -- the same
    small-to-big need ReportsRagPy's code chunker has for its own overview/method split. Must be
    called with the exact same ``doc_type``/``key``/``title``/``text`` the later ``build_node``
    call for that same node uses, or the ids won't match."""
    content = _cap(f"# {title}\n\n{text}".strip())
    return _compute_id(doc_type, key, content)


def build_node(text: str, title: str, key: str, doc_type: str, metadata: dict):
    """Build one finalized TextNode: stable id, Chroma-safe metadata, metadata excluded from embedding.

    ``key`` is a Confluence page id (or ``page_id::ControllerName`` for a controller-context section
    split out of a repo-registry page's body). ``metadata`` must already carry every tag field for
    this ``doc_type`` (see confluence_reader.py) -- this function does not derive tags from content.
    """
    from llama_index.core.schema import TextNode

    content = _cap(f"# {title}\n\n{text}".strip())
    md = {}
    for k, v in metadata.items():
        if k in LIST_FIELDS:
            md[k] = json.dumps(v or [])
        else:
            md[k] = "" if v is None else v
    md["doc_type"] = doc_type
    md["title"] = title
    keys = list(md.keys())
    node_id = _compute_id(doc_type, key, content)
    return TextNode(
        text=content,
        id_=node_id,
        metadata=md,
        excluded_embed_metadata_keys=keys,
        excluded_llm_metadata_keys=keys,
    )
