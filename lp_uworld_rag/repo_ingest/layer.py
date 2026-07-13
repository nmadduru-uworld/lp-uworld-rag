"""Path-based layer classification -- language-agnostic, shared by every chunker.

Ported verbatim from ``reports_rag/code_reader.py``'s ``infer_layer``: a chunk's ``layer`` is
derived purely from its file path (which conventional folder it lives under), never from the AST,
so the same rule serves C#, and any future language, without change. The ``layer`` value feeds the
retrieval engine's quota buckets via the manifest's ``categoryMap`` (see ``direct_index._category``).
"""
from __future__ import annotations

LAYER_BY_DIR = {
    "controllers": "controller",
    "services": "service",
    "repositories": "repository",
    "entities": "entity",
}


def infer_layer(rel_path: str) -> str:
    """Map a repo-relative file path to a layer (controller/service/repository/entity/utility/other)."""
    lowered = rel_path.replace("\\", "/").lower()
    for key, layer in LAYER_BY_DIR.items():
        if f"/{key}/" in lowered:
            return layer
    if "/entities/" in lowered or lowered.endswith("entity.cs"):
        return "entity"
    if ".utility/" in lowered or "/utility/" in lowered:
        return "utility"
    return "other"
