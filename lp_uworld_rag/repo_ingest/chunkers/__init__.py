"""Chunker registry + resolver -- the per-language factory registry, and the ingestion-side override
seam (WS8-E's "chunker registry").

Resolution (``get_chunker``) is the ingestion half of the per-repo override ladder:
  * L0 -- a registered key (``"csharp"``, ``"generic"``, ``"markdown"``); repo ships no code.
  * L1 -- a dotted import path (``pkg.mod:factory`` / ``pkg.mod``) or a ``.py`` file path to a
    repo-supplied object satisfying :class:`ChunkerFactory`. Loaded via importlib; one file on
    lp-uworld-rag's venv, never an in-repo tool. The engine still owns node-building, ids, embedding,
    persistence, and delta -- the repo overrides only how source becomes chunks.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from .base import Chunker, ChunkerFactory, CodeChunk  # re-exported
from .csharp import CSharpChunkerFactory
from .generic import GenericChunkerFactory
from .markdown import MarkdownChunkerFactory

_REGISTRY: dict[str, ChunkerFactory] = {}


def register(key: str, factory: ChunkerFactory) -> None:
    _REGISTRY[key] = factory


register("csharp", CSharpChunkerFactory())
register("markdown", MarkdownChunkerFactory())
register("generic", GenericChunkerFactory())


def registered_keys() -> list[str]:
    return sorted(_REGISTRY)


def _load_override(spec: str) -> ChunkerFactory:
    """Import an L1 override. ``spec`` is either a ``.py`` file path, or a dotted module path with
    an optional ``:attr`` (defaults to a module-level ``factory`` or ``get_factory()``)."""
    attr = None
    if ":" in spec and not Path(spec).exists():
        spec, attr = spec.rsplit(":", 1)

    if spec.endswith(".py") or ("/" in spec) or ("\\" in spec):
        path = Path(spec).resolve()
        if not path.exists():
            raise RuntimeError(f"chunker override file not found: {path}")
        mod_spec = importlib.util.spec_from_file_location(f"_repo_chunker_{path.stem}", path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(spec)

    if attr:
        factory = getattr(module, attr)
    elif hasattr(module, "factory"):
        factory = module.factory
    elif hasattr(module, "get_factory"):
        factory = module.get_factory()
    else:
        raise RuntimeError(
            f"chunker override {spec!r} exposes neither a module-level 'factory' nor 'get_factory()'"
        )
    factory = factory() if isinstance(factory, type) else factory
    if not hasattr(factory, "create"):
        raise RuntimeError(f"chunker override {spec!r} is not a ChunkerFactory (no .create method)")
    return factory


def get_chunker(spec: str, root: Path, source_dirs: list[str], exclude: list[str], **opts) -> Chunker:
    """Resolve a chunker for ``spec`` (a registered key OR an L1 override path) and build it."""
    factory = _REGISTRY[spec] if spec in _REGISTRY else _load_override(spec)
    return factory.create(root, source_dirs, exclude, **opts)


__all__ = ["Chunker", "ChunkerFactory", "CodeChunk", "register", "registered_keys", "get_chunker"]
