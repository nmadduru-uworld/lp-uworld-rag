"""Shared L1-override loader -- the one ``importlib`` resolver behind both per-repo override seams:
the chunker override (``repo_ingest/chunkers``) and the retriever override (``repo_registry``). Both
accept the same ``spec`` shape and differ only in what they validate on the loaded factory, so the
resolution itself lives here once.

``spec`` is either a ``.py`` file path, or a dotted module path with an optional ``:attr`` (defaults
to a module-level ``factory`` or ``get_factory()``). A file path is loaded off lp-uworld-rag's own
venv via importlib -- never an in-repo tool. ``kind`` ("chunker"/"retriever") only labels errors and
the synthetic module name. The returned factory is instantiated if it's a class; the *caller* does
its own post-load check (a chunker factory needs ``.create``; a retriever factory's product needs
``.query``).
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path


def load_factory(spec: str, kind: str):
    attr = None
    if ":" in spec and not Path(spec).exists():
        spec, attr = spec.rsplit(":", 1)

    if spec.endswith(".py") or ("/" in spec) or ("\\" in spec):
        path = Path(spec).resolve()
        if not path.exists():
            raise RuntimeError(f"{kind} override file not found: {path}")
        mod_spec = importlib.util.spec_from_file_location(f"_repo_{kind}_{path.stem}", path)
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
            f"{kind} override {spec!r} exposes neither a module-level 'factory' nor 'get_factory()'"
        )
    return factory() if isinstance(factory, type) else factory
