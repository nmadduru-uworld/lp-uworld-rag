"""Generic language chunker -- LlamaIndex ``CodeSplitter`` (tree-sitter under the hood) for any
language we don't have a bespoke chunker for. Splits on real functions/classes rather than character
windows, applies path-based :func:`infer_layer`, and carries 1-based line spans. It does NOT produce
the rich type-context / overview-parent metadata the C# chunker does -- those parent fields are
optional in the contract, so a generic-chunked repo simply has none.

Deferred dependency: ``CodeSplitter`` resolves a language grammar via ``tree_sitter_language_pack``,
which is intentionally NOT a pinned dependency (see pyproject). This chunker raises a clear install
hint if it's used before that package is present -- so the seam exists now and is one ``pip install``
away from working when a non-C# repo actually onboards.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..layer import infer_layer
from .base import CodeChunk

# Map a repo's declared language to (CodeSplitter language string, file glob extensions).
_LANG = {
    "python": ("python", ("*.py",)),
    "typescript": ("typescript", ("*.ts",)),
    "javascript": ("javascript", ("*.js",)),
    "java": ("java", ("*.java",)),
    "go": ("go", ("*.go",)),
    "ruby": ("ruby", ("*.rb",)),
    "rust": ("rust", ("*.rs",)),
    "cpp": ("cpp", ("*.cpp", "*.cc", "*.h", "*.hpp")),
}


class GenericCodeReader:
    def __init__(self, root: Path, source_dirs: list[str], exclude: list[str], language: str) -> None:
        self.root = Path(root)
        self.source_dirs = source_dirs
        self.exclude = exclude
        if language not in _LANG:
            raise ValueError(
                f"generic chunker has no mapping for language {language!r}; known: {sorted(_LANG)} "
                f"(or write a bespoke chunker and register it)"
            )
        self.ts_language, self.globs = _LANG[language]

    def _splitter(self):
        try:
            from llama_index.core.node_parser import CodeSplitter
            import tree_sitter_language_pack  # noqa: F401 -- presence check
        except ImportError as exc:
            raise RuntimeError(
                "generic (CodeSplitter) chunker needs 'tree-sitter-language-pack' -- "
                "run `pip install tree-sitter-language-pack` (deferred dependency; see "
                "docs/repo-rag-contract.md). C# does not need it."
            ) from exc
        return CodeSplitter(language=self.ts_language)

    def iter_files(self):
        for rel_dir in self.source_dirs:
            base = self.root / rel_dir
            if not base.exists():
                continue
            for pattern in self.globs:
                for path in base.rglob(pattern):
                    norm = str(path).replace("\\", "/")
                    if any(ex in norm for ex in self.exclude):
                        continue
                    yield path

    def read_all(self) -> list[CodeChunk]:
        from llama_index.core import Document

        splitter = self._splitter()
        chunks: list[CodeChunk] = []
        for path in self.iter_files():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = os.path.relpath(path, self.root).replace("\\", "/")
            layer = infer_layer(rel)
            nodes = splitter.get_nodes_from_documents([Document(text=text)])
            for i, n in enumerate(nodes):
                heading = f"{rel}::chunk{i}"
                chunks.append(CodeChunk(heading=heading, content=n.get_content(), file_path=rel,
                                        layer=layer, source_type="code"))
        return chunks


class GenericChunkerFactory:
    def create(self, root: Path, source_dirs: list[str], exclude: list[str], **opts) -> GenericCodeReader:
        language = opts.get("language")
        if not language:
            raise ValueError("generic chunker requires a 'language' option")
        return GenericCodeReader(root, source_dirs, exclude, language)
