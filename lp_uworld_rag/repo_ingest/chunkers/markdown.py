"""Markdown/rules chunker -- ``CLAUDE.md`` + ``<rulesDir>/*.md``, ported from
``reports_rag/ingest._rule_nodes``. Emits ``source_type="rule"`` chunks with no ``layer`` (a rule
is quota-exempt; ``direct_index`` buckets a null layer to "other"). Always run by the pipeline
alongside the language chunker, so a repo's authored rules are searchable next to its code.
"""
from __future__ import annotations

import glob as _glob
import os
from pathlib import Path

from .base import CodeChunk

# Defensive ceiling for prose chunks (MarkdownNodeParser has no chunk_size of its own). Distinct
# from code's cap -- for prose the binding constraint is retrieval precision, not token budget.
MAX_DOC_CHARS = 3000


def _is_heading_only(text: str) -> bool:
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    return bool(lines) and all(ln.strip().startswith("#") for ln in lines)


def _cap_doc_chunk(text: str) -> str:
    if len(text) <= MAX_DOC_CHARS:
        return text
    return text[:MAX_DOC_CHARS].rstrip() + "\n\n_[... truncated]_"


class MarkdownChunker:
    def __init__(self, root: Path, rules_dir: str | None, claude_md: str | None) -> None:
        self.root = Path(root)
        self.rules_dir = rules_dir
        self.claude_md = claude_md

    def _files(self) -> list[str]:
        files: list[str] = []
        if self.claude_md:
            claude = str((self.root / self.claude_md).resolve())
            if os.path.exists(claude):
                files.append(claude)
        if self.rules_dir:
            files += _glob.glob(str((self.root / self.rules_dir).resolve() / "*.md"))
        return files

    def read_all(self) -> list[CodeChunk]:
        files = self._files()
        if not files:
            return []
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import MarkdownNodeParser

        docs = SimpleDirectoryReader(input_files=files).load_data()
        parser = MarkdownNodeParser()
        chunks: list[CodeChunk] = []
        for d in docs:
            rel = os.path.relpath(d.metadata.get("file_path", ""), self.root).replace("\\", "/")
            for n in parser.get_nodes_from_documents([d]):
                content = n.get_content()
                if _is_heading_only(content):
                    continue
                content = _cap_doc_chunk(content)
                hp = (n.metadata or {}).get("header_path")
                heading = f"{rel} > {hp}" if hp and hp != "/" else rel
                chunks.append(CodeChunk(heading=heading, content=content, file_path=rel,
                                        layer=None, source_type="rule"))
        return chunks


class MarkdownChunkerFactory:
    """Not language-keyed -- the pipeline always constructs this with the spec's rulesDir/claudeMd."""

    def create(self, root: Path, source_dirs: list[str], exclude: list[str], **opts) -> MarkdownChunker:
        return MarkdownChunker(root, opts.get("rules_dir"), opts.get("claude_md"))
