"""The chunker contract: what every language/format factory produces, and the shape the pipeline
consumes. A ``Chunker`` reads a checkout into ``CodeChunk``s; a ``ChunkerFactory`` builds a Chunker
bound to one repo's source layout. Both are ``Protocol``s so a repo's own override module (see
``chunkers.get_chunker`` L1) only has to duck-type them -- no import of this package required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class CodeChunk:
    """One retrievable unit. Mirrors ``reports_rag/code_reader.CodeChunk`` exactly so the ported C#
    chunker and any new one share a single shape.

    ``parent_heading`` links a leaf (e.g. one method) to its overview/parent chunk (small-to-big:
    the class-level doc-comment lives once on the parent, joined back at query time via ``parent_id``
    -- see ``direct_index``). ``metadata["parent_summary"]`` on an overview carries that doc-comment.
    ``start_line``/``end_line`` are the 1-based source span (0 = unknown, e.g. a markdown rule).
    """
    heading: str
    content: str
    file_path: str            # repo-relative, forward-slashed
    layer: str | None = "other"
    source_type: str = "code"
    parent_heading: str | None = None
    metadata: dict = field(default_factory=dict)
    start_line: int = 0
    end_line: int = 0


@runtime_checkable
class Chunker(Protocol):
    """Reads a repo checkout into chunks. One instance is bound to one repo's source layout."""

    def read_all(self) -> list[CodeChunk]:
        ...


@runtime_checkable
class ChunkerFactory(Protocol):
    """Builds a :class:`Chunker` for one repo. The registry (and a repo's L1 override module) hands
    back one of these; the pipeline calls ``create`` with the resolved checkout + source config."""

    def create(self, root: Path, source_dirs: list[str], exclude: list[str], **opts) -> Chunker:
        ...
