"""C# chunker -- ported wholesale from ``reports_rag/code_reader.py`` (the one proven piece of
bespoke logic in the old per-repo tool), rewired only to use the shared :class:`CodeChunk` and
:func:`infer_layer`. Behavior is byte-identical to reports' own reader, so a repo migrated to this
engine indexes the same chunks it did before.

Parses each ``.cs`` with tree-sitter and emits context-rich chunks: file/namespace header, a
type-context block (doc-comment + class/interface/record signature incl. primary-constructor DI +
class attributes + field/property signatures), and the member's own trivia + body. Small types are
kept whole; large types become an overview/parent chunk + one leaf per method/constructor.

tree-sitter is imported lazily so the package imports without the grammar installed.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from ..layer import infer_layer
from .base import ChunkerFactory, CodeChunk

# Entity type name -> literal Mongo collection name, e.g. GetCollection<FacultyLedGroupPerformance>
# ("group-performance"). Cross-referenced into the entity's OWN chunk (see read_all) so "what fields
# does the X collection store" resolves in one chunk. C#/Mongo-specific -- lives here, not in the
# language-agnostic pipeline.
_MONGO_COLLECTION_RE = re.compile(r'GetCollection<(\w+)>\(\s*"([^"]+)"\s*\)')

_TYPE_NODES = {
    "class_declaration",
    "interface_declaration",
    "struct_declaration",
    "record_declaration",
    "record_struct_declaration",
    "enum_declaration",
}
_MEMBER_NODES = {"method_declaration", "constructor_declaration"}

# A non-entity type whose full source is at or below this size is emitted whole, so tiny helper /
# utility methods keep their sibling context instead of becoming contextless fragments.
SMALL_TYPE_CHARS = 1600
# Hard cap per chunk (nomic context ~= 2048 tokens ~= ~8k chars; leave margin).
MAX_CHUNK_CHARS = 6000
# When a method chunk would overflow the cap, shrink the prepended type-context (fields/
# properties -- boilerplate) to this budget FIRST, before ever touching the method body itself.
MAX_TYPE_CTX_CHARS = 1200
# If the method body alone still overflows after shrinking context, keep this fraction as the head
# (signature + start of logic) and the rest as the tail (often the return/result construction).
HEAD_TAIL_HEAD_RATIO = 0.65


class CSharpCodeReader:
    """Parse ``.cs`` files into context-rich :class:`CodeChunk`s via tree-sitter-c-sharp."""

    def __init__(self, root: Path, source_dirs: list[str], exclude: list[str]) -> None:
        self.root = Path(root)
        self.source_dirs = source_dirs
        self.exclude = exclude
        self._parser = None

    # -- parser -----------------------------------------------------------------
    def _get_parser(self):
        if self._parser is None:
            from tree_sitter import Language, Parser
            import tree_sitter_c_sharp as tscs

            self._parser = Parser(Language(tscs.language()))
        return self._parser

    # -- file discovery ---------------------------------------------------------
    def iter_files(self):
        for rel_dir in self.source_dirs:
            base = self.root / rel_dir
            if not base.exists():
                continue
            for path in base.rglob("*.cs"):
                norm = str(path).replace("\\", "/")
                if any(ex in norm for ex in self.exclude):
                    continue
                yield path

    # -- small helpers ----------------------------------------------------------
    @staticmethod
    def _text(src: bytes, node) -> str:
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _name(node) -> str:
        n = node.child_by_field_name("name")
        return n.text.decode("utf-8", errors="replace") if n is not None else "<anon>"

    @staticmethod
    def _body(type_node):
        return type_node.child_by_field_name("body")

    @staticmethod
    def _cap(text: str) -> str:
        if len(text) <= MAX_CHUNK_CHARS:
            return text
        return text[:MAX_CHUNK_CHARS].rstrip() + "\n// [... truncated]"

    @staticmethod
    def _shrink_type_ctx(type_ctx: str, budget: int) -> str:
        """If the type-context block is too large, truncate it (boilerplate; safe to lose detail
        here before ever touching the method body, which is what actually gets edited)."""
        if len(type_ctx) <= budget:
            return type_ctx
        return type_ctx[:budget].rstrip() + "\n// [... fields/properties truncated ...]"

    @staticmethod
    def _truncate_body_head_tail(body: str, budget: int, file_path: str = "",
                                  start_line: int = 0, end_line: int = 0) -> str:
        """Keep the head (signature + start of logic) AND the tail (often the return/result
        construction) of an oversized method body, omitting the middle -- better for diagnosis than
        a blunt end-cut. The marker names exactly where to read the full method."""
        if len(body) <= budget:
            return body
        # NOTE: the marker text (incl. the em-dash) is kept byte-identical to reports_rag's original
        # code_reader.py -- it's part of the chunk content that the content-hash id is computed over,
        # so any change here would re-key every truncated chunk and break the migrated-index parity.
        loc = f" — full method at {file_path}:{start_line}-{end_line}" if file_path else ""
        marker = "\n// [... {n} chars omitted" + loc + " ...]\n"
        reserve = len(marker.format(n=len(body)))
        avail = max(budget - reserve, 200)
        head_n = int(avail * HEAD_TAIL_HEAD_RATIO)
        tail_n = avail - head_n
        omitted = len(body) - head_n - tail_n
        return body[:head_n].rstrip() + marker.format(n=omitted) + body[-tail_n:].lstrip()

    @staticmethod
    def _first_sig(text: str) -> str:
        """First signature line of a member: text up to the first '{', '=>', or newline."""
        cut = len(text)
        for token in ("{", "=>", "\n"):
            i = text.find(token)
            if i != -1:
                cut = min(cut, i)
        return text[:cut].strip()

    @staticmethod
    def _leading_trivia(lines: list[str], start_row: int) -> str:
        """Doc-comments (``///``/``/** */``) and attributes (``[...]``) above a node.

        Tolerates a single blank line between trivia blocks and tracks bracket depth so a multi-line
        attribute is captured in full rather than only its last physical line.
        """
        out: list[str] = []
        i = start_row - 1
        blank_run = 0
        bracket_depth = 0  # >0 while scanning upward through a multi-line [...] attribute
        while i >= 0:
            raw = lines[i]
            s = raw.strip()

            if bracket_depth > 0:
                out.append(raw)
                bracket_depth += s.count("]") - s.count("[")
                i -= 1
                continue

            if not s:
                blank_run += 1
                if blank_run > 1:
                    break
                out.append(raw)
                i -= 1
                continue
            blank_run = 0

            if s.startswith("///") or s.startswith("*") or s.startswith("/*") or s.endswith("*/"):
                out.append(raw)
            elif s.startswith("[") and s.endswith("]"):
                out.append(raw)
            elif s.endswith("]") and not s.startswith("["):
                out.append(raw)
                bracket_depth = max(s.count("]") - s.count("["), 1)
            else:
                break
            i -= 1
        while out and not out[-1].strip():
            out.pop()
        return "\n".join(reversed(out))

    def _namespace(self, root_node) -> str:
        for child in root_node.children:
            if child.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
                return self._name(child)
        return ""

    def _walk(self, node, path: str):
        """Yield ``(type_node, enclosing_dotted_path)`` for every type declaration in the tree."""
        for child in node.children:
            if child.type in _TYPE_NODES:
                yield child, path
                new_path = f"{path}.{self._name(child)}" if path else self._name(child)
                body = self._body(child)
                if body is not None:
                    yield from self._walk(body, new_path)
            else:
                yield from self._walk(child, path)

    # -- context blocks ---------------------------------------------------------
    def _type_declaration_line(self, type_node, src: bytes) -> str:
        body = self._body(type_node)
        end = body.start_byte if body is not None else type_node.end_byte
        return src[type_node.start_byte:end].decode("utf-8", errors="replace").strip()

    def _type_signature(self, type_node, src: bytes, lines: list[str]) -> str:
        sig = self._type_declaration_line(type_node, src)
        trivia = self._leading_trivia(lines, type_node.start_point[0])
        return "\n".join(p for p in (trivia, sig) if p)

    def _members_context(self, type_node, src: bytes) -> str:
        body = self._body(type_node)
        if body is None:
            return ""
        out: list[str] = []
        for c in body.children:
            if c.type == "field_declaration":
                out.append(self._text(src, c).strip())
            elif c.type == "property_declaration":
                out.append(self._first_sig(self._text(src, c)) + " { … }")
        return "// Fields & properties:\n" + "\n".join(out) if out else ""

    def _type_context_full(self, type_node, src: bytes, lines: list[str]) -> str:
        return "\n".join(p for p in (self._type_signature(type_node, src, lines),
                                     self._members_context(type_node, src)) if p)

    def _type_context_lean(self, type_node, src: bytes) -> str:
        return "\n".join(p for p in (self._type_declaration_line(type_node, src),
                                     self._members_context(type_node, src)) if p)

    def _assemble_method_chunk(self, header: str, type_ctx: str, trivia: str, method_text: str,
                                file_path: str = "", start_line: int = 0, end_line: int = 0) -> str:
        fixed = "\n".join(p for p in (header, trivia) if p)
        candidate = "\n".join(p for p in (fixed, type_ctx, method_text) if p)
        if len(candidate) <= MAX_CHUNK_CHARS:
            return candidate

        shrunk_ctx = self._shrink_type_ctx(type_ctx, MAX_TYPE_CTX_CHARS)
        candidate = "\n".join(p for p in (fixed, shrunk_ctx, method_text) if p)
        if len(candidate) <= MAX_CHUNK_CHARS:
            return candidate

        body_budget = MAX_CHUNK_CHARS - len("\n".join(p for p in (fixed, shrunk_ctx) if p)) - 2
        trimmed_body = self._truncate_body_head_tail(method_text, max(body_budget, 500),
                                                      file_path, start_line, end_line)
        return "\n".join(p for p in (fixed, shrunk_ctx, trimmed_body) if p)

    # -- main -------------------------------------------------------------------
    def read_file(self, path: Path) -> list[CodeChunk]:
        try:
            raw = path.read_bytes()
        except OSError:
            return []
        lines = raw.decode("utf-8", errors="replace").splitlines()
        tree = self._get_parser().parse(raw)
        rel = os.path.relpath(path, self.root).replace("\\", "/")
        ns = self._namespace(tree.root_node)
        layer = infer_layer(rel)
        header = f"// File: {rel}\n// Namespace: {ns}".rstrip()

        chunks: list[CodeChunk] = []
        for type_node, enclosing in self._walk(tree.root_node, ""):
            type_name = self._name(type_node)
            qname = f"{enclosing}.{type_name}" if enclosing else type_name
            body = self._body(type_node)
            members = [c for c in body.children if c.type in _MEMBER_NODES] if body else []
            whole = self._text(src=raw, node=type_node)

            type_start = type_node.start_point[0] + 1
            type_end = type_node.end_point[0] + 1
            if (type_node.type == "enum_declaration" or layer == "entity"
                    or not members or len(whole) <= SMALL_TYPE_CHARS):
                trivia = self._leading_trivia(lines, type_node.start_point[0])
                content = self._cap("\n".join(p for p in (header, trivia, whole) if p))
                chunks.append(CodeChunk(qname, content, rel, layer,
                                        start_line=type_start, end_line=type_end))
                continue

            type_ctx_full = self._type_context_full(type_node, raw, lines)
            type_ctx_lean = self._type_context_lean(type_node, raw)
            class_doc = self._leading_trivia(lines, type_node.start_point[0])
            sigs = [self._first_sig(self._text(raw, m)) for m in members]
            overview_heading = f"{qname} (overview)"
            overview = self._cap("\n".join(
                p for p in (header, type_ctx_full, "// Members:", *[f"  {s};" for s in sigs]) if p))
            chunks.append(CodeChunk(overview_heading, overview, rel, layer,
                                    metadata={"parent_summary": class_doc} if class_doc else {},
                                    start_line=type_start, end_line=type_end))

            for m in members:
                member_name = self._name(m)
                trivia = self._leading_trivia(lines, m.start_point[0])
                method_text = self._text(raw, m)
                m_start, m_end = m.start_point[0] + 1, m.end_point[0] + 1
                content = self._assemble_method_chunk(
                    header, type_ctx_lean, trivia, method_text,
                    file_path=rel, start_line=m_start, end_line=m_end)
                chunks.append(CodeChunk(f"{qname}::{member_name}", content, rel, layer,
                                        parent_heading=overview_heading,
                                        start_line=m_start, end_line=m_end))
        return chunks

    def read_all(self) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        for path in self.iter_files():
            chunks.extend(self.read_file(path))
        # Mongo cross-reference: annotate each entity chunk with the literal collection name found in
        # any GetCollection<Entity>("name") call across the corpus (matches reports_rag/ingest.py's
        # behavior; kept here since it's C#/Mongo-specific). Entities are whole-type leaf chunks
        # (never parents), so mutating their content doesn't affect any parent-id linkage.
        collection_by_entity: dict[str, str] = {}
        for c in chunks:
            for type_name, collection_name in _MONGO_COLLECTION_RE.findall(c.content):
                collection_by_entity.setdefault(type_name, collection_name)
        for c in chunks:
            if c.layer == "entity" and c.heading in collection_by_entity:
                c.content = f'{c.content}\n// Mongo collection: "{collection_by_entity[c.heading]}"'
        return chunks


class CSharpChunkerFactory:
    """Registry entry for ``language: "csharp"``."""

    def create(self, root: Path, source_dirs: list[str], exclude: list[str], **opts) -> CSharpCodeReader:
        return CSharpCodeReader(root, source_dirs, exclude)
