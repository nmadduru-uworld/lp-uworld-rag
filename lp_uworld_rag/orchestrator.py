"""Docs -> route -> per-repo code retrieval.

Sits on top of ``index.query()`` (unchanged) and :class:`repo_registry.RepoRegistry` (this
project's only client of another repo's code-RAG MCP server) to answer a question with both what
the shared Confluence knowledge base says AND the actual code it's describing.

Routing derives repo keys purely from the stable-id naming convention already baked into every
endpoint/controller doc chunk and its citations -- ``ep::<repo>::<op>`` / ``ctrl::<repo>::<name>``
-- never from intent classification (no bugfix/enhancement/new-feature branch anywhere in here).
A query whose docs cite no registered repo (or cite one that isn't registered) falls back to
querying every registered repo with the raw question text, flagged via ``fallback_used`` -- never
a silent empty result.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .config import RagConfig
from .index import COLLECTIONS, RetrieverCache, _normalize_scores, build_retriever_cache
from .index import query as index_query
from .repo_registry import RepoRegistry

_STABLE_ID_RE = re.compile(r"^(ep|ctrl)::([^:]+)::")
_FILE_LINE_RE = re.compile(r"\b([\w.]+\.cs):(\d+)\b")


def _repo_from_id(id_: str | None) -> str | None:
    if not id_:
        return None
    m = _STABLE_ID_RE.match(id_)
    return m.group(2) if m else None


def _derive_hints(chunk: dict) -> dict:
    """Per-chunk file/symbol hints for the code stage: ``File.cs:line`` citations found in the doc
    text, plus the endpoint/controller name segment already carried in its own stable id."""
    files = {m.group(1) for m in _FILE_LINE_RE.finditer(chunk.get("content") or "")}
    symbols = set()
    id_ = chunk.get("id") or ""
    if _STABLE_ID_RE.match(id_):
        symbols.add(id_.split("::")[-1])
    return {"files": sorted(files), "symbols": sorted(symbols)}


def _route(registry: RepoRegistry, docs: dict, requested_repos: list[str] | None) -> dict:
    registered = set(registry.repo_keys())

    if requested_repos is not None:
        repos = sorted(set(requested_repos) & registered)
        return {
            "repos": repos or sorted(registered),
            "derived_from": list(requested_repos),
            "fallback_used": not repos,
            "hints": {},
        }

    derived: set[str] = set()
    hints_by_repo: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"files": set(), "symbols": set()})
    for chunk in docs.get("chunks", []):
        chunk_ids = [chunk.get("id"), *(c.get("id") for c in (chunk.get("citations") or []))]
        chunk_repos = {r for r in (_repo_from_id(cid) for cid in chunk_ids) if r}
        derived |= chunk_repos
        if not chunk_repos:
            continue
        chunk_hints = _derive_hints(chunk)
        for r in chunk_repos:
            hints_by_repo[r]["files"].update(chunk_hints["files"])
            hints_by_repo[r]["symbols"].update(chunk_hints["symbols"])

    routed = sorted(derived & registered)
    fallback = not routed
    repos = routed if routed else sorted(registered)
    hints = {
        r: {"files": sorted(hints_by_repo[r]["files"]), "symbols": sorted(hints_by_repo[r]["symbols"])}
        for r in repos if r in hints_by_repo
    }
    return {"repos": repos, "derived_from": sorted(derived), "fallback_used": fallback, "hints": hints}


class _ScoreProxy:
    """Adapts a ``{"score": float, ...}`` code-chunk dict to the ``.score`` get/set interface
    ``index._normalize_scores`` expects, so a multi-repo merge can reuse it instead of
    reimplementing the same min-max normalization for a second, unrelated pool shape."""

    __slots__ = ("d",)

    def __init__(self, d: dict):
        self.d = d

    @property
    def score(self):
        return self.d.get("score")

    @score.setter
    def score(self, value):
        self.d["score"] = value


def deep_query(cfg: RagConfig, registry: RepoRegistry, question: str, top_k_docs: int | None = None,
                top_k_code: int | None = None, repos: list[str] | None = None,
                cache: RetrieverCache | None = None) -> dict:
    """Docs stage (existing ``index.query()``, both collections) -> route -> code stage (one
    ``registry.query()`` per routed repo) -> assemble. ``repos`` lets a caller pin the routing
    explicitly (still validated against the registry, still falls back on an empty intersection)
    instead of deriving it from the doc hits."""
    cache = cache or build_retriever_cache(cfg)
    docs = index_query(cfg, question, top_k=top_k_docs, cache=cache)
    routing = _route(registry, docs, repos)

    code: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for repo_key in routing["repos"]:
        hints = routing["hints"].get(repo_key) or {}
        file_hints = hints.get("files") or None
        try:
            result = registry.query(repo_key, question, top_k=top_k_code, file_hints=file_hints)
        except Exception as exc:
            errors[repo_key] = str(exc)
            continue
        code[repo_key] = [c.model_dump() for c in result.chunks]

    if len(code) > 1:
        # Same reasoning as docs' cross-collection merge (index._normalize_scores): a repo whose
        # code-RAG pool happens to be small shouldn't have every result look artificially strong
        # next to a repo with a bigger candidate pool purely from pool-size, not relevance.
        pooled = [_ScoreProxy(c) for chunks in code.values() for c in chunks]
        _normalize_scores(pooled)

    out = {"docs": docs, "routing": routing, "code": code}
    if errors:
        out["code_errors"] = errors
    return out


def query_code(registry: RepoRegistry, question: str, repo: str | None = None,
                file_hints: list[str] | None = None, top_k: int | None = None) -> dict:
    """Thin passthrough to one or every registered repo's code-RAG, no docs stage."""
    repos = [repo] if repo else registry.repo_keys()
    code: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for repo_key in repos:
        try:
            result = registry.query(repo_key, question, top_k=top_k, file_hints=file_hints)
        except Exception as exc:
            errors[repo_key] = str(exc)
            continue
        code[repo_key] = [c.model_dump() for c in result.chunks]

    if len(code) > 1:
        pooled = [_ScoreProxy(c) for chunks in code.values() for c in chunks]
        _normalize_scores(pooled)

    out = {"code": code}
    if errors:
        out["errors"] = errors
    return out


def validate_citations(cfg: RagConfig, registry: RepoRegistry, cache: RetrieverCache | None = None) -> list[dict]:
    """Every ``File.cs:line`` citation in an ingested doc chunk, checked two ways: the file must
    exist under that repo's manifest ``repoRoot``, and -- when the repo's own code-RAG returns
    ``startLine``/``endLine`` -- the cited line must fall inside a chunk that RAG actually returns
    for that file. Keeps citations honest as a fact-check pass without turning them into a hard
    filter anywhere in normal retrieval (guardrail C1)."""
    cache = cache or build_retriever_cache(cfg)
    stale: list[dict] = []

    for collection in COLLECTIONS:
        docstore = cache.docstores[collection]
        if docstore is None:
            continue
        for doc in docstore.docs.values():
            md = doc.metadata or {}
            repo_key = _repo_from_id(md.get("endpoint_id")) or _repo_from_id(md.get("controller_id"))
            if not repo_key or repo_key not in registry.repo_keys():
                continue

            manifest = registry.manifests[repo_key]
            repo_root = manifest.resolved_repo_root()
            content = doc.get_content()
            citations = set(_FILE_LINE_RE.findall(content))
            if not citations:
                continue

            probed: dict | None = None  # lazily fetched, shared across this page's citations
            for file_name, line_str in citations:
                line = int(line_str)
                where = {
                    "page_title": md.get("title"), "page_id": md.get("page_id"),
                    "repo": repo_key, "file": file_name, "line": line,
                }
                if not list(repo_root.rglob(file_name)):
                    stale.append({**where, "reason": "file not found under repoRoot"})
                    continue

                if probed is None:
                    try:
                        probed = registry.query(repo_key, file_name, top_k=5, file_hints=[file_name])
                    except Exception:
                        probed = False  # can't probe right now -- file-existence check above still stands
                if not probed:
                    continue
                ranged = [
                    c for c in probed.chunks
                    if Path(c.filePath).name == file_name and c.startLine is not None and c.endLine is not None
                ]
                if ranged and not any(c.startLine <= line <= c.endLine for c in ranged):
                    stale.append({**where, "reason": "line falls outside every chunk this repo's RAG returns for this file"})

    return stale
