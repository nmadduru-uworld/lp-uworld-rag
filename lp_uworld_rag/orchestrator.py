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


# WS1 (token-economy plan): deep_query fans out to BOTH docs and every routed repo's code in one
# call -- narrower per-stage default than either stage's own config default (5), since the two
# combine into one payload. A caller needing breadth still overrides top_k_docs/top_k_code
# explicitly; this only changes what happens when they're omitted.
_DEFAULT_TOP_K = 4

# WS2 (token-economy plan): how many of a repo's own top-ranked code chunks get their full `content`
# inlined by default; the rest come back as citations (heading/filePath/line-range/score, no body) --
# the same "cite, don't inline" rule the doc side already applies to linked/sibling nodes, extended
# to primary code hits, which up to now always inlined every hit's full body regardless of rank.
_CODE_INLINE_DEFAULT = 2


def _code_citation(chunk: dict) -> dict:
    """Strip `content` from a code-chunk dict, keeping everything needed to identify and later
    `expand_code()` it. `parentSummary` (a short class-level doc-comment, not the chunk body) stays
    -- cheap, and often enough context on its own without a full expand round-trip."""
    return {k: v for k, v in chunk.items() if k != "content"}


def _apply_code_inline_limit(chunks: list[dict], inline_top: int) -> list[dict]:
    """Keep full `content` on the first ``inline_top`` chunks (already rank-ordered by the
    underlying retriever; cross-repo score normalization rescales values but never reorders), return
    the rest as citations. ``inline_top`` <= 0 means every chunk in this repo's result is a citation."""
    return [c if i < inline_top else _code_citation(c) for i, c in enumerate(chunks)]


# WS3 (token-economy plan): cap the tail of a deep_query/query_code response -- a weak hit isn't
# worth returning at all (score floor), and the whole response shouldn't exceed a predictable token
# ceiling regardless of how many repos/chunks got routed (token budget).
_CODE_SCORE_FLOOR = 0.3
_DEFAULT_TOKEN_BUDGET = 6000

_ENC = None


def _count_tokens(text: str | None) -> int:
    """tiktoken cl100k_base -- same tokenizer choice as eval.py's WS0 accounting (not imported from
    there to avoid a circular import: eval.py imports this module for its Tier 5 orchestrator cases)."""
    global _ENC
    if not text:
        return 0
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text, disallowed_special=()))


def _apply_score_floor(chunks: list[dict], floor: float | None) -> list[dict]:
    """Drop chunks whose (already cross-repo-normalized) score is below ``floor`` -- but never down
    to zero, so a routed repo never comes back with nothing at all just because every hit happened
    to score modestly (the top hit still stands even if it's under the floor)."""
    if floor is None or len(chunks) <= 1:
        return chunks
    kept = [c for c in chunks if (c.get("score") if c.get("score") is not None else 0.0) >= floor]
    return kept if kept else chunks[:1]


def _apply_token_budget(docs_tokens: int, code: dict[str, list[dict]], budget: int | None) -> int:
    """Trim the lowest-scoring code chunks (across every routed repo combined, ascending by score)
    until the combined docs+code payload fits ``budget`` tokens or no code chunks remain. Docs
    chunks are never touched here -- their count is already capped by top_k_docs (WS1) and their
    quota/diversity guarantee (index._apply_quota) shouldn't be undermined by a blind token trim.
    Mutates ``code`` in place; returns how many chunks were dropped."""
    if budget is None:
        return 0
    flat = [(repo_key, c, _count_tokens(c.get("content")))
            for repo_key, chunks in code.items() for c in chunks]
    flat.sort(key=lambda t: t[1].get("score") if t[1].get("score") is not None else 0.0)  # weakest first

    total = docs_tokens + sum(t[2] for t in flat)
    dropped = 0
    for repo_key, victim, victim_tokens in flat:
        if total <= budget:
            break
        code[repo_key] = [c for c in code[repo_key] if c is not victim]
        total -= victim_tokens
        dropped += 1
    return dropped


def deep_query(cfg: RagConfig, registry: RepoRegistry, question: str, top_k_docs: int | None = None,
                top_k_code: int | None = None, repos: list[str] | None = None,
                inline_top: int = _CODE_INLINE_DEFAULT, score_floor: float | None = _CODE_SCORE_FLOOR,
                token_budget: int | None = _DEFAULT_TOKEN_BUDGET, include_siblings: bool = True,
                cache: RetrieverCache | None = None) -> dict:
    """Docs stage (existing ``index.query()``, both collections) -> route -> code stage (one
    ``registry.query()`` per routed repo) -> assemble. ``repos`` lets a caller pin the routing
    explicitly (still validated against the registry, still falls back on an empty intersection)
    instead of deriving it from the doc hits.

    ``top_k_docs``/``top_k_code`` default to a narrow 4 each when omitted -- retrieve narrow first;
    pass an explicit, larger value only once a first narrow call shows the answer isn't there yet.

    ``inline_top`` (WS2, cite-don't-inline for code): only the top ``inline_top`` code chunks PER
    REPO carry their full ``content``; the rest come back as citations (heading, filePath,
    startLine/endLine, score -- no body). Call ``expand_code()`` on a citation to pull its body on
    demand. Set higher (or to ``top_k_code`` to disable citation-only entirely) if you already know
    you need every routed chunk's full text.

    ``score_floor`` (WS3): a routed repo's code hits scoring below this (post cross-repo
    normalization) are dropped -- except the single best hit, which always stands even if it's
    under the floor, so a routed repo never comes back with nothing. ``None`` disables the floor.

    ``token_budget`` (WS3): a ceiling on the combined docs+code payload; if exceeded, the
    lowest-scoring code chunks (across every routed repo) are dropped -- never docs chunks, whose
    count is already capped by ``top_k_docs`` -- until it fits or none remain. A response trimmed
    this way carries an ``"omitted"`` key: ``{"count": N, "hint": "..."}``. ``None`` disables it.

    ``include_siblings`` (WS5, progressive disclosure -- exposed here for the first time; previously
    hardcoded on via an implicit default with no way to turn it off from ``deep_query``): whether a
    doc hit's linked nodes (controller, data stores, feature hub/technical hub, and -- since WS4 --
    its own section siblings: an overview cites its leaf sections, a leaf cites its overview) come
    back as lightweight `{id, title, doc_type}` citations. Left ``True`` by default DELIBERATELY,
    not flipped to ``False`` as a blanket "citations off" lean mode: citations cost only a few dozen
    tokens each (never a body), and after WS4 they're the ONLY way a hit on one section (say,
    Gotchas) reveals that its sibling sections (Contract, Flow) exist at all -- defaulting this off
    would quietly break that discovery path for a token saving too small to be worth it. Set
    ``False`` only when you already know you want the single narrowest possible response (e.g.
    chasing one specific, already-identified section) and don't need sibling discovery this call.
    """
    cache = cache or build_retriever_cache(cfg)
    top_k_docs = top_k_docs if top_k_docs is not None else _DEFAULT_TOP_K
    docs = index_query(cfg, question, top_k=top_k_docs, include_siblings=include_siblings, cache=cache)
    routing = _route(registry, docs, repos)

    code: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    effective_top_k_code = top_k_code if top_k_code is not None else _DEFAULT_TOP_K
    for repo_key in routing["repos"]:
        hints = routing["hints"].get(repo_key) or {}
        file_hints = hints.get("files") or None
        try:
            result = registry.query(repo_key, question, top_k=effective_top_k_code, file_hints=file_hints)
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

    # WS3: drop weak hits (post-normalization scores are directly comparable now) before deciding
    # what to inline -- a chunk not worth returning at all shouldn't consume an inline slot either.
    code = {repo_key: _apply_score_floor(chunks, score_floor) for repo_key, chunks in code.items()}

    # WS2: apply the inline/citation split AFTER normalization (order is unaffected by it -- only
    # score magnitude is rescaled) so a returned citation's score is the same comparable, normalized
    # value as an inlined chunk's, not the raw pre-normalization one.
    code = {repo_key: _apply_code_inline_limit(chunks, inline_top) for repo_key, chunks in code.items()}

    # WS3: hard ceiling on the assembled payload, trimming code (never docs) if still over budget
    # after the floor + inline-limit already shrank it.
    docs_tokens = sum(_count_tokens(c.get("content")) for c in docs.get("chunks", []))
    dropped = _apply_token_budget(docs_tokens, code, token_budget)

    out = {"docs": docs, "routing": routing, "code": code}
    if dropped:
        out["omitted"] = {"count": dropped,
                           "hint": "token budget reached -- lowest-scoring code chunks were dropped; "
                                   "call query_code/expand_code with a narrower question for more"}
    if errors:
        out["code_errors"] = errors
    return out


def query_code(registry: RepoRegistry, question: str, repo: str | None = None,
                file_hints: list[str] | None = None, top_k: int | None = None,
                inline_top: int = _CODE_INLINE_DEFAULT, score_floor: float | None = _CODE_SCORE_FLOOR,
                token_budget: int | None = _DEFAULT_TOKEN_BUDGET) -> dict:
    """Thin passthrough to one or every registered repo's code-RAG, no docs stage.

    ``top_k`` defaults to a narrow 4 per repo when omitted (WS1) -- widen explicitly if a first
    narrow call doesn't surface the answer, especially when ``repo`` is omitted and this fans out
    to every registered repo at once.

    ``inline_top`` (WS2): only the top ``inline_top`` chunks PER REPO carry full ``content``; the
    rest are citations. Use ``expand_code()`` to pull a citation's body on demand.

    ``score_floor``/``token_budget`` (WS3): same weak-hit floor and payload ceiling as
    ``deep_query`` (see its docstring) -- here the ceiling covers code only, since there's no docs
    stage. A trimmed response carries an ``"omitted"`` key.
    """
    repos = [repo] if repo else registry.repo_keys()
    code: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    effective_top_k = top_k if top_k is not None else _DEFAULT_TOP_K
    for repo_key in repos:
        try:
            result = registry.query(repo_key, question, top_k=effective_top_k, file_hints=file_hints)
        except Exception as exc:
            errors[repo_key] = str(exc)
            continue
        code[repo_key] = [c.model_dump() for c in result.chunks]

    if len(code) > 1:
        pooled = [_ScoreProxy(c) for chunks in code.values() for c in chunks]
        _normalize_scores(pooled)

    code = {repo_key: _apply_score_floor(chunks, score_floor) for repo_key, chunks in code.items()}
    code = {repo_key: _apply_code_inline_limit(chunks, inline_top) for repo_key, chunks in code.items()}
    dropped = _apply_token_budget(0, code, token_budget)

    out = {"code": code}
    if dropped:
        out["omitted"] = {"count": dropped,
                           "hint": "token budget reached -- lowest-scoring code chunks were dropped; "
                                   "narrow the question or pin a single repo for more"}
    if errors:
        out["errors"] = errors
    return out


def expand_code(registry: RepoRegistry, repo: str, file_path: str, heading: str) -> dict | None:
    """Pull a code citation's full body on demand -- the code-side counterpart of the doc side's
    ``expand(id)``. Code chunks carry no persisted stable id (unlike ``ep::``/``ctrl::``/``db::``),
    so this re-derives the chunk by re-querying the same repo with the citation's own heading text,
    boosted by both ``file_path`` and ``heading`` as hints -- reliable in practice because the
    heading text itself is a strong lexical+semantic match for its own chunk, but not a guaranteed
    exact re-fetch (best-effort, same spirit as file_hints being advisory everywhere else in this
    contract). Falls back to the top hit if no exact (filePath, heading) match comes back.

    Returns the matching ``CodeChunkResult`` dict (with ``content``), or ``None`` if the repo has
    no results at all for the probe.
    """
    result = registry.query(repo, heading, top_k=5, file_hints=[file_path, heading])
    if not result.chunks:
        return None
    for c in result.chunks:
        if c.filePath == file_path and c.heading == heading:
            return c.model_dump()
    return result.chunks[0].model_dump()


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
