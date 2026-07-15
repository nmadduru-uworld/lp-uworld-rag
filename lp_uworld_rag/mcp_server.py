"""MCP server -- exposes ``query_rag``, ``expand``, ``rag_status``, ``deep_query``, and
``query_code`` over stdio via FastMCP.

``mcp`` is imported lazily inside :func:`serve` so the package imports without it.
"""
from __future__ import annotations

import json


def serve() -> None:
    from mcp.server.fastmcp import FastMCP

    from .config import load_config
    from .repo_registry import RepoRegistry
    from .retrieval import docs_index as index, orchestrator

    cfg = load_config()
    # Built ONCE for the server's whole lifetime -- reused by every call. Without this, each call
    # would reload both collections' embedding models and re-tokenize the full BM25 corpus from
    # scratch every time.
    cache = index.build_retriever_cache(cfg)
    # Same reasoning for repo sessions: each registered repo's MCP subprocess is spawned lazily on
    # first use and then kept alive for this server's whole lifetime (see repo_registry.py).
    registry = RepoRegistry.from_config(cfg)
    app = FastMCP("lp-uworld-rag")

    @app.tool()
    def query_rag(question: str, collection: str | None = None, top_k: int | None = None,
                  include_siblings: bool = True) -> str:
        """Retrieve shared functional/technical documentation (Feature Hubs, Technical Hubs,
        Endpoint Documents, Controller Context, Data Stores catalog).

        Args:
            question: natural-language question or task description.
            collection: "functional" or "technical" to scope to one; omit to query both.
            top_k: override the configured result count (default 5). Retrieve narrow first --
                only pass a larger value once a first narrow call shows the answer isn't there yet;
                widening by default costs real tokens for results you likely won't use.
            include_siblings: when True (default), a hit's linked nodes (its controller, DB
                collections, and the Technical Hub of every feature it belongs to) are resolved and
                returned as lightweight citations. Set False for the narrowest possible response.

        Returns:
            JSON string: {"chunks": [{id, title, docType, collection, content, score,
            citations: [{id, title, docType}]}]}. A citation is an id + title only -- call
            ``expand(id)`` to fetch its full content on demand.
        """
        return json.dumps(
            index.query(cfg, question, collection=collection, top_k=top_k,
                        include_siblings=include_siblings, cache=cache),
            ensure_ascii=False,
        )

    @app.tool()
    def expand(id: str) -> str:
        """Fetch the full content of a stable id (ep::..., ctrl::..., db::...) or bare page_id
        returned as a citation by query_rag, without re-running retrieval.

        Returns:
            JSON string: list of {title, docType, collection, content} (usually one entry; a
            repo-registry page_id can map to several controller-context sections).
        """
        return json.dumps(index.expand(cfg, id, cache=cache), ensure_ascii=False)

    @app.tool()
    def deep_query(question: str, top_k_docs: int | None = None, top_k_code: int | None = None,
                    repos: list[str] | None = None, inline_top: int = 2,
                    score_floor: float | None = 0.3, token_budget: int | None = 6000,
                    include_siblings: bool = True) -> str:
        """Docs -> route -> code: retrieve shared Confluence documentation for a question, derive
        which registered repo(s)' own code-RAG it's actually about from the doc hits' stable ids
        (ep::<repo>::.../ctrl::<repo>::...), then retrieve from each routed repo's code-RAG too.
        If no repo can be derived (or ``repos`` names one that isn't registered), falls back to
        querying every registered repo -- never a silent empty result -- flagged via
        routing.fallback_used.

        Retrieve narrow first: both stages default to top_k=4 (this fans out to docs AND every
        routed repo's code in one call, so keeping each stage narrow keeps the combined payload
        small). Only pass a larger top_k_docs/top_k_code once a first narrow call shows the answer
        isn't there yet -- don't default to widening.

        Code results are citation-first (WS2): only the top ``inline_top`` chunks PER ROUTED REPO
        carry full ``content`` (default 2); the rest come back as citations -- {heading, filePath,
        layer, sourceType, score, startLine?, endLine?, parentSummary?} with no body. Call
        ``expand_code(repo, filePath, heading)`` on a citation to pull its full body on demand,
        same "cite, don't inline" rule the docs side already applies to linked nodes.

        Args:
            question: natural-language question or task description.
            top_k_docs: override the docs stage's result count (default 4).
            top_k_code: override the code stage's per-repo result count (default 4).
            repos: pin routing to these repo keys explicitly instead of deriving it from doc hits
                (still validated against the registry; still falls back if none are registered).
            inline_top: how many of each repo's top code chunks get full content inlined (default
                2). Raise it (up to top_k_code, to disable citation-only entirely) only if you
                already know you'll need every routed chunk's full text.
            score_floor: a routed repo's code hits scoring below this (default 0.3) are dropped --
                except its single best hit, which always stands. Set None to disable.
            token_budget: a ceiling (default 6000) on the combined docs+code payload; if exceeded,
                the lowest-scoring code chunks are dropped (never docs) until it fits. A trimmed
                response carries an "omitted": {"count", "hint"} key. Set None to disable.
            include_siblings: whether a doc hit's linked nodes come back as citations (default
                True) -- controller, data stores, feature/technical hub, and (since WS4) its own
                section siblings (a Gotchas hit cites its Contract/Flow siblings, and vice versa).
                Left on by default: citations cost only a few dozen tokens each and are the only
                way a one-section hit reveals its siblings exist. Set False only for the narrowest
                possible single-hit response when you don't need sibling discovery this call.

        Returns:
            JSON string: {"docs": <query_rag's own shape>, "routing": {"repos", "derived_from",
            "fallback_used", "hints"}, "code": {"<repoKey>": [{heading, filePath, layer,
            sourceType, score, startLine?, endLine?, parentSummary?, content?}]}, "omitted"?:
            {"count", "hint"}}. ``content`` is present only on a chunk within its repo's
            ``inline_top`` that survived the score floor and token budget.
        """
        return json.dumps(
            orchestrator.deep_query(cfg, registry, question, top_k_docs=top_k_docs,
                                     top_k_code=top_k_code, repos=repos, inline_top=inline_top,
                                     score_floor=score_floor, token_budget=token_budget,
                                     include_siblings=include_siblings, cache=cache),
            ensure_ascii=False,
        )

    @app.tool()
    def query_code(question: str, repo: str | None = None, file_hints: list[str] | None = None,
                    top_k: int | None = None, inline_top: int = 2,
                    score_floor: float | None = 0.3, token_budget: int | None = 6000) -> str:
        """Query one or every registered repo's own code-RAG directly, with no docs stage.

        Citation-first (WS2): only the top ``inline_top`` chunks per repo carry full ``content``
        (default 2); the rest are citations. Call ``expand_code(repo, filePath, heading)`` to pull
        a citation's full body on demand.

        Args:
            question: natural-language question or task description.
            repo: a registered repo key to scope to; omit to query every registered repo.
            file_hints: advisory file paths/symbols to rank-boost (never a hard filter -- a repo's
                code-RAG may ignore these entirely).
            top_k: override each repo's result count (default 4 -- retrieve narrow first,
                especially when ``repo`` is omitted and this fans out to every registered repo).
            inline_top: how many of each repo's top chunks get full content inlined (default 2).
            score_floor: a repo's hits scoring below this (default 0.3) are dropped -- except its
                single best hit. Set None to disable.
            token_budget: a ceiling (default 6000) on the total payload; if exceeded, the
                lowest-scoring chunks are dropped until it fits. Set None to disable.

        Returns:
            JSON string: {"code": {"<repoKey>": [{heading, filePath, layer, sourceType, score,
            startLine?, endLine?, parentSummary?, content?}]}, "omitted"?: {"count", "hint"},
            "errors"?: {"<repoKey>": "..."}}. ``content`` is present only on a chunk within its
            repo's ``inline_top`` that survived the score floor and token budget.
        """
        return json.dumps(
            orchestrator.query_code(registry, question, repo=repo, file_hints=file_hints,
                                     top_k=top_k, inline_top=inline_top, score_floor=score_floor,
                                     token_budget=token_budget),
            ensure_ascii=False,
        )

    @app.tool()
    def expand_code(repo: str, file_path: str, heading: str) -> str:
        """Fetch a code citation's full body on demand -- the code-side counterpart of ``expand``
        for a citation returned by ``deep_query``/``query_code`` without its ``content``.

        Args:
            repo: the registered repo key the citation came from.
            file_path: the citation's ``filePath``.
            heading: the citation's ``heading``.

        Returns:
            JSON string: {heading, filePath, layer, sourceType, content, score, startLine?,
            endLine?, parentSummary?}, or ``null`` if the repo has no results for this citation.
        """
        return json.dumps(orchestrator.expand_code(registry, repo, file_path, heading), ensure_ascii=False)

    @app.tool()
    def rag_status() -> str:
        """Indexed chunk counts grouped by collection and doc_type, plus every registered repo's
        code-RAG session status (plain text)."""
        lines = [index.status(cfg)]
        repo_keys = registry.repo_keys()
        if repo_keys or registry.errors:
            lines.append("")
            lines.append("registered repos:")
        for key in repo_keys:
            try:
                detail = registry.status(key)
                lines.append(f"  {key} | ok | {detail}")
            except Exception as exc:
                lines.append(f"  {key} | FAIL | {exc}")
        for path, err in registry.errors.items():
            lines.append(f"  (manifest error) {path} | {err}")
        return "\n".join(lines)

    app.run()


if __name__ == "__main__":
    serve()
