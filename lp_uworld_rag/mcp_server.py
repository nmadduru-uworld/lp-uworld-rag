"""MCP server -- exposes ``query_rag``, ``expand``, ``rag_status``, ``deep_query``, and
``query_code`` over stdio via FastMCP.

``mcp`` is imported lazily inside :func:`serve` so the package imports without it.
"""
from __future__ import annotations

import json


def serve() -> None:
    from mcp.server.fastmcp import FastMCP

    from . import index, orchestrator
    from .config import load_config
    from .repo_registry import RepoRegistry

    cfg = load_config()
    # Built ONCE for the server's whole lifetime -- reused by every call. Without this, each call
    # would reload both collections' embedding models and re-tokenize the full BM25 corpus from
    # scratch every time.
    cache = index.build_retriever_cache(cfg)
    # Same reasoning for repo sessions: each registered repo's MCP subprocess is spawned lazily on
    # first use and then kept alive for this server's whole lifetime (see repo_registry.py).
    registry = RepoRegistry(cfg.resolved_manifest_paths())
    app = FastMCP("lp-uworld-rag")

    @app.tool()
    def query_rag(question: str, collection: str | None = None, top_k: int | None = None,
                  include_siblings: bool = True) -> str:
        """Retrieve shared functional/technical documentation (Feature Hubs, Technical Hubs,
        Endpoint Documents, Controller Context, Data Stores catalog).

        Args:
            question: natural-language question or task description.
            collection: "functional" or "technical" to scope to one; omit to query both.
            top_k: override the configured result count -- size this to your own intent (narrow for
                a bugfix chasing one endpoint, wider when scoping a new feature).
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
                    repos: list[str] | None = None) -> str:
        """Docs -> route -> code: retrieve shared Confluence documentation for a question, derive
        which registered repo(s)' own code-RAG it's actually about from the doc hits' stable ids
        (ep::<repo>::.../ctrl::<repo>::...), then retrieve from each routed repo's code-RAG too.
        If no repo can be derived (or ``repos`` names one that isn't registered), falls back to
        querying every registered repo -- never a silent empty result -- flagged via
        routing.fallback_used.

        Args:
            question: natural-language question or task description.
            top_k_docs: override the docs stage's result count (see query_rag's top_k).
            top_k_code: override the code stage's per-repo result count.
            repos: pin routing to these repo keys explicitly instead of deriving it from doc hits
                (still validated against the registry; still falls back if none are registered).

        Returns:
            JSON string: {"docs": <query_rag's own shape>, "routing": {"repos", "derived_from",
            "fallback_used", "hints"}, "code": {"<repoKey>": [{heading, filePath, layer,
            sourceType, content, score, startLine?, endLine?, parentSummary?}]}}.
        """
        return json.dumps(
            orchestrator.deep_query(cfg, registry, question, top_k_docs=top_k_docs,
                                     top_k_code=top_k_code, repos=repos, cache=cache),
            ensure_ascii=False,
        )

    @app.tool()
    def query_code(question: str, repo: str | None = None, file_hints: list[str] | None = None,
                    top_k: int | None = None) -> str:
        """Query one or every registered repo's own code-RAG directly, with no docs stage.

        Args:
            question: natural-language question or task description.
            repo: a registered repo key to scope to; omit to query every registered repo.
            file_hints: advisory file paths/symbols to rank-boost (never a hard filter -- a repo's
                code-RAG may ignore these entirely).
            top_k: override each repo's configured result count.

        Returns:
            JSON string: {"code": {"<repoKey>": [{heading, filePath, layer, sourceType, content,
            score, startLine?, endLine?, parentSummary?}]}, "errors"?: {"<repoKey>": "..."}}.
        """
        return json.dumps(
            orchestrator.query_code(registry, question, repo=repo, file_hints=file_hints, top_k=top_k),
            ensure_ascii=False,
        )

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
