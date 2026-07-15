"""Retrieval layer -- the query-time engines that build on ``common``: ``docs`` (the two
functional/technical Confluence collections, with citation resolve) and ``code`` (a repo's own
"index"-mode Chroma store), plus the ``orchestrator`` that chains docs -> route -> per-repo code."""
