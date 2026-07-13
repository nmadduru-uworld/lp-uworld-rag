"""Shared code-ingestion engine (the token-economy plan's WS8-E).

One engine, on lp-uworld-rag's single venv, ingests any repo's code by being pointed at its
checkout -- so a repo needs no in-tree RAG tool of its own. Each language/format is a
:class:`~lp_uworld_rag.repo_ingest.chunkers.base.ChunkerFactory` in a registry; a repo can override
the chunker with its own module (see ``chunkers.get_chunker``). The produced index conforms to the
same v2 retrieval contract the orchestrator already reads (see ``direct_index`` / ``repo_registry``).
"""
