"""Token counting -- tiktoken's ``cl100k_base`` as a consistent, public proxy for Claude's token
count (Claude's own tokenizer isn't published; same choice ReportsRagPy's token_usage_bench.py
makes, so numbers are comparable across both projects).

One definition, shared by ``eval.py`` (WS0 token accounting) and ``orchestrator.py`` (WS3 token
budget). A standalone module both import is what lets there be a single copy: previously the two
lived apart only to avoid a circular import (``eval.py`` imports ``orchestrator`` for its Tier 5
cases, so ``orchestrator`` couldn't import back from ``eval``). The encoder is built once, lazily,
so the ``--help``/config-loading path doesn't require tiktoken installed.
"""
from __future__ import annotations

_ENC = None


def count_tokens(text: str | None) -> int:
    global _ENC
    if not text:
        return 0
    if _ENC is None:
        import tiktoken
        _ENC = tiktoken.get_encoding("cl100k_base")
    return len(_ENC.encode(text, disallowed_special=()))
