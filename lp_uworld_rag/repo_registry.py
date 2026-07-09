"""Orchestrator-side client of the frozen Repo RAG Retrieval Contract (v1, see
``docs/repo-rag-contract.md``). This project never imports another repo's code -- every repo's
code-RAG is reached exclusively through its own stdio MCP server, spawned from its manifest and
kept alive for this process's lifetime (spawning a fresh subprocess per call would pay embedding
model / index load cost on every query).

``mcp`` is imported lazily inside the session machinery so the package (and ``config``/CLI
``--help``) keeps working without it installed.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .direct_index import IndexConfig

# v1: manifest must declare "serve" (spawn an MCP server) -- the only mode that existed at first.
# v2: manifest declares "serve", "index" (direct Chroma read, no server -- see direct_index.py), or
# both; "index" is preferred when both are present. A v1 manifest keeps working unchanged forever;
# it just can't add "index" without bumping to 2 (see RepoManifest.load).
SUPPORTED_CONTRACT_VERSIONS = (1, 2)

# Matches a python-style "unexpected keyword argument"/"positional argument" TypeError message a
# pre-contract server's tool wrapper raises when called with a kwarg it doesn't accept yet --
# distinguishes "this server predates file_hints/top_k" from a genuine error worth surfacing.
_SIGNATURE_MISMATCH_RE = re.compile(r"unexpected keyword argument|positional argument")


class ServeConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class RepoManifest(BaseModel):
    contractVersion: int
    repoKey: str
    displayName: str
    repoRoot: str = "."
    serve: ServeConfig | None = None
    index: IndexConfig | None = None
    known_risk: str | None = None
    tracked_in: str | None = None

    # Set by load(); excluded from the schema itself.
    manifest_path: Path = Field(default=Path("."), exclude=True)

    @classmethod
    def load(cls, path: str | Path) -> "RepoManifest":
        p = Path(path).resolve()
        if not p.exists():
            raise RuntimeError(f"repo manifest not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"repo manifest {p} is not valid JSON: {exc}") from exc
        try:
            manifest = cls(**data)
        except ValidationError as exc:
            raise RuntimeError(f"repo manifest {p} does not match the contract schema: {exc}") from exc
        if manifest.contractVersion not in SUPPORTED_CONTRACT_VERSIONS:
            raise RuntimeError(
                f"repo manifest {p} declares contractVersion {manifest.contractVersion}, "
                f"this orchestrator speaks {SUPPORTED_CONTRACT_VERSIONS}"
            )
        if manifest.contractVersion == 1:
            if manifest.serve is None or manifest.index is not None:
                raise RuntimeError(
                    f"repo manifest {p}: contractVersion 1 must declare exactly \"serve\" "
                    f"(no \"index\" -- bump to contractVersion 2 to add it)"
                )
        elif manifest.serve is None and manifest.index is None:
            raise RuntimeError(
                f"repo manifest {p}: contractVersion 2 must declare at least one of \"serve\"/\"index\""
            )
        manifest.manifest_path = p
        return manifest

    def resolved_repo_root(self) -> Path:
        return (self.manifest_path.parent / self.repoRoot).resolve()

    def resolved_command(self) -> str:
        """``serve.command`` relative to the manifest's own directory, not the orchestrator's cwd --
        a bare executable name (no path separator, or one that doesn't resolve to a real file next
        to the manifest) is left untouched for PATH to resolve."""
        cmd = self.serve.command
        candidate = Path(cmd)
        if candidate.is_absolute():
            return cmd
        resolved = (self.manifest_path.parent / candidate).resolve()
        return str(resolved) if resolved.exists() else cmd

    def resolved_index_persist_dir(self) -> Path:
        """``index.store.persistDir`` relative to the manifest's own directory -- unlike
        ``repoRoot``, this is a data directory that lives inside the repo's RAG tool dir, not the
        repo checkout itself."""
        p = Path(self.index.store.persistDir)
        return p if p.is_absolute() else (self.manifest_path.parent / p).resolve()


class CodeChunkResult(BaseModel):
    heading: str
    filePath: str
    layer: str | None = None  # absent on non-code chunks (e.g. a rule/markdown chunk)
    sourceType: str
    content: str
    score: float
    startLine: int | None = None
    endLine: int | None = None
    parentSummary: str | None = None


class CodeQueryResult(BaseModel):
    chunks: list[CodeChunkResult] = Field(default_factory=list)


class _RepoSession:
    """One repo's persistent stdio MCP ClientSession, owned by a dedicated background thread
    running its own asyncio event loop. The session's async context managers (``stdio_client`` /
    ``ClientSession``) must stay entered for the subprocess to stay alive, so a plain
    ``asyncio.run()`` per call would respawn the subprocess (and re-pay its embedding/index load
    cost) every query -- this keeps one loop alive for the process's lifetime instead, and sync
    callers cross into it via ``asyncio.run_coroutine_threadsafe``."""

    def __init__(self, manifest: RepoManifest):
        self.manifest = manifest
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._start_error: Exception | None = None
        self._lock = threading.Lock()

    def _ensure_started(self, timeout: float = 30.0) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._start_error is None:
                return
            self._ready.clear()
            self._start_error = None
            self._session = None
            self._thread = threading.Thread(
                target=self._run_loop, name=f"rag-repo-{self.manifest.repoKey}", daemon=True
            )
            self._thread.start()
            if not self._ready.wait(timeout=timeout):
                raise RuntimeError(
                    f"repo {self.manifest.repoKey!r}: MCP server did not become ready within {timeout:.0f}s"
                )
            if self._start_error is not None:
                raise RuntimeError(
                    f"repo {self.manifest.repoKey!r}: failed to start MCP server "
                    f"({self.manifest.serve.command} {' '.join(self.manifest.serve.args)}): {self._start_error}"
                ) from self._start_error

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        finally:
            loop.close()

    async def _main(self) -> None:
        from mcp import ClientSession, StdioServerParameters, stdio_client

        self._stop_event = asyncio.Event()
        params = StdioServerParameters(
            command=self.manifest.resolved_command(),
            args=self.manifest.serve.args,
            env=self.manifest.serve.env or None,
            cwd=str(self.manifest.resolved_repo_root()),
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop_event.wait()
        except Exception as exc:  # surfaced to the waiting caller in _ensure_started
            self._start_error = exc
            self._ready.set()

    def call_tool(self, name: str, arguments: dict, timeout: float = 60.0) -> str:
        self._ensure_started()
        loop = self._loop
        session = self._session
        if loop is None or session is None:
            raise RuntimeError(f"repo {self.manifest.repoKey!r}: no active session")

        async def _do() -> str:
            result = await session.call_tool(name, arguments)
            if result.isError:
                text = "; ".join(getattr(block, "text", str(block)) for block in result.content)
                raise RuntimeError(f"tool {name!r} on repo {self.manifest.repoKey!r} returned an error: {text}")
            texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
            if not texts:
                raise RuntimeError(f"tool {name!r} on repo {self.manifest.repoKey!r} returned no text content")
            return texts[0]

        future = asyncio.run_coroutine_threadsafe(_do(), loop)
        try:
            return future.result(timeout=timeout)
        except RuntimeError:
            raise  # a real tool-level error (e.g. bad JSON, isError) -- leave the session running
        except Exception:
            self.stop()  # transport-level failure (timeout, broken pipe) -- next call respawns
            raise

    def stop(self) -> None:
        with self._lock:
            loop, stop_event, thread = self._loop, self._stop_event, self._thread
            if loop is not None and stop_event is not None:
                try:
                    loop.call_soon_threadsafe(stop_event.set)
                except RuntimeError:
                    pass
            if thread is not None:
                thread.join(timeout=5)
            self._loop = None
            self._thread = None
            self._session = None
            self._stop_event = None


class _DirectIndexSession:
    """A repo's "index" manifest, retrieved with no subprocess and no IPC: builds -- and keeps for
    the process's lifetime -- one :class:`direct_index.RetrieverCache` (embedding model + Chroma +
    BM25 corpus), the same amortization :class:`_RepoSession` gets from keeping an MCP subprocess
    alive, minus the subprocess."""

    def __init__(self, manifest: RepoManifest):
        self.manifest = manifest
        self._cache = None
        self._lock = threading.Lock()

    def _ensure_cache(self):
        if self._cache is None:
            with self._lock:
                if self._cache is None:
                    from . import direct_index
                    self._cache = direct_index.build_retriever_cache(
                        self.manifest.resolved_index_persist_dir(), self.manifest.index
                    )
        return self._cache

    def query(self, question: str, top_k: int | None, file_hints: list[str] | None) -> dict:
        from . import direct_index
        cache = self._ensure_cache()
        return direct_index.query(
            self.manifest.resolved_index_persist_dir(), self.manifest.index, question,
            top_k=top_k, file_hints=file_hints, cache=cache,
        )

    def status(self) -> str:
        from . import direct_index
        self._ensure_cache()  # surfaces load errors at the same point _RepoSession's ready-wait would
        return direct_index.status(self.manifest.resolved_index_persist_dir(), self.manifest.index)


class RepoRegistry:
    """Loads every configured repo manifest up front (bad manifests are recorded, not raised, so
    one broken repo doesn't block the others) and lazily builds one persistent backend per repo on
    first use -- an MCP session for "serve" mode, or an in-process retriever cache for "index" mode
    (preferred when a manifest declares both)."""

    def __init__(self, manifest_paths: list[str | Path]):
        self.manifests: dict[str, RepoManifest] = {}
        self.errors: dict[str, str] = {}
        self._serve_sessions: dict[str, _RepoSession] = {}
        self._index_sessions: dict[str, _DirectIndexSession] = {}
        # repoKey -> resolved index persist dir, for cross-repo collision detection (see below).
        self._persist_dirs: dict[str, Path] = {}
        for raw_path in manifest_paths:
            try:
                manifest = RepoManifest.load(raw_path)
            except Exception as exc:
                self.errors[str(raw_path)] = str(exc)
                continue
            if manifest.repoKey in self.manifests:
                existing = self.manifests[manifest.repoKey].manifest_path
                self.errors[str(raw_path)] = (
                    f"duplicate repoKey {manifest.repoKey!r} (already registered from {existing})"
                )
                continue
            # Cross-collection guard: an index-mode repo must own a Chroma persist directory that no
            # other registered repo already uses. Without this, a second repo whose manifest was
            # copy-pasted (and whose persistDir wasn't changed) would write into the first repo's
            # store, silently merging two unrelated code corpora into one collection with no error --
            # the single most likely failure when onboarding a new repo. repoKey uniqueness (above)
            # doesn't catch it: two distinct repoKeys can still resolve to the same directory.
            if manifest.index is not None:
                pdir = manifest.resolved_index_persist_dir()
                clash = next((k for k, d in self._persist_dirs.items() if d == pdir), None)
                if clash is not None:
                    self.errors[str(raw_path)] = (
                        f"repo {manifest.repoKey!r} index.store.persistDir resolves to {pdir}, "
                        f"already used by repo {clash!r} -- each repo needs its own store directory"
                    )
                    continue
                self._persist_dirs[manifest.repoKey] = pdir
            self.manifests[manifest.repoKey] = manifest

    def repo_keys(self) -> list[str]:
        return sorted(self.manifests)

    def backend(self, repo_key: str) -> str:
        """Which mode a registered repo will actually be queried through -- "index" is preferred
        whenever a manifest declares both."""
        manifest = self.manifests[repo_key]
        return "index" if manifest.index is not None else "serve"

    def _serve_session_for(self, repo_key: str) -> _RepoSession:
        manifest = self.manifests[repo_key]
        if manifest.serve is None:
            raise RuntimeError(f"repo {repo_key!r} has no \"serve\" block in its manifest")
        if repo_key not in self._serve_sessions:
            self._serve_sessions[repo_key] = _RepoSession(manifest)
        return self._serve_sessions[repo_key]

    def _index_session_for(self, repo_key: str) -> _DirectIndexSession:
        if repo_key not in self._index_sessions:
            self._index_sessions[repo_key] = _DirectIndexSession(self.manifests[repo_key])
        return self._index_sessions[repo_key]

    def _call_query_rag_mcp(self, repo_key: str, kwargs: dict) -> str:
        """Call the "serve" backend's ``query_rag``, retrying once without ``file_hints`` and/or
        ``top_k`` if the server's tool signature doesn't accept them yet (a pre-contract server) --
        these are the only two optional kwargs the contract allows a server to ignore outright, so
        dropping them is always a safe degrade, never a silent behavior change in what's asked."""
        session = self._serve_session_for(repo_key)
        attempt = dict(kwargs)
        for optional_key in ("file_hints", "top_k"):
            try:
                return session.call_tool("query_rag", attempt)
            except RuntimeError as exc:
                if optional_key in attempt and _SIGNATURE_MISMATCH_RE.search(str(exc)):
                    attempt = {k: v for k, v in attempt.items() if k != optional_key}
                    continue
                raise
        return session.call_tool("query_rag", attempt)

    def query(self, repo_key: str, question: str, top_k: int | None = None,
              file_hints: list[str] | None = None) -> CodeQueryResult:
        if repo_key not in self.manifests:
            raise KeyError(f"unregistered repo {repo_key!r}; registered: {self.repo_keys()}")

        if self.backend(repo_key) == "index":
            data = self._index_session_for(repo_key).query(question, top_k, file_hints)
        else:
            kwargs: dict[str, Any] = {"question": question}
            if top_k is not None:
                kwargs["top_k"] = top_k
            if file_hints:
                kwargs["file_hints"] = file_hints
            raw = self._call_query_rag_mcp(repo_key, kwargs)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"repo {repo_key!r}: query_rag returned invalid JSON: {exc}") from exc

        try:
            return CodeQueryResult(**data)
        except ValidationError as exc:
            raise RuntimeError(f"repo {repo_key!r}: query_rag response violates the contract schema: {exc}") from exc

    def status(self, repo_key: str) -> str:
        if repo_key not in self.manifests:
            raise KeyError(f"unregistered repo {repo_key!r}; registered: {self.repo_keys()}")
        if self.backend(repo_key) == "index":
            return self._index_session_for(repo_key).status()
        return self._serve_session_for(repo_key).call_tool("rag_status", {})

    def validate(self, repo_key: str) -> dict:
        """manifest ok -> backend launches -> rag_status non-empty -> probe query_rag -> schema
        check. Each step recorded independently so ``repos --validate`` can point at exactly which
        stage failed rather than just "repo X is broken"."""
        steps: list[dict] = []
        result = {"repo": repo_key, "ok": False, "steps": steps}
        if repo_key not in self.manifests:
            steps.append({"step": "manifest", "ok": False, "error": "not registered"})
            result["error"] = "not registered"
            return result
        result["backend"] = self.backend(repo_key)
        steps.append({"step": "manifest", "ok": True})

        try:
            status_text = self.status(repo_key)
            if not status_text or not status_text.strip():
                raise RuntimeError("rag_status returned an empty string")
            steps.append({"step": "rag_status", "ok": True})
        except Exception as exc:
            steps.append({"step": "rag_status", "ok": False, "error": str(exc)})
            result["error"] = str(exc)
            return result

        try:
            self.query(repo_key, "sanity check probe query", top_k=1)
            steps.append({"step": "query_rag probe", "ok": True})
        except Exception as exc:
            steps.append({"step": "query_rag probe", "ok": False, "error": str(exc)})
            result["error"] = str(exc)
            return result

        result["ok"] = True
        manifest = self.manifests[repo_key]
        if manifest.known_risk:
            result["known_risk"] = manifest.known_risk
            result["tracked_in"] = manifest.tracked_in
        return result

    def close(self) -> None:
        # Only "serve" sessions hold a resource that needs an explicit stop (a subprocess) --
        # "index" sessions are just an in-memory cache, nothing to tear down.
        for session in self._serve_sessions.values():
            session.stop()
