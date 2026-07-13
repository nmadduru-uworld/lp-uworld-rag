"""Per-repo ingestion spec: ``repos/common.json`` (shared defaults) deep-merged under
``repos/<repoKey>/ingest.json`` (per-repo override). Committed with lp-uworld-rag so specs ship with
the engine; the machine-local checkout path lives separately in gitignored ``config.json``
(``repos.checkouts``).

Merge rule (WS8-E): dict values deep-merge (override-not-replace); non-dict values (incl. lists like
``sourceExclude``/``sourceDirs``) replace. persistDir/collection are NOT in the spec -- they're
derived from ``repoKey`` (``code_stores/<repoKey>`` / ``code_<repoKey>``), collision-proof.

A spec yields two things: an :class:`~lp_uworld_rag.repo_ingest.pipeline.IngestJob` (for
``ingest-code``) and an in-memory :class:`~lp_uworld_rag.repo_registry.RepoManifest` (so the existing
retrieval registry reads the produced index with zero changes to its query path).
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..direct_index import IndexConfig, IndexEmbedConfig, IndexRetrievalConfig, IndexStoreConfig
from .pipeline import IngestJob


class RepoSpec(BaseModel):
    language: str = "csharp"
    chunker: str | None = None            # None -> use `language` as the registry key
    retriever: str | None = None          # None -> built-in direct_index retriever (L0)
    displayName: str | None = None
    sourceDirs: list[str] = Field(default_factory=list)
    sourceExclude: list[str] = Field(default_factory=lambda: ["bin", "obj", ".g.cs"])
    rulesDir: str | None = ".claude/rules"
    claudeMd: str | None = "CLAUDE.md"
    embed: IndexEmbedConfig
    retrieval: IndexRetrievalConfig = Field(default_factory=IndexRetrievalConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base``. Dicts merge; everything else (incl. lists)
    replaces -- so a repo's ``sourceExclude`` fully replaces the common one."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def discover_repo_keys(repos_dir: Path) -> list[str]:
    """repoKeys with a ``repos/<repoKey>/ingest.json`` under ``repos_dir``."""
    if not repos_dir.is_dir():
        return []
    return sorted(p.parent.name for p in repos_dir.glob("*/ingest.json"))


def load_spec(repos_dir: Path, repo_key: str) -> RepoSpec:
    common_path = repos_dir / "common.json"
    ingest_path = repos_dir / repo_key / "ingest.json"
    if not ingest_path.exists():
        raise RuntimeError(f"no ingestion spec for repo {repo_key!r}: {ingest_path} not found")
    base = json.loads(common_path.read_text(encoding="utf-8")) if common_path.exists() else {}
    override = json.loads(ingest_path.read_text(encoding="utf-8"))
    merged = _deep_merge(base, override)
    return RepoSpec(**merged)


def load_job(cfg, repo_key: str) -> IngestJob:
    """Build the pipeline job: spec + machine-local checkout + repoKey-derived persistDir/collection."""
    spec = load_spec(cfg.repos_dir(), repo_key)
    checkout = cfg.resolved_checkout(repo_key)
    if checkout is None:
        raise RuntimeError(
            f"repo {repo_key!r} has no checkout path -- set repos.checkouts.{repo_key} in config.json "
            f"to this machine's checkout of the repo"
        )
    return IngestJob(
        repo_key=repo_key,
        checkout_path=checkout,
        source_dirs=spec.sourceDirs,
        exclude=spec.sourceExclude,
        chunker_spec=spec.chunker or spec.language,
        language=spec.language,
        embed=spec.embed,
        persist_dir=cfg.code_store_dir(repo_key),
        collection=f"code_{repo_key}",
        rules_dir=spec.rulesDir,
        claude_md=spec.claudeMd,
    )


def to_manifest(cfg, repo_key: str):
    """Build an in-memory RepoManifest (contractVersion 2, index mode) from a repo's spec, so the
    existing RepoRegistry retrieves the produced index unchanged. persistDir/collection are the
    repoKey-derived absolute paths; the retriever override (spec.retriever) rides on the index block.
    """
    from ..repo_registry import RepoManifest  # local import to avoid an import cycle

    spec = load_spec(cfg.repos_dir(), repo_key)
    persist = cfg.code_store_dir(repo_key)
    checkout = cfg.resolved_checkout(repo_key)
    index = IndexConfig(
        embed=spec.embed,
        store=IndexStoreConfig(persistDir=str(persist), collection=f"code_{repo_key}"),
        retrieval=spec.retrieval,
    )
    if spec.retriever is not None:
        index.retriever = spec.retriever
    manifest = RepoManifest(
        contractVersion=2,
        repoKey=repo_key,
        displayName=spec.displayName or repo_key,
        repoRoot=str(checkout) if checkout else ".",
        index=index,
    )
    # No file on disk backs this manifest; point manifest_path at the spec file so any
    # relative resolution has a sensible base (persistDir/repoRoot are already absolute).
    manifest.manifest_path = cfg.repos_dir() / repo_key / "ingest.json"
    return manifest
