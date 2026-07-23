"""Per-repo ingestion spec.

Two sources of a repo's spec, in precedence order (see ``resolve_repo_sources``):

1. **Repo-owned** (recommended) -- ``<checkout>/.rag/ingest.json`` shipped inside the target repo,
   so the repo owner controls their own ingest/retrieval config. Discovered automatically when the
   repo is checked out as a sibling of lp-uworld-rag (``cfg.sibling_root()``); no central edit and
   no ``config.json`` path needed. A repo may instead ship ``.rag/rag-manifest.json`` (the
   query-time contract, serve/index) -- accepted too.
2. **Central** (legacy/back-compat) -- ``repos/<repoKey>/ingest.json`` committed with lp-uworld-rag,
   keyed by directory name, paired with a machine-local ``config.json repos.checkouts`` path.

``repos/common.json`` (shared defaults) is deep-merged as the base under either ingest.json source.
Merge rule (WS8-E): dict values deep-merge (override-not-replace); non-dict values (incl. lists like
``sourceExclude``/``sourceDirs``) replace. persistDir/collection are NOT in the spec -- they're
derived from ``repoKey`` (``code_stores/<repoKey>`` / ``code_<repoKey>``), collision-proof.

A spec yields two things: an :class:`~lp_uworld_rag.repo_ingest.pipeline.IngestJob` (for
``ingest-code``) and an in-memory :class:`~lp_uworld_rag.repo_registry.RepoManifest` (so the existing
retrieval registry reads the produced index with zero changes to its query path). A repo that ships a
hand-authored ``rag-manifest.json`` is loaded directly as that manifest (query-time only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ..retrieval.code_index import IndexConfig, IndexEmbedConfig, IndexRetrievalConfig, IndexStoreConfig
from .pipeline import IngestJob

# Repo-owned spec lives here inside each target repo.
RAG_DIR = ".rag"
INGEST_FILENAME = "ingest.json"
MANIFEST_FILENAME = "rag-manifest.json"

# Shared defaults, baked in (formerly repos/common.json). A repo-owned .rag/ingest.json declares
# only what differs (repoKey, sourceDirs, ...) and inherits everything below. If a legacy
# repos/common.json still exists it is layered on top of these (back-compat), but the folder is no
# longer required. The embed block especially must live here -- RepoSpec.embed is required, so a
# minimal repo spec would fail to validate without it.
COMMON_DEFAULTS = {
    "language": "csharp",
    "chunker": None,
    "sourceExclude": ["bin", "obj", ".g.cs"],
    "rulesDir": ".claude/rules",
    "claudeMd": "CLAUDE.md",
    "retriever": None,
    "embed": {
        "model": "nomic-ai/CodeRankEmbed",
        "trustRemoteCode": True,
        "queryPrefix": "Represent this query for searching relevant code: ",
        "textPrefix": "",
        "device": "auto",
    },
    "retrieval": {
        "topK": 5, "poolSize": 24, "fusionMode": "reciprocal_rerank", "numQueries": 1,
        "priorityOrder": ["code", "entity", "rule"],
        "quotas": {"code": 2, "entity": 1},
        "minCategoryChars": 250, "minCategoryExtra": 2, "hintBoost": 1.5,
        "categoryMap": {"entity": "entity", "controller": "code", "service": "code",
                        "repository": "code", "utility": "code", "other": "code"},
        "overviewHeadingSuffix": " (overview)",
    },
}


class RepoSpec(BaseModel):
    repoKey: str | None = None            # required for repo-owned specs; central specs derive it
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


@dataclass
class RepoSource:
    """Where a repo's config comes from, after precedence resolution. Exactly one of
    ``spec_path`` (an ingest.json) or ``manifest_path`` (a hand-authored rag-manifest.json) is set."""
    repo_key: str
    checkout: Path | None          # resolved source-code checkout (None only for central w/o checkout)
    spec_path: Path | None         # .rag/ingest.json or repos/<key>/ingest.json
    manifest_path: Path | None     # .rag/rag-manifest.json (query-time contract)
    origin: str                    # "explicit" | "sibling" | "central" (for logging)


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


def _read_repo_key(path: Path) -> str | None:
    """Best-effort read of the ``repoKey`` field from an ingest.json / rag-manifest.json."""
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("repoKey")
    except (OSError, ValueError):
        return None


def load_spec_from_file(common_path: Path, ingest_path: Path) -> RepoSpec:
    """Merge an ingest.json onto the baked-in ``COMMON_DEFAULTS``. A legacy ``repos/common.json``
    (``common_path``), if it still exists, is layered between the two for back-compat -- but it is
    no longer required, so the ``repos/`` folder can be removed entirely."""
    if not ingest_path.exists():
        raise RuntimeError(f"ingestion spec not found: {ingest_path}")
    base = dict(COMMON_DEFAULTS)
    if common_path.exists():
        base = _deep_merge(base, json.loads(common_path.read_text(encoding="utf-8")))
    override = json.loads(ingest_path.read_text(encoding="utf-8"))
    return RepoSpec(**_deep_merge(base, override))


# -- discovery -----------------------------------------------------------------------------------

def discover_repo_keys(repos_dir: Path) -> list[str]:
    """Central-only repoKeys: those with a ``repos/<repoKey>/ingest.json`` under ``repos_dir``."""
    if not repos_dir.is_dir():
        return []
    return sorted(p.parent.name for p in repos_dir.glob("*/ingest.json"))


def discover_sibling_specs(cfg) -> list[RepoSource]:
    """Scan ``cfg.sibling_root()`` for repos shipping their own ``.rag/`` spec. Each sibling dir is
    checked for ``.rag/ingest.json`` (preferred) then ``.rag/rag-manifest.json``; repoKey comes from
    the file. lp-uworld-rag itself and non-dirs are skipped. Checkout = the sibling dir (recomputed
    from the current machine's layout every run -> portable, no absolute paths persisted)."""
    root = cfg.sibling_root()
    if not root.is_dir():
        return []
    self_dir = cfg.root_path.resolve()
    out: list[RepoSource] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.resolve() == self_dir:
            continue
        ingest = child / RAG_DIR / INGEST_FILENAME
        manifest = child / RAG_DIR / MANIFEST_FILENAME
        if ingest.exists():
            key = _read_repo_key(ingest)
            if not key:
                continue  # a repo-owned ingest.json must self-identify via repoKey
            out.append(RepoSource(repo_key=key, checkout=child.resolve(), spec_path=ingest,
                                  manifest_path=None, origin="sibling"))
        elif manifest.exists():
            key = _read_repo_key(manifest)
            if not key:
                continue
            out.append(RepoSource(repo_key=key, checkout=child.resolve(), spec_path=None,
                                  manifest_path=manifest, origin="sibling"))
    return out


def resolve_repo_sources(cfg) -> dict[str, RepoSource]:
    """All onboarded repos, keyed by repoKey. **Sibling layout is enforced**: a repo is included
    iff it ships a ``.rag/ingest.json`` (or ``.rag/rag-manifest.json``) AND is checked out under
    ``cfg.sibling_root()`` (by default the folder containing lp-uworld-rag). No per-repo config,
    no central specs, no checkout paths -- onboarding is just "drop ``.rag/ingest.json`` in your
    repo, clone it beside lp-uworld-rag."
    """
    sources: dict[str, RepoSource] = {}
    for src in discover_sibling_specs(cfg):
        sources.setdefault(src.repo_key, src)
    return sources


def all_repo_keys(cfg) -> list[str]:
    """Every onboarded repoKey across all sources (sibling + explicit + central)."""
    return sorted(resolve_repo_sources(cfg).keys())


# -- job / manifest builders ---------------------------------------------------------------------

def load_job(cfg, repo_key: str) -> IngestJob:
    """Build the pipeline job: spec + checkout + repoKey-derived persistDir/collection.
    Resolves the spec source (sibling/explicit/central). Manifest-only repos can't be ingested here
    (they ship their own built index / serve their own RAG)."""
    src = resolve_repo_sources(cfg).get(repo_key)
    if src is None:
        raise RuntimeError(
            f"repo {repo_key!r} not found -- it must ship {RAG_DIR}/{INGEST_FILENAME} (with "
            f'"repoKey": "{repo_key}") and be checked out under {cfg.sibling_root()} '
            f"(beside lp-uworld-rag)")
    if src.spec_path is None:
        raise RuntimeError(f"repo {repo_key!r} ships a rag-manifest.json (query-time only), not an "
                           f"ingest.json -- it manages its own index; nothing to ingest centrally")
    spec = load_spec_from_file(cfg.repos_dir() / "common.json", src.spec_path)
    return IngestJob(
        repo_key=repo_key,
        checkout_path=src.checkout,
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
    """Build a RepoManifest for retrieval. For an ingest.json source, synthesize an in-memory
    contractVersion-2 index-mode manifest (persistDir/collection repoKey-derived). For a repo that
    ships a hand-authored rag-manifest.json, load that file directly."""
    from ..repo_registry import RepoManifest  # local import to avoid an import cycle

    src = resolve_repo_sources(cfg).get(repo_key)
    if src is None:
        raise RuntimeError(f"no source for repo {repo_key!r}")
    if src.manifest_path is not None:
        return RepoManifest.load(src.manifest_path)

    spec = load_spec_from_file(cfg.repos_dir() / "common.json", src.spec_path)
    persist = cfg.code_store_dir(repo_key)
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
        repoRoot=str(src.checkout) if src.checkout else ".",
        index=index,
    )
    # Point manifest_path at the spec file so any relative resolution has a sensible base
    # (persistDir/repoRoot are already absolute).
    manifest.manifest_path = src.spec_path
    return manifest
