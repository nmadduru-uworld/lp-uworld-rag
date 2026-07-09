"""Typed configuration loaded from ``config.json`` (+ ``LP_RAG_CONFIG`` env override).

Mirrors ``config.json.example``. Uses pydantic for validation. This project is self-contained — it
does not read any other repo's config or code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


class EmbedConfig(BaseModel):
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    trustRemoteCode: bool = True
    # nomic requires asymmetric task prefixes for good doc/query separation.
    queryPrefix: str = "search_query: "
    textPrefix: str = "search_document: "
    dim: int = 768
    device: str = "auto"  # "auto" -> cuda if available else cpu; or force "cuda"/"cpu"


class StoreConfig(BaseModel):
    # Separate Chroma PersistentClient directories -- two genuinely independent local databases,
    # not two collections sharing one client. Each carries its own docstore (for BM25) and its own
    # delta-ingest state file (see ingest.py), so one can be wiped/rebuilt/backed up without
    # touching the other.
    functionalPersistDir: str = "chroma_functional"
    technicalPersistDir: str = "chroma_technical"


class CollectionsConfig(BaseModel):
    functional: str = "docs_functional"
    technical: str = "docs_technical"
    # Optional per-collection embedding model override. Each collection is already queried by an
    # independent retriever (see index.py), so a different model per collection is a config change,
    # not a redesign -- unset (None) means "use EmbedConfig.model", the default for both today.
    functionalEmbedModel: str | None = None
    technicalEmbedModel: str | None = None


class RetrievalConfig(BaseModel):
    topK: int = 8
    # Candidate pool fetched from fusion before quota selection cuts it down to topK.
    poolSize: int = 24
    fusionMode: str = "reciprocal_rerank"  # QueryFusionRetriever RRF
    numQueries: int = 1  # 1 = no query expansion (no LLM call)
    # Minimum guaranteed slots per doc_type within docs_technical's share of topK, so an
    # endpoint-heavy query doesn't crowd out its controller/DB context. docs_functional has one
    # doc_type today (feature-hub) so no quota is needed there.
    quotas: dict[str, int] = Field(
        default_factory=lambda: {"endpoint": 2, "controller-context": 1, "db-collection": 1}
    )


class RerankConfig(BaseModel):
    """Cross-encoder rerank -- on by default. Confirmed live: without it, a tiny docs_functional
    pool's RRF scores trivially outrank genuinely relevant docs_technical hits (RRF score reflects
    rank-within-pool, not absolute relevance, and a 2-document pool's rank-1 looks artificially
    strong next to a 20-document pool's real rank-1). Quota alone doesn't fix this; rerank does."""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidatePoolCap: int = 40


class ConfluenceConfig(BaseModel):
    baseUrl: str = "https://uworld.atlassian.net"  # human-facing site URL, not used for API calls
    # "AI Context -- Technical Feature Knowledge Base" container page. Crawled into docs_technical.
    rootPageId: str = "7472545873"
    # "AI Context -- Functional Feature Knowledge Base" container -- the sibling Feature Hub tree,
    # crawled into docs_functional. Both roots are walked by the same crawl(); see confluence_reader.py.
    functionalRootPageId: str = "7475658997"
    # Page titles to skip entirely during the crawl -- not fetched, not chunked, not recursed into.
    # "Artifacts" links out to design-reference HTML pages (e.g. claude.ai artifacts), not knowledge
    # base content itself, so it has no business being retrievable alongside real Feature/Technical
    # Hub content. "Order(api)" (a new repo folder under Repos) is out of scope for now -- its
    # endpoint pages also have real metadata quality issues (inconsistent repo naming across its own
    # endpoints -- "orders-api" vs "uwwebtech.uwweb.orders.api" -- and at least one page with a
    # multi-value `controller` field that corrupts controller_id) that need fixing at the source
    # before this repo is worth indexing.
    excludedTitles: list[str] = Field(default_factory=lambda: ["Artifacts", "Order(api)"])
    # This org's scoped API tokens only work through the api.atlassian.com gateway, never the
    # site-direct domain -- confirmed live: Basic Auth against baseUrl returns a 401 + "WWW-Authenticate:
    # OAuth" challenge (an org-level policy against personal-token Basic Auth on the site domain
    # itself), while the identical Basic Auth against this gateway URL succeeds. Look this up for a
    # different site via `GET https://<site>.atlassian.net/_edge/tenant_info` -> {"cloudId": "..."}.
    cloudId: str = "a9db67d5-6074-4da6-9204-57a0edf50495"
    # Optional audit-only cross-check against a page-id manifest maintained elsewhere. None
    # (disabled) by default -- this project has no hard dependency on another repo's files.
    manifestPath: str | None = None
    # Names of env vars holding credentials -- never the secrets themselves.
    emailEnvVar: str = "CONFLUENCE_EMAIL"
    apiTokenEnvVar: str = "CONFLUENCE_API_TOKEN"
    # Direct override, checked before the env var -- an escape hatch for a machine where env vars
    # set via `setx` aren't propagating to the shell this runs in. config.json is gitignored, so a
    # real value here doesn't reach version control. config.json.example keeps these null -- the
    # example should never suggest committing a real secret.
    emailValue: str | None = None
    apiTokenValue: str | None = None

    def email(self) -> str:
        val = self.emailValue or os.environ.get(self.emailEnvVar)
        if not val:
            raise RuntimeError(f"Missing Confluence email -- set env var {self.emailEnvVar} "
                                f"or confluence.emailValue in config.json")
        return val

    def api_token(self) -> str:
        val = self.apiTokenValue or os.environ.get(self.apiTokenEnvVar)
        if not val:
            raise RuntimeError(f"Missing Confluence API token -- set env var {self.apiTokenEnvVar} "
                                f"or confluence.apiTokenValue in config.json")
        return val

    def api_base(self) -> str:
        """Base URL for REST calls -- the api.atlassian.com gateway, not ``baseUrl`` (see cloudId)."""
        return f"https://api.atlassian.com/ex/confluence/{self.cloudId}"


class ReposConfig(BaseModel):
    # Paths to other repos' rag-manifest.json (see docs/repo-rag-contract.md) -- absolute, or
    # relative to this config file's own directory (RagConfig.root_path), resolved lazily by
    # resolved_manifest_paths() so nothing here needs the manifests to exist just to load config.
    manifests: list[str] = Field(default_factory=list)


class RagConfig(BaseModel):
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    collections: CollectionsConfig = Field(default_factory=CollectionsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    confluence: ConfluenceConfig = Field(default_factory=ConfluenceConfig)
    repos: ReposConfig = Field(default_factory=ReposConfig)

    # Absolute path to the project root, resolved at load time (not persisted).
    root_path: Path = Field(default=Path("."), exclude=True)

    def embed_model_name(self, collection: str) -> str:
        """Resolve the embedding model for "functional" or "technical", falling back to the shared default."""
        override = getattr(self.collections, f"{collection}EmbedModel", None)
        return override or self.embed.model

    def collection_name(self, collection: str) -> str:
        return getattr(self.collections, collection)

    def persist_dir(self, collection: str) -> str:
        """Each collection's own Chroma PersistentClient directory -- two independent local
        databases, not two collections sharing one client (see StoreConfig)."""
        return getattr(self.store, f"{collection}PersistDir")

    def resolved_manifest_paths(self) -> list[Path]:
        """Every configured repo manifest path, resolved against ``root_path`` (this config
        file's own directory) when not already absolute."""
        out = []
        for raw in self.repos.manifests:
            p = Path(raw)
            out.append(p if p.is_absolute() else (self.root_path / p).resolve())
        return out


def _default_config_path() -> Path:
    """LP_RAG_CONFIG env wins; else config.json next to this package's parent."""
    env = os.environ.get("LP_RAG_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "config.json"


def load_config(path: str | os.PathLike[str] | None = None) -> RagConfig:
    """Load and validate configuration. Falls back to defaults if the file is absent."""
    cfg_path = Path(path) if path else _default_config_path()
    data: dict = {}
    if cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = RagConfig(**data)
    cfg.root_path = (cfg_path.parent if cfg_path.exists() else Path.cwd()).resolve()
    return cfg
