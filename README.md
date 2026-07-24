# lp-uworld-rag

Shared functional/technical RAG for the UWorld Learning Platform Confluence knowledge base --
Feature Hubs, Technical Hubs, Endpoint Documents, Controller Context, and the Data Stores catalog.
It never imports another repo's code as a library. It does two things across repos:

- **Ingests repo code centrally** -- a shared engine (`lp_uworld_rag/repo_ingest/`) builds each
  repo's code index by being pointed at that repo's checkout, so a repo needs no in-tree RAG tool
  of its own (see "Central code ingestion" below).
- **Orchestrates at query time** -- `deep_query` routes a doc hit to the repo(s) it's actually
  about and retrieves from their code index too (see "Cross-repo routing" below).

> **New here? → [ONBOARDING.md](ONBOARDING.md)** — the step-by-step get-running guide (query-only in
> ~10 min with a prebuilt index, or a full build from Confluence). This README is the **technical
> reference**: usage, architecture, configuration, and how to plug a repo's code in.

## Getting started

Full setup — prerequisites, the two paths, and troubleshooting — lives in
**[ONBOARDING.md](ONBOARDING.md)**. In short, from the repo root:

```
.\scripts\setup.ps1 -QueryOnly        # query with a prebuilt index (no Confluence token needed)
.\scripts\setup.ps1 -Email you@uworld.com -Token <token>   # OR a full build from Confluence
.\scripts\Register-LpRag.ps1    # register lp-rag for Claude Code, Cowork, and Desktop
```

`setup.ps1` is idempotent and delta-based (re-runs only re-embed what changed). Flags: `-QueryOnly`,
`-Full`, `-SkipCode`, `-DryRun`. See ONBOARDING.md for the prebuilt-index download and prerequisites.

## Usage

```
python -m lp_uworld_rag ingest [--full] [--strict]                                  # crawl + chunk + embed Confluence docs (--strict: fail on metadata errors)
python -m lp_uworld_rag query "POST faculty-led/group-performance" [--collection functional|technical] [--top-k 5] [--no-siblings]
python -m lp_uworld_rag expand ctrl::reports::FacultyLedPerformanceController        # pull one citation's full content
python -m lp_uworld_rag status                                                       # indexed chunk counts by collection/doc_type
python -m lp_uworld_rag eval                                                         # 5-tier quality harness -> RESULT: PASS
python -m lp_uworld_rag validate                                                     # metadata-integrity check only (no re-ingest, no model load)
python -m lp_uworld_rag mcp                                                          # stdio MCP server for Claude Code / other agents
python -m lp_uworld_rag ingest-code (--repo reports | --all) [--full]                # ingest a repo's code (sibling .rag/ingest.json)
python -m lp_uworld_rag deep-query "why does ... return null body" [--repo R ...] [--top-k-docs N] [--top-k-code N]
python -m lp_uworld_rag query-code "group performance date range cap" [--repo reports] [--top-k N] [--file-hint PATH ...]
python -m lp_uworld_rag repos [--validate]                                           # list registered repos (or probe conformance)
python -m lp_uworld_rag validate-citations                                           # fact-check every File.cs:line doc citation
```

`docs_functional` is expected to be **empty** until Feature Hub pages actually exist in Confluence --
both are 404 as of this project's creation; the Technical Hub pages that link to them say so
explicitly ("functional hub -- link pending").

### Verify it worked

```
python -m lp_uworld_rag status            # per-collection chunk counts, e.g. "technical | endpoint | 30 chunks"
python -m lp_uworld_rag repos --validate  # healthy: [PASS] <repo> (backend=index) -- manifest/rag_status/probe ok
python -m lp_uworld_rag eval              # 5 tiers (metadata -> resolve -> retrieval -> expand -> routing) -> RESULT: PASS
```

A small/empty `functional` line is fine (see above). A tier-3 "NEAR"/"FAIL" in `eval` means
retrieval quality slipped, not a crash.

## Project layout

Every module under `lp_uworld_rag/`, by role:

| Module                                                              | What it does                                                                                                                                                               |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Entry / config**                                            |                                                                                                                                                                            |
| `__main__.py`                                                     | CLI verbs + dispatch (`ingest`, `query`, `deep-query`, `ingest-code`, `repos`, …)                                                                               |
| `config.py`                                                       | Typed (`pydantic`) load of `config.json` -- embed/store/collections/retrieval/rerank/confluence/repos                                                                  |
| `mcp_server.py`                                                   | FastMCP stdio server exposing`query_rag`/`expand`/`deep_query`/`query_code`/`rag_status`                                                                         |
| **`common/`** (core — reusable, depends on nothing above)  |                                                                                                                                                                            |
| `common/retrieval_engine.py`                                      | **Shared primitives** -- embed model, Chroma client, docstore, BM25, fusion retriever, rerank (used by both retrieval engines)                                       |
| `common/store_sync.py`                                            | Ingest-time content-hash delta + docstore persist, shared by docs + code ingest                                                                                            |
| `common/tokens.py`                                                | One`count_tokens` (tiktoken `cl100k_base`) shared by `eval` + `orchestrator`                                                                                       |
| `common/overrides.py`                                             | Shared L1-override`importlib` loader behind the chunker + retriever override seams                                                                                       |
| **Docs pipeline**                                             |                                                                                                                                                                            |
| `confluence_client.py`                                            | Confluence Cloud REST v2 client (children, body → markdown)                                                                                                               |
| `confluence_reader.py`                                            | Crawl the two Confluence trees, parse metadata, classify doc_type, section-split                                                                                           |
| `chunker.py`                                                      | One capped, Chroma-safe`TextNode` per page/section (content-hash id)                                                                                                     |
| `ingest.py`                                                       | Crawl → chunk → content-hash**delta** upsert into `docs_functional` + `docs_technical`                                                                         |
| **`retrieval/`** (query-time engines, build on `common/`) |                                                                                                                                                                            |
| `retrieval/docs_index.py`                                         | Docs engine (was`index.py`): retrieve → resolve linked ids as citations → quota/rerank/rank                                                                            |
| `retrieval/code_index.py`                                         | Repo-code "index"-mode engine (was`direct_index.py`): hint-boost, priority order, small-to-big parent join                                                               |
| `retrieval/orchestrator.py`                                       | `deep_query`/`query_code`: docs → route by stable-id → each routed repo's code-RAG; score floor + token budget                                                       |
| **Repo plug-in**                                              |                                                                                                                                                                            |
| `repo_registry.py`                                                | Load repo manifests; run each via`serve` (MCP subprocess) or `index` (in-process); conformance validation                                                              |
| `repo_ingest/`                                                    | Central code ingestion:`spec.py` (per-repo spec deep-merge), `pipeline.py` (checkout → chunks → Chroma), `layer.py`, `chunkers/*` (C#/markdown/generic registry) |
| **Quality**                                                   |                                                                                                                                                                            |
| `eval.py`                                                         | 5-tier eval**harness/engine** (metadata integrity, resolve, retrieval quality, expand, routing)                                                                      |
| `eval_cases.py`                                                   | The domain-specific query/expand/routing**fixtures** the harness runs (kept out of the harness so it stays domain-agnostic)                                          |

The package is layered like an onion: **`common/`** (core primitives, no intra-package deps) →
**`retrieval/`** + docs pipeline (build on core) → repo plug-in → `eval`, with `config.py` /
`__main__.py` / `mcp_server.py` at the top. `common/` exists to remove duplication — the two
retrieval engines used to reimplement the same embed/Chroma/BM25/rerank plumbing and the token
counter lived in two places. See [docs/repo-rag-contract.md](docs/repo-rag-contract.md) for the
interface a repo's code-RAG plugs in through.

## How results are shaped: citations, not inlined content

A query hit's linked nodes (its controller, the DB collections it touches, the Technical Hub of every
feature it belongs to) come back as `{id, title, docType}` **citations**, not full text -- call
`expand(id)` to pull one in on demand. This keeps a typical response small even though a single
endpoint hit can legitimately reference several other nodes. Pass `--no-siblings` (CLI) /
`include_siblings=False` (MCP) for the narrowest possible response when you already know you only
want the one hit -- e.g. chasing a bugfix in one specific endpoint.

## Ingest is a version-tracked delta

Re-running `ingest` (no `--full`) only re-embeds pages whose content actually changed since the last
run (detected via each page's resulting chunk content-hash, not just Confluence's `version` number --
see `ingest.py`'s docstring for why that's more robust) and prunes chunks for pages that disappeared.
`--full` ignores all of that and rebuilds both collections from scratch -- use it after a
`chunker.py`/tagging logic change.

## Cross-repo routing: `deep_query` / `query_code`

`deep_query(question, ...)` runs the docs stage above, then derives which registered repo(s)'
own code-RAG the question is actually about from the doc hits' stable ids
(`ep::<repo>::<op>`, `ctrl::<repo>::<name>`) and queries each routed repo's code-RAG MCP server
too. There is no intent classification anywhere in this path -- routing is a plain id-prefix match.
A question whose docs cite no registered repo (or cite one that isn't registered) falls back to
querying *every* registered repo with the raw question text, flagged via
`routing.fallback_used` -- it never returns silently empty code results.

```json
{
  "docs": { "chunks": [ ... ] },
  "routing": { "repos": ["reports"], "derived_from": ["reports"], "fallback_used": false, "hints": {...} },
  "code": { "reports": [ { "heading", "filePath", "layer", "sourceType", "content", "score", "startLine"?, "endLine"?, "parentSummary"? } ] }
}
```

`query_code(question, repo=None, file_hints=None, top_k=None)` is a thin passthrough to one or
every registered repo's code-RAG, with no docs stage.

## Central code ingestion

The code-ingestion engine lives **here**, in `lp_uworld_rag/repo_ingest/`, on this project's single
venv -- a repo needs no RAG tool (venv, deps, ingest code) checked into its own tree. You point the
engine at a repo's checkout and it builds that repo's code index.

### Onboard a repo — one file, in that repo (recommended)

The repo **owns its spec**: add `.rag/ingest.json` at the repo root, check the repo out **as a
sibling of lp-uworld-rag**, and it's auto-discovered — no edit to lp-uworld-rag, no `config.json`
path.

```jsonc
// <your-repo>/.rag/ingest.json
{ "repoKey": "reports",                          // MUST equal the `repo` in Confluence doc metadata
  "displayName": "uwwebtech.learningplatform.reports.api",
  "language": "csharp",
  "sourceDirs": ["uwwebtech.learningplatform.reports.api", "…application", "…infrastructure"],
  "sourceExclude": ["bin","obj",".g.cs","Migrations","Properties"] }
```

```
python -m lp_uworld_rag ingest-code --all     # discovers every sibling shipping .rag/ingest.json
python -m lp_uworld_rag repos --validate       # confirm it loads + conforms
```

`repoKey` **must equal** the `repo` value in the Confluence doc metadata (`ep::<repoKey>::...`) — that
routes a doc hit to this index. A repo may instead ship `.rag/rag-manifest.json` (the serve/index
contract) if it manages its own index. Shared defaults (embed model, retrieval tuning, excludes) are
**baked into the engine** (`repo_ingest/spec.py: COMMON_DEFAULTS`), so your `.rag/ingest.json`
declares only what differs — typically just `repoKey` + `sourceDirs`.

**Sibling layout is required — there is no per-repo config.** A repo is discovered iff it ships
`.rag/ingest.json` **and** is cloned under the same parent folder as lp-uworld-rag. No
`config.json` entries, no central specs, no checkout paths. (The scan folder can be redirected with
`repos.siblingRoot` for an unusual layout, but the default — actual siblings — needs nothing.)

Structure:

```
<parent dir>/
  lp-uworld-rag/
    code_stores/<repoKey>/       # gitignored built index (Chroma + docstore), derived from repoKey
    config.json                  # gitignored; NO repos entries needed
  <your-repo>/.rag/ingest.json   # repo-owned spec, auto-discovered as a sibling
```

Ingest is a content-hash **delta**: a re-run only re-embeds files whose chunks actually changed and
prunes files that disappeared; `--full` rebuilds.

### Overriding the defaults

Most repos need nothing beyond `ingest.json`. When a repo's needs exceed config, it can override one
piece with a single `.py` module (loaded onto this venv -- still no in-tree tool), or run its own
server -- see the override ladder (L0/L1/L2) and the frozen interface in
[docs/repo-rag-contract.md](docs/repo-rag-contract.md):

- **`chunker`** (in `ingest.json`) -- a custom chunker for a language/layout the built-ins don't fit.
- **`retriever`** -- a custom retriever for a repo whose retrieval is too complex for config alone.
- **serve mode** -- a wholly bespoke MCP server, for a repo that fits neither (the orchestrator
  spawns it and calls `query_rag`/`rag_status`).

`python -m lp_uworld_rag validate-citations` fact-checks every `File.cs:line` citation an ingested
doc chunk carries against a registered repo's actual checkout (and, where the code index carries
line ranges, against what it actually retrieves) -- citations stay a rank boost everywhere else,
this is the one place they're checked as fact.

## Configuration (`config.json`)

Copy `config.json.example` → `config.json` (the setup script does this). Key fields:

| Section / field | What it does |
| --- | --- |
| `embed.model` | Docs embedding model. |
| `embed.trustRemoteCode` | Allow the embedding model to run its own remote code (nomic needs it). |
| `embed.queryPrefix` / `embed.textPrefix` | Asymmetric task prefixes nomic wants for query vs document. |
| `embed.device` | `"auto"` / `"cuda"` / `"cpu"`. |
| `store.functionalPersistDir` / `store.technicalPersistDir` | The two docs Chroma directories. |
| `collections.functional` / `collections.technical` | Chroma collection names inside those dirs. |
| `collections.functionalEmbedModel` / `technicalEmbedModel` | Optional per-collection embed override (`null` = use `embed.model`). |
| `retrieval.topK` | Results returned per query (default 5). |
| `retrieval.poolSize` | Candidate pool fetched before quota/rerank trims to `topK`. |
| `retrieval.fusionMode` | RRF mode (`reciprocal_rerank`). |
| `retrieval.numQueries` | 1 = no query expansion (no LLM call). |
| `retrieval.quotas` | Min guaranteed slots per `doc_type` in the technical collection. |
| `rerank.enabled` / `rerank.model` | Cross-encoder rerank on/off (on by default) + the model. |
| `confluence.baseUrl` | Human-facing site URL (API calls go through the gateway, not this). |
| `confluence.rootPageId` / `functionalRootPageId` | The two Confluence tree roots to crawl. |
| `confluence.cloudId` | Atlassian cloud id — REST calls go through the `api.atlassian.com` gateway. |
| `confluence.excludedTitles` | Page titles skipped entirely during crawl. |
| `confluence.emailEnvVar` / `apiTokenEnvVar` | Names of the env vars holding the creds (ingest only). |
| `confluence.emailValue` / `apiTokenValue` | Inline cred override (checked **before** the env var); keep `null` in the committed example. |
| `repos.siblingScan` | Auto-discover sibling repos shipping `.rag/ingest.json` (default `true`). |
| `repos.siblingRoot` | Optional override of the folder scanned for siblings (`null` = parent of this repo). |

Repo **code** specs are repo-owned `.rag/ingest.json` files, auto-discovered from siblings (see
"Central code ingestion") — there are no central specs or checkout paths in `config.json`.

## Scope

Read-only against Confluence for docs. Code is ingested from a repo's checkout (never imported as a
library); no swagger ingestion (contract content is authored directly in Endpoint Documents).
