# lp-uworld-rag

Shared functional/technical RAG for the UWorld Learning Platform Confluence knowledge base --
Feature Hubs, Technical Hubs, Endpoint Documents, Controller Context, and the Data Stores catalog.
It never imports another repo's code as a library. It does two things across repos:

- **Ingests repo code centrally** -- a shared engine (`lp_uworld_rag/repo_ingest/`) builds each
  repo's code index by being pointed at that repo's checkout, so a repo needs no in-tree RAG tool
  of its own (see "Central code ingestion" below).
- **Orchestrates at query time** -- `deep_query` routes a doc hit to the repo(s) it's actually
  about and retrieves from their code index too (see "Cross-repo routing" below).

> **New here?** Start with **[docs/getting-started.md](docs/getting-started.md)** — a plain-language
> tour (what/why, glossary, how it works end to end, the three tiers, config reference,
> troubleshooting, "how do I know it worked"). There's also a slide deck under
> [`presentation/`](presentation/) (`lp-uworld-rag-pitch.pptx`, or the rendered
> `presentation/slides/slide-01.png`…`slide-16.png`).

## Setup

### Quick start (one command)

From a PowerShell prompt in the repo root (a Confluence API token comes from
https://id.atlassian.com/manage-profile/security/api-tokens):

```
.\setup.ps1 -Email you@uworld.com -Token <token>
```

That single command creates the `.venv`, `pip install -e .`, copies `config.json` from the example,
sets the two Confluence env vars (persisted to your user environment **and** the current session --
your token is never written to any file in the repo), ingests the Confluence docs, ingests every
configured repo's code (`ingest-code --all`), then runs `repos --validate` and `status`. It's
idempotent: re-running skips venv creation, never clobbers an existing `config.json`, and the delta
ingest only re-embeds what changed. Flags: `-Full` (rebuild from scratch), `-SkipCode` (docs only --
use on a machine without the repo checkouts), `-DryRun` (print the steps without running them).
Omit `-Email`/`-Token` to be prompted (token input hidden).

### Manual setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
cp config.json.example config.json   # defaults match the live space; edit if needed
```

Set two environment variables (a Confluence API token from
https://id.atlassian.com/manage-profile/security/api-tokens):

```
setx CONFLUENCE_EMAIL "you@uworld.com"
setx CONFLUENCE_API_TOKEN "<token>"
```

## Usage

```
python -m lp_uworld_rag ingest [--full]
python -m lp_uworld_rag query "POST faculty-led/group-performance" [--collection technical] [--top-k 5] [--no-siblings]
python -m lp_uworld_rag expand ctrl::reports::FacultyLedPerformanceController
python -m lp_uworld_rag status
python -m lp_uworld_rag mcp        # stdio MCP server for Claude Code / other agents
python -m lp_uworld_rag ingest-code --repo reports [--full]   # ingest a repo's code (or --all)
python -m lp_uworld_rag deep-query "why does POST faculty-led/group-performance return null body"
python -m lp_uworld_rag query-code "group performance date range cap" --repo reports
python -m lp_uworld_rag repos [--validate]
python -m lp_uworld_rag validate-citations
```

`docs_functional` is expected to be **empty** until Feature Hub pages actually exist in Confluence --
both are 404 as of this project's creation; the Technical Hub pages that link to them say so
explicitly ("functional hub -- link pending").

## Project layout

Every module under `lp_uworld_rag/`, by role:

| Module | What it does |
| --- | --- |
| **Entry / config** | |
| `__main__.py` | CLI verbs + dispatch (`ingest`, `query`, `deep-query`, `ingest-code`, `repos`, …) |
| `config.py` | Typed (`pydantic`) load of `config.json` -- embed/store/collections/retrieval/rerank/confluence/repos |
| `mcp_server.py` | FastMCP stdio server exposing `query_rag`/`expand`/`deep_query`/`query_code`/`rag_status` |
| **`common/`** (core — reusable, depends on nothing above) | |
| `common/retrieval_engine.py` | **Shared primitives** -- embed model, Chroma client, docstore, BM25, fusion retriever, rerank (used by both retrieval engines) |
| `common/store_sync.py` | Ingest-time content-hash delta + docstore persist, shared by docs + code ingest |
| `common/tokens.py` | One `count_tokens` (tiktoken `cl100k_base`) shared by `eval` + `orchestrator` |
| `common/overrides.py` | Shared L1-override `importlib` loader behind the chunker + retriever override seams |
| **Docs pipeline** | |
| `confluence_client.py` | Confluence Cloud REST v2 client (children, body → markdown) |
| `confluence_reader.py` | Crawl the two Confluence trees, parse metadata, classify doc_type, section-split |
| `chunker.py` | One capped, Chroma-safe `TextNode` per page/section (content-hash id) |
| `ingest.py` | Crawl → chunk → content-hash **delta** upsert into `docs_functional` + `docs_technical` |
| **`retrieval/`** (query-time engines, build on `common/`) | |
| `retrieval/docs_index.py` | Docs engine (was `index.py`): retrieve → resolve linked ids as citations → quota/rerank/rank |
| `retrieval/code_index.py` | Repo-code "index"-mode engine (was `direct_index.py`): hint-boost, priority order, small-to-big parent join |
| `retrieval/orchestrator.py` | `deep_query`/`query_code`: docs → route by stable-id → each routed repo's code-RAG; score floor + token budget |
| **Repo plug-in** | |
| `repo_registry.py` | Load repo manifests; run each via `serve` (MCP subprocess) or `index` (in-process); conformance validation |
| `repo_ingest/` | Central code ingestion: `spec.py` (per-repo spec deep-merge), `pipeline.py` (checkout → chunks → Chroma), `layer.py`, `chunkers/*` (C#/markdown/generic registry) |
| **Quality** | |
| `eval.py` | 5-tier eval **harness/engine** (metadata integrity, resolve, retrieval quality, expand, routing) |
| `eval_cases.py` | The domain-specific query/expand/routing **fixtures** the harness runs (kept out of the harness so it stays domain-agnostic) |

The package is layered like an onion: **`common/`** (core primitives, no intra-package deps) →
**`retrieval/`** + docs pipeline (build on core) → repo plug-in → `eval`, with `config.py` /
`__main__.py` / `mcp_server.py` at the top. `common/` exists to remove duplication — the two
retrieval engines used to reimplement the same embed/Chroma/BM25/rerank plumbing and the token
counter lived in two places. See [docs/repo-rag-contract.md](docs/repo-rag-contract.md) for the
interface a repo's code-RAG plugs in through, and
[docs/getting-started.md](docs/getting-started.md) for the layered walkthrough.

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

Repo-wise structure:

```
lp-uworld-rag/
  repos/                     # committed, ships with this project
    common.json              #   shared defaults (embed model, retrieval/quotas, excludes)
    <repoKey>/ingest.json    #   per-repo override (language, sourceDirs, ...); deep-merges over common
  code_stores/<repoKey>/     # gitignored -- the built index (Chroma + docstore), derived from repoKey
  config.json                # gitignored -- repos.checkouts: { "<repoKey>": "<abs path to checkout>" }
  lp_uworld_rag/repo_ingest/ # the shared engine: pipeline + per-language chunker registry
```

To onboard a repo: add `repos/<repoKey>/ingest.json` (see `repos/reports/ingest.json`), set its
checkout path under `repos.checkouts` in `config.json`, then:

```
python -m lp_uworld_rag ingest-code --repo <repoKey> [--full]   # or --all for every configured repo
python -m lp_uworld_rag repos --validate                        # confirm it loads + conforms
```

Ingest is a content-hash **delta**: a re-run only re-embeds files whose chunks actually changed and
prunes files that disappeared; `--full` rebuilds. `repoKey` **must equal** the `repo` value in the
Confluence doc metadata (`ep::<repoKey>::...`) -- that's how the orchestrator routes a doc hit to
this index.

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

## Scope

Read-only against Confluence for docs. Code is ingested from a repo's checkout (never imported as a
library); no swagger ingestion (contract content is authored directly in Endpoint Documents).
