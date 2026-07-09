# lp-uworld-rag

Shared functional/technical RAG for the UWorld Learning Platform Confluence knowledge base --
Feature Hubs, Technical Hubs, Endpoint Documents, Controller Context, and the Data Stores catalog.
This project only reads Confluence itself -- it never imports another repo's code. It does,
however, orchestrate across repos at query time: `deep_query` routes a doc hit to the repo(s) it's
actually about and queries their own code-RAG MCP servers too (see "Cross-repo routing" below).

## Setup

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
python -m lp_uworld_rag deep-query "why does POST faculty-led/group-performance return null body"
python -m lp_uworld_rag query-code "group performance date range cap" --repo reports
python -m lp_uworld_rag repos [--validate]
python -m lp_uworld_rag validate-citations
```

`docs_functional` is expected to be **empty** until Feature Hub pages actually exist in Confluence --
both are 404 as of this project's creation; the Technical Hub pages that link to them say so
explicitly ("functional hub -- link pending").

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

### Onboarding a repo

Each repo owns its own ingestion (chunking, embedding, storage) completely -- this project only
enforces the **retrieval interface** a repo's code-RAG must plug in through. That interface is
frozen as the Repo RAG Retrieval Contract (v2): see [docs/repo-rag-contract.md](docs/repo-rag-contract.md).
A manifest picks one of two modes (or both -- `index` wins when both are present):

- **`index`** (recommended) -- the repo just ingests into a Chroma persist dir with the contract's
  fixed metadata field names; the orchestrator reads it directly, in-process, via a generalized
  retrieval engine (`lp_uworld_rag/direct_index.py`). No server to write or run -- the repo's own
  RAG tool becomes mostly an ingestion pipeline (see `reports`' own `rag-manifest.json` for a
  worked example).
- **`serve`** -- the orchestrator spawns the repo's own MCP server and calls `query_rag`/
  `rag_status`. Keep this as a fallback for a repo whose retrieval logic doesn't fit the generalized
  engine (custom postprocessing, a non-Chroma store), or during migration to `index`.

To register a repo, add its `rag-manifest.json` path to `repos.manifests` in `config.json`, then
run `python -m lp_uworld_rag repos --validate` to confirm it launches (or loads) and conforms
before relying on it -- its output names which backend (`index`/`serve`) is actually in use.

`python -m lp_uworld_rag validate-citations` fact-checks every `File.cs:line` citation an ingested
doc chunk carries against a registered repo's actual checkout (and, where the repo's code-RAG
returns line ranges, against what it actually retrieves for that file) -- citations stay a rank
boost everywhere else, this is the one place they're checked as fact.

## Scope

Read-only against Confluence. No code chunking of its own (that's each repo's own concern, see
above), no swagger ingestion (contract content is authored directly in Endpoint Documents).
