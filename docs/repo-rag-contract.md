# Repo RAG Retrieval Contract (v2)

This is the interface `lp-uworld-rag`'s orchestrator (`lp_uworld_rag/orchestrator.py` +
`lp_uworld_rag/repo_registry.py`) requires from any repo that wants its own code-RAG plugged into
`deep_query`/`query_code`. Each repo owns its ingestion however it likes -- chunking strategy,
embedding model, storage -- this contract only fixes what the orchestrator calls (or reads) and
what shape comes back. **It is frozen**: a repo built against a given version keeps working
against every future orchestrator release unless `contractVersion` changes, and this document is
the only place that happens. If a change here is unavoidable, bump `contractVersion` and update
every onboarded repo's manifest in the same change.

A manifest declares one of two modes:

- **`serve`** (v1+) -- the orchestrator spawns this repo's own MCP server and calls its
  `query_rag`/`rag_status` tools. Full flexibility -- a repo can run any retrieval logic, any
  vector store, any postprocessing -- at the cost of writing and maintaining that server.
- **`index`** (v2+) -- the orchestrator reads this repo's Chroma persist directory directly,
  in-process, using a generalized retrieval engine (`lp_uworld_rag/direct_index.py`) driven
  entirely by config declared in the manifest. No server to write or run -- the repo's own RAG
  tool becomes just an ingestion pipeline. Requires the repo's chunks to use a fixed metadata
  field-name schema (see below); a repo whose chunk schema doesn't fit should use `serve` instead.

A manifest may declare both; **`index` is preferred whenever both are present** (this is the
recommended path -- keep `serve` only as a fallback for manual debugging or until `index` is
proven out). `contractVersion: 1` manifests may only declare `serve` (adding `index` requires
bumping to `2`); `contractVersion: 2` manifests must declare at least one of the two.

## 1. Manifest

Each repo ships a `rag-manifest.json` inside its own RAG tool directory. All paths in it are
resolved relative to the manifest file's own location, not the orchestrator's working directory.

```json
{
  "contractVersion": 2,
  "repoKey": "reports",
  "displayName": "uwwebtech.learningplatform.reports.api",
  "repoRoot": "../..",
  "index": {
    "embed": {
      "model": "nomic-ai/CodeRankEmbed",
      "trustRemoteCode": true,
      "queryPrefix": "Represent this query for searching relevant code: ",
      "textPrefix": "",
      "device": "auto"
    },
    "store": { "persistDir": "chroma", "collection": "code_reports" },
    "retrieval": {
      "topK": 8, "poolSize": 24, "fusionMode": "reciprocal_rerank", "numQueries": 1,
      "priorityOrder": ["code", "entity", "rule"],
      "quotas": { "code": 2, "entity": 1 },
      "minCategoryChars": 250, "minCategoryExtra": 2, "hintBoost": 1.5,
      "categoryMap": {
        "entity": "entity", "controller": "code", "service": "code",
        "repository": "code", "utility": "code", "other": "code"
      },
      "overviewHeadingSuffix": " (overview)"
    }
  },
  "serve": {
    "command": "../../.venv/Scripts/python",
    "args": ["-m", "reports_rag", "mcp"],
    "env": { "RAG_CONFIG": "config.json" }
  }
}
```

- `contractVersion` -- `1` or `2` today; the orchestrator hard-fails registration (with a clear
  message) on any other value, or on a `1` that isn't `serve`-only.
- `repoKey` -- **must equal** the `repo` value baked into this repo's stable ids in Confluence doc
  metadata (`ep::<repoKey>::<op>`, `ctrl::<repoKey>::<name>`). This is how the orchestrator routes
  a doc hit to a code-RAG without any intent classification -- it just reads the id.
- `repoRoot` -- resolved to an absolute path; used by `validate-citations` to confirm a
  `File.cs:line` citation in a doc chunk actually exists on disk.
- `serve.command` / `serve.args` / `serve.env` -- how the orchestrator spawns this repo's MCP
  server as a stdio subprocess. `command`, if it looks like a relative path, is resolved against
  the manifest's own directory (so `../../.venv/Scripts/python` doesn't depend on cwd); a bare
  executable name (e.g. `"python"`) is left for `PATH` to resolve.
- `index.embed` / `index.store` / `index.retrieval` -- see section 2a below.

### Optional metadata keys

Reserved for a repo that wants to escalate a known issue rather than silently under-serve:

- `known_risk` (string) -- a short, human-readable caveat about this repo's code-RAG (e.g. "index
  is 3 weeks stale", "no line-range data yet"). Surfaced by `repos --validate` so it's visible
  without reading the manifest directly.
- `tracked_in` (string) -- a ticket/issue URL for that risk.

## 2a. `index` mode -- direct Chroma read, no server

`index.store.persistDir` (resolved relative to the manifest, e.g. `"chroma"` for a folder sitting
next to `rag-manifest.json`) must be a Chroma `PersistentClient` directory containing:

- the Chroma collection itself (`index.store.collection`), embedded with `index.embed.model` +
  `index.embed.queryPrefix`/`textPrefix` -- these **must exactly match** whatever ingestion used
  to build the vectors, or query embeddings land in the wrong space.
- a `docstore/docstore.json` (a persisted `llama_index` `SimpleDocumentStore`) holding every
  chunk's full node -- used for BM25 and as the source of every chunk's metadata.

Every chunk's metadata must carry these exact field names (the schema `direct_index.py`'s engine
reads):

| field | required | meaning |
| --- | --- | --- |
| `heading` | yes | short label for the chunk (e.g. `ClassName.MethodName`) |
| `source_type` | yes | e.g. `"code"` or `"rule"` |
| `layer` | no | e.g. `"controller"`/`"service"`/`"repository"`/`"entity"` -- absent on non-code chunks |
| `file_path` | yes | repo-relative source path |
| `start_line` / `end_line` | no | 1-based source span |
| `parent_id` | no | this chunk's parent node's id, for the small-to-big join below |

A parent (overview) node additionally carries `parent_summary` on its own metadata -- when a
retrieved leaf's `parent_id` wasn't independently retrieved, its `parent_summary` is joined in at
query time (never duplicated per leaf).

`index.retrieval` mirrors this project's own `RetrievalConfig`/`RerankConfig` shape, plus two
knobs that generalize what used to be hardcoded quota logic in a hand-written pipeline:

- `categoryMap` -- maps a chunk's raw `layer` value to a quota bucket name (any `layer` not in the
  map, or absent, buckets to `"other"`). `quotas`/`priorityOrder` refer to these bucket names.
- `overviewHeadingSuffix` -- a chunk whose `heading` ends with this string is always bucketed
  `"other"` regardless of `layer` -- an overview/class-signature chunk must never consume a
  guaranteed code/entity quota slot ahead of the sibling chunk that actually answers the question.
- `hintBoost` -- multiplier applied to a chunk's fused score when `file_hints` matches its
  `file_path`/`heading` (case-insensitive substring). Always a boost, never a filter (guardrail C1).

## 2b. Served MCP tools (stdio) -- `serve` mode

A conforming repo's `serve.command`/`args` must launch an MCP server (stdio transport) exposing:

```
query_rag(question: str, top_k: int | None = None, file_hints: list[str] | None = None) -> str  # JSON
rag_status() -> str
```

- `file_hints` are **advisory rank boosts only** -- a list of file paths or symbol names the
  orchestrator has reason to believe are relevant (parsed from doc citations). A conforming server
  MUST NOT hard-filter results to only these files (guardrail C1: citations are a boost, never a
  filter). A server that doesn't use `file_hints` at all is still conformant; the orchestrator
  retries without extra kwargs if an older server's tool signature doesn't accept them yet (see
  `repo_registry.RepoRegistry._call_with_fallback`).
- `rag_status()` returns any non-empty human-readable string; used only as a liveness check by
  `repos --validate`.

## 3. Result schema (both modes)

Whether served over MCP or read directly, a query resolves to the same shape (`serve` mode's
`query_rag` returns it as a JSON string; `index` mode's `direct_index.query()` returns it as a
dict that gets validated the same way):

```json
{
  "chunks": [
    {
      "heading": "string",
      "filePath": "string",
      "layer": "string | null",
      "sourceType": "string",
      "content": "string",
      "score": 0.0,
      "startLine": null,
      "endLine": null,
      "parentSummary": null
    }
  ]
}
```

`heading`, `filePath`, `sourceType`, `content`, `score` are required. `layer` is nullable -- a
non-code chunk (e.g. a rule/markdown chunk) legitimately has none. `startLine`, `endLine`,
`parentSummary` are optional (`null`/absent is fine) -- when present, `validate-citations` uses
`startLine`/`endLine` to cross-check a doc citation's line number against what the repo's own RAG
actually returns for that file.

## 4. Enforcement

The orchestrator validates every query result against this schema (pydantic) before using it --
whether it came back over MCP or was read directly off disk -- a malformed response raises with a
clear "which repo, which field" message rather than failing silently downstream.
`python -m lp_uworld_rag repos --validate` runs a standing conformance check per registered repo:
manifest loads and matches `contractVersion` -> backend launches (server spawn for `serve`, or
embedding-model + Chroma load for `index`) -> `rag_status` returns non-empty -> a probe query call
returns a schema-valid response. Its output reports which backend (`index`/`serve`) was actually
exercised.

## Ownership & lifecycle

*(Stub -- fill in as repos onboard.)* Each repo's own team owns its manifest, its server process,
and its ingestion pipeline. Open questions not yet settled: who re-validates a repo's manifest
after a code-RAG upgrade; where `known_risk`/`tracked_in` escalations get triaged; whether
onboarding a new repo needs a review from whoever owns this contract.
