# Getting started with lp-uworld-rag

A newcomer's guide: what this is, the vocabulary, how it works end to end, how it's laid out, and
how to tell it's working. If you just want to run it, the one-command setup is in the
[README](../README.md#quick-start-one-command); this doc is the "understand it" companion.

---

## What it is & why it exists

The UWorld Learning Platform's knowledge lives in two disconnected places:

- **Confluence** documents the platform functionally and technically (feature write-ups, endpoint
  contracts, controller/DB context).
- **The actual C# repos** hold the implementation.

Answering a real question — *"why does this endpoint return null for a valid id?"* — normally means
a human hopping between Confluence pages **and** several source files by hand. And every repo that
wanted code search used to clone a whole RAG tool in-tree, which drifts.

lp-uworld-rag connects the two: it retrieves the relevant **docs**, figures out **which repo** the
question is about from the doc hit itself, and pulls the relevant **code** too — one assembled
answer. Ingestion is centralized (one engine); a repo plugs in with a small JSON spec + a checkout
path, no in-tree tool.

---

## Glossary

| Term | Meaning |
| --- | --- |
| **chunk** | The unit of retrieval: a capped, self-contained slice of a page or a code file, embedded as one vector. |
| **stable id** | A durable, human-readable id baked into a doc's Confluence metadata: `ep::<repo>::<operation>` (endpoint), `ctrl::<repo>::<controller>` (controller), `db::<store>::<collection>` (data store). It's the **join key** for citations and the **routing key** for code (the `<repo>` segment says which repo the doc is about). Feature/Technical Hub pages have no scheme and fall back to their Confluence `page_id`. |
| **doc_type** | What kind of doc a chunk is: `feature-hub`, `technical-hub`, `endpoint`, `controller-context`, `db-collection` (+ a few index pages). Drives quotas and which collection it lands in. |
| **citation** | A hit's linked-but-not-inlined neighbors, returned as `{id, title, doc_type}` only — "cite, don't inline". Call `expand(id)` to pull the full text on demand. Keeps responses small. |
| **RRF fusion** (`reciprocal_rerank`) | How dense (embedding) and lexical (BM25) results are merged: each list contributes by *rank*, and the reciprocal-rank scores are summed. Robust when the two retrievers disagree. |
| **rerank** | A cross-encoder (`ms-marco-MiniLM`) re-scores the fused candidates against the query before truncation — fixes small-pool RRF scores outranking genuinely better hits. |
| **quota** | A guaranteed minimum number of slots per `doc_type` so, e.g., an endpoint-heavy query still surfaces its controller/DB context. |
| **routing** | Deriving which repo(s) a question is about — purely by reading the `<repo>` segment of the doc hits' stable ids. **No intent classification.** No match → fall back to querying every registered repo (flagged, never silently empty). |
| **index vs serve mode** | The two ways a repo's code-RAG plugs in: **index** = the orchestrator reads the repo's Chroma store directly, in-process (recommended); **serve** = the orchestrator spawns the repo's own MCP server. See the [contract](repo-rag-contract.md). |

---

## How it works, end to end

```
Confluence (2 roots)                          repo checkout (C#)
      │  crawl()                                    │  ingest-code
      ▼                                             ▼
  parse metadata → classify doc_type          tree-sitter chunk (class overview + methods)
      │  section-split (endpoint/db)               │  + markdown chunker (CLAUDE.md / rules)
      ▼                                             ▼
  embed (nomic-embed-text)                     embed (CodeRankEmbed)
      ▼                                             ▼
  chroma_functional / chroma_technical         code_stores/<repoKey>
                     ╲                             ╱
                      ╲          query            ╱
                       ▼                          ▼
        dense + BM25  →  RRF fuse  →  quota  →  rerank  →  cite
                       │
                       │  deep_query only:
                       ▼
     read the hit's stable id → route to <repo> → query that repo's code-RAG → assemble
```

Retrieval is the same shape on both sides (dense + BM25 → RRF → quota → rerank); only the ranking
*policy* and the result fields differ. `deep_query` chains the two: docs first, then the repo its
doc hits point at.

---

## The store map

Everything is a local, on-disk Chroma store — no server, no cloud vector DB. **One Chroma client
directory per logical store** (the same pattern the code side uses):

| On disk | Chroma collection | Holds | Embedding model |
| --- | --- | --- | --- |
| `chroma_functional/` | `docs_functional` | Feature Hubs (business "why") | `nomic-embed-text-v1.5` |
| `chroma_technical/` | `docs_technical` | Technical Hubs, endpoints, controllers, data-stores catalog | `nomic-embed-text-v1.5` |
| `code_stores/<repoKey>/` | `code_<repoKey>` | one repo's code + rules | `CodeRankEmbed` |

Each store carries its own docstore (for BM25) and its own delta-ingest state, so any one can be
wiped/rebuilt/backed up on its own. **Functional and technical are independently usable** — scope a
query with `query --collection functional` or `--collection technical`. (They're kept as two
separate DBs deliberately; a single-DB-two-collections layout is possible but adds no capability —
see the repo-rag-contract doc's roadmap notes.)

> `docs_functional` is expected to be **empty/tiny** today: the Feature Hub pages are largely
> not-yet-authored (404), and the Technical Hubs that link to them say "functional hub — link
> pending". An empty functional collection is *working as designed*, not a bug.

---

## How modular this is (onion layers)

Dependencies point **inward** — inner layers never import outer ones:

- **Core — the `common/` package (reusable, depends on nothing above)** — `common/retrieval_engine.py`
  (embed model, Chroma client, docstore, BM25, RRF fusion retriever, rerank), `common/tokens.py`
  (token counting), `common/store_sync.py` (content-hash delta + docstore persist),
  `common/overrides.py` (the L1-override loader), plus the chunk id-hash and result schema.
- **Use-case** — the ingest runner (`ingest.py`) and the `retrieval/` package:
  `retrieval/docs_index.py` (docs query assembly), `retrieval/code_index.py` (repo-code query
  assembly), and the cross-repo `retrieval/orchestrator.py`.
- **Adapters** — `confluence_client.py` / `confluence_reader.py` (Confluence), and the
  `repo_ingest/` chunkers + retriever backends (code).
- **Wiring** — `config.py`/`config.json`, `repos/<key>/ingest.json` specs, repo `rag-manifest.json`.

(`common/` and `retrieval/` are real packages — the layering above is literally the directory
structure, not just a diagram.)

The **important/key** pieces are pluggable while the core stays shared. Today the **code** side is
fully pluggable: a repo picks `index` or `serve` mode and can override the chunker or the retriever
through a two-level ladder (below), adding nothing to the core. (A matching per-tier seam for the
*docs* side — for when functional docs eventually need different ingestion, e.g. no metadata line —
is noted as future work in the [contract](repo-rag-contract.md); not built yet.)

---

## The three tiers, one section each

Same four questions for each — *what / how ingested / how retrieved / when you'd reach for it.*

### Functional docs — the business "why"
- **What:** Feature Hub pages — the product-level purpose of a feature.
- **Ingested:** crawled from the functional Confluence root (`functionalRootPageId`); `doc_type`
  `feature-hub`; whole-page chunks; embedded with `nomic-embed-text-v1.5`. Public id = the Confluence
  `page_id` (no `ep::`/`ctrl::` scheme).
- **Retrieved:** `query --collection functional`.
- **When:** "what is this feature for / why does it behave this way at the product level." (Sparse
  today — Hubs mostly unauthored.)

### Technical docs — the "what & where"
- **What:** Technical Hubs, Endpoint Documents, Controller Context, the Data Stores catalog, and
  index pages.
- **Ingested:** crawled from `rootPageId`; `doc_type` `technical-hub`/`endpoint`/
  `controller-context`/`db-collection` (+ index types). Endpoint and db-collection pages are
  **section-split** (an overview parent + one leaf per section, small-to-big); the rest are
  whole-page. Embedded with `nomic-embed-text-v1.5`. Carries the **stable ids**
  (`ep::<repo>::<op>`, `ctrl::<repo>::<name>`, `db::<store>::<collection>`) that power citations and
  routing.
- **Retrieved:** `query --collection technical` (or just `query` for both docs collections).
- **When:** "what's the contract / which controller / which DB collections / how do these endpoints
  relate."

### Code — the actual implementation
- **What:** a repo's C# source + its `CLAUDE.md`/rules.
- **Ingested:** by `ingest-code` from a **repo checkout** (not Confluence). A tree-sitter C# chunker
  produces a class overview chunk + one chunk per method; a markdown chunker handles rules.
  `source_type` `code`/`rule`, `layer` controller/service/repository/entity/utility. Embedded with
  the code-specific `CodeRankEmbed`. Ids are content hashes; each chunk carries `file_path`,
  `heading`, and line spans.
- **Retrieved:** `query-code --repo <key>` directly, or automatically as the **second stage of
  `deep_query`** once a doc hit's `ep::<repo>::` id routes to it.
- **When:** "show me the code behind this endpoint / where is this logic." This is the tier that
  plugs in per the contract.

---

## Models & first run

First ingest/query downloads models from Hugging Face (cached under your HF home afterward):

| Model | Used for | Notes |
| --- | --- | --- |
| `nomic-ai/nomic-embed-text-v1.5` | docs embeddings | `trustRemoteCode: true` (runs the model's own code) |
| `nomic-ai/CodeRankEmbed` | code embeddings | code-specialized; `trustRemoteCode: true` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | rerank | small cross-encoder |

`device: "auto"` uses CUDA if available, else CPU. First run is slower (download + model load,
tens of seconds); afterward the MCP server loads them once and keeps them warm for its lifetime.

---

## Config reference (`config.json`)

Copy `config.json.example` → `config.json` (the setup script does this). Fields:

| Section / field | What it does |
| --- | --- |
| `embed.model` | Docs embedding model. |
| `embed.trustRemoteCode` | Allow the embedding model to run its own remote code (nomic needs it). |
| `embed.queryPrefix` / `embed.textPrefix` | Asymmetric task prefixes nomic wants for query vs document. |
| `embed.device` | `"auto"` / `"cuda"` / `"cpu"`. |
| `store.functionalPersistDir` / `store.technicalPersistDir` | The two docs Chroma directories. |
| `collections.functional` / `collections.technical` | The Chroma collection names inside those dirs. |
| `collections.functionalEmbedModel` / `technicalEmbedModel` | Optional per-collection embed-model override (`null` = use `embed.model`). |
| `retrieval.topK` | Results returned per query (default 5). |
| `retrieval.poolSize` | Candidate pool fetched before quota/rerank trims to `topK`. |
| `retrieval.fusionMode` | RRF mode (`reciprocal_rerank`). |
| `retrieval.numQueries` | 1 = no query expansion (no LLM call). |
| `retrieval.quotas` | Min guaranteed slots per `doc_type` in the technical collection. |
| `rerank.enabled` | Cross-encoder rerank on/off (on by default). |
| `rerank.model` | The rerank cross-encoder. |
| `confluence.baseUrl` | Human-facing site URL (not used for API calls — those go through the gateway). |
| `confluence.rootPageId` / `functionalRootPageId` | The two Confluence tree roots to crawl. |
| `confluence.manifestPath` | Optional audit-only page-id manifest cross-check (`null` = disabled). |
| `confluence.excludedTitles` | Page titles skipped entirely during crawl. |
| `confluence.cloudId` | Atlassian cloud id — REST calls go through the `api.atlassian.com` gateway. |
| `confluence.emailEnvVar` / `apiTokenEnvVar` | Names of the env vars holding the creds. |
| `confluence.emailValue` / `apiTokenValue` | Inline cred override (checked **before** the env var); keep `null` in the committed example. |
| `repos.manifests` | Paths to `serve`/legacy repos' `rag-manifest.json`. |
| `repos.checkouts` | Machine-local `{ repoKey: checkout path }` for `ingest-code` (gitignored). |

---

## How do I know it worked?

```
python -m lp_uworld_rag status
```
Healthy output is a per-collection chunk count, e.g. `technical | endpoint | 30 chunks` … `total | | 76 chunks`.
A small/empty `functional` line is fine (see the store-map note).

```
python -m lp_uworld_rag repos --validate
```
Healthy: `[PASS] reports (backend=index)` with `manifest: ok / rag_status: ok / query_rag probe: ok`.

```
python -m lp_uworld_rag eval
```
Runs 5 tiers (metadata integrity → resolve check → retrieval quality → expand → routing) and ends
in `RESULT: PASS`. A tier-3 "NEAR"/"FAIL" means retrieval quality slipped, not necessarily a crash.

---

## Add your repo in N steps

1. Create `repos/<repoKey>/ingest.json` (inherits `repos/common.json`; override `sourceDirs`,
   `language`, excludes as needed). **`<repoKey>` must equal the `<repo>` segment in that repo's
   Confluence stable ids** (`ep::<repoKey>::…`) — that's how routing finds it.
2. Point `repos.checkouts.<repoKey>` in your `config.json` at the repo's checkout on your machine.
3. `python -m lp_uworld_rag ingest-code --repo <repoKey>` (or `--all`).
4. `python -m lp_uworld_rag repos --validate` — confirm `[PASS] <repoKey> (backend=index)`.

For a repo whose retrieval doesn't fit the built-in engine, use `serve` mode or an override (below).
Full interface: [repo-rag-contract.md](repo-rag-contract.md).

---

## Overriding the shared/common code

Two seams, same two-level ladder — **L0** a built-in registry key (no code), **L1** one `.py` file
on lp-uworld-rag's venv (never an in-repo tool). Both resolved by `common/overrides.load_factory`. The core
engine still owns ids/embedding/persistence/delta — you override only *how source becomes chunks* or
*how a store is queried*.

**Chunker override** — in `repos/<key>/ingest.json`: `"chunker": "D:/path/to/my_chunker.py"`. The
file exposes a module-level `factory` (or `get_factory()`):
```python
class MyChunkerFactory:
    def create(self, root, source_dirs, exclude, **opts):
        return MyChunker(root, source_dirs, exclude)   # has .read_all() -> iterable of CodeChunk
factory = MyChunkerFactory()
```

**Retriever override** — in the manifest's `index` block: `"retriever": "D:/path/to/my_retriever.py"`.
```python
class MyRetrieverFactory:
    def create(self, persist_dir, index_cfg):
        return MyRetriever(persist_dir, index_cfg)     # has .query(question, top_k, file_hints) -> {"chunks": [...]}
factory = MyRetrieverFactory()
```
L0 examples: `"chunker": "csharp"` (or `"markdown"`/`"generic"`) uses the built-ins, no file needed.

---

## One Confluence reader, not two

Functional and technical are **not** two readers. `confluence_reader.crawl()` walks *both* roots in
one pass, and `read_all()` parses each page's metadata, classifies its `doc_type`, section-splits,
and resolves cross-references (an endpoint pointing at its controller / DB collections / feature).
The functional-vs-technical split is a downstream routing decision by `doc_type` — not a second
reader. This is deliberate: shared parsing/classification, and cross-tree references (a technical
endpoint referencing a feature whose hub lives in the functional tree) resolve in a single crawl.

---

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `Missing Confluence email/API token` | Env vars not set (or `confluence.emailValue`/`apiTokenValue` not filled). Re-run `setup.ps1`, or `setx CONFLUENCE_EMAIL/CONFLUENCE_API_TOKEN`. |
| First run hangs "downloading" | Model download from Hugging Face — first-run only; let it finish, it's cached after. |
| `dependencies not installed` | Run `pip install -e .` (or `setup.ps1`). |
| `docs_functional` empty | Expected — Feature Hubs largely unauthored (404). Not a failure. |
| `ingest-code` says a repo can't start | Its `repos.checkouts.<key>` isn't set or the path doesn't exist on this machine. Set it, or run `setup.ps1 -SkipCode` to skip code entirely. |
| `query-code` returns `[]` | That repo's code store hasn't been ingested — run `ingest-code --repo <key>`. |
| Retrieval looks wrong after a chunker/tagging change | Re-run `ingest --full` (and `ingest-code --repo <key> --full`) to rebuild from scratch. |

---

## Wiring into Claude Code (or any MCP agent)

`.mcp.json` in the repo points an agent at the stdio server:
```
python -m lp_uworld_rag mcp
```
It exposes `query_rag`, `expand`, `deep_query`, `query_code`, `expand_code`, `rag_status`.

---

## Deep dive

- **[repo-rag-contract.md](repo-rag-contract.md)** — the frozen interface a repo's code-RAG plugs in
  through (manifest, index/serve modes, result schema, override ladder, roadmap).
- **Presentation deck** — `../presentation/lp-uworld-rag-pitch.pptx`, or browse the rendered slides
  `../presentation/slides/slide-01.png` … `slide-16.png` (no PowerPoint needed).
- **Module map** — the table in the [README](../README.md#project-layout).
