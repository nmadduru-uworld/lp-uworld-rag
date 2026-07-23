---
name: lp-rag
description: >-
  Answer a question about the UWorld Learning Platform by retrieving from the lp-uworld-rag
  knowledge base (Confluence functional + technical docs, and repo code) over its MCP server.
  Use whenever the user asks why/how an LP endpoint, feature, controller, or data store behaves,
  wants the contract or DB collections for an endpoint, needs the code behind a doc, or is doing a
  bugfix / enhancement / new-feature investigation on the platform. First asks which knowledge
  tier(s) the question needs (functional / technical / code / everything), then queries only those.
---
# lp-rag — retrieve LP functional/technical docs (and code) to resolve a question

This skill answers a Learning Platform question by calling the **`lp-uworld-rag`** MCP server. That
server exposes three knowledge tiers, each independently queryable:

- **Functional docs** — the business "why" (Feature Hubs). *Often sparse/empty today.*
- **Technical docs** — contracts, endpoints, Controller Context, the Data Stores catalog (carry the
  stable ids `ep::<repo>::…`, `ctrl::<repo>::…`, `db::<store>::…`).
- **Code** — the actual repo implementation (retrieved per-repo, or routed automatically from a doc
  hit's stable id).

## Preconditions

The MCP server must be connected. Its tools appear as `mcp__lp-uworld-rag__*`
(`query_rag`, `expand`, `deep_query`, `query_code`, `expand_code`, `rag_status`). If they aren't
available, tell the user to ensure the server is configured (`.mcp.json` in the lp-uworld-rag repo)
and, in an interactive session, connected via `/mcp`; then stop — do not fabricate an answer.
Optionally call `rag_status` once to confirm the index is populated.

## Steps

1. **Get the question.** If the user invoked the skill without one, ask for the LP question/task.
2. **Ask which knowledge tier(s) it needs.** Infer a likely default from the wording, then confirm
   with the user using these options (let them pick one or more):

   - **Functional** — product/business intent of a feature.
   - **Technical** — endpoint contract, controller/DI, which DB collections, how endpoints relate.
   - **Code** — the implementation behind it.
   - **Everything** — docs first, then the code they point at (the deep, cross-repo path).

   Inference hints (still confirm): "what does X do / why does the product…" → Functional;
   "contract / request / response / which collections / auth / controller" → Technical;
   "where is it implemented / show me the code / method / repository" → Code;
   "why does endpoint X return … / trace it end to end / bugfix" → Everything.
3. **Query only the chosen tier(s):**

   | Choice                 | Call                                                                     |
   | ---------------------- | ------------------------------------------------------------------------ |
   | Functional             | `query_rag(question, collection="functional")`                         |
   | Technical              | `query_rag(question, collection="technical")`                          |
   | Functional + Technical | `query_rag(question)` (omit `collection`)                            |
   | Code                   | `query_code(question, repo="<repoKey>")` (omit `repo` to search all) |
   | Everything             | `deep_query(question)` — docs → route by stable id → per-repo code  |

   Size retrieval to intent: a narrow bugfix can pass a small `top_k`; scoping a new feature can go
   wider. Pass `file_hints` to `query_code`/`deep_query` when the user names a file or symbol.
4. **Pull cited detail on demand.** Results return linked neighbors as **citations**
   (`{id, title, docType}`), not full text. When a citation is needed to answer, fetch it:
   `expand(id)` for a doc stable id / page_id, or `expand_code(repo, file_path, heading)` for a code
   chunk. Don't ask for everything up front — expand only what the answer requires.
5. **Answer, grounded in what came back.** Cite doc **stable ids** and code **`file:line`** spans so
   the user can verify. If the functional tier returned nothing, say so plainly (Feature Hubs are
   largely unauthored today) rather than guessing. If `deep_query` reports `routing.fallback_used = true`, note that no specific repo matched and results came from a broad search.

## Notes

- **Cite, don't inline:** keep responses tight — surface the top hit(s) in full, reference the rest
  by id, and expand on request.
- **Routing has no intent model:** `deep_query` picks the repo purely from the `<repo>` segment of a
  doc hit's stable id; if the docs don't name a registered repo it falls back to all repos.
- This skill lives in the lp-uworld-rag repo. To use it from any project, copy this folder to
  `~/.claude/skills/lp-rag/` and point `.mcp.json` at the same server command.
