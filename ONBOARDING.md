# lp-uworld-rag — Onboarding

Ask Claude Code natural-language questions about the UWorld Learning Platform — endpoints,
features, controllers, and the **Data Stores catalog** (all SQL tables + MongoDB collections) — and
get answers grounded in the Confluence knowledge base and repo code.

There are two ways in. Most people want **Path A**.

---

## Path A — Query only (fastest, ~10 min, no Confluence token)

Use this if you just want to *ask questions*. You download a prebuilt search index instead of
building it. Querying never contacts Confluence, so **no API token is required**.

### Prerequisites
| Need | Notes |
|---|---|
| Windows + PowerShell | 5.1 or 7 both work |
| Python 3.10–3.14, **64-bit, standard build** | not the "free-threaded / t" build (no prebuilt ML wheels). `winget install Python.Python.3.12` if unsure. **Any drive is fine** — setup finds it via the `py` launcher / PATH, not a fixed location |
| Git | to clone the repo |
| Claude Code | the CLI/desktop you're reading this in |

### Steps
1. **Clone the repo as a sibling** of your other UWorld repos (side-by-side under the same parent
   folder — this is what lets it auto-discover repo code specs later):
   ```
   git clone <lp-uworld-rag repo URL>
   ```
2. **Download the prebuilt index** (kept up to date by the maintainer):
   👉 **Latest chroma folders (Google Drive): `<GOOGLE_DRIVE_SHARE_LINK>`**
   Download `lp-uworld-rag-index.zip` and **extract it into the repo root** so you have:
   ```
   lp-uworld-rag/
     chroma_functional/
     chroma_technical/
     code_stores/        (optional — enables code answers)
   ```
3. **Set up the environment** (creates the venv, installs deps, verifies the index):
   ```
   cd lp-uworld-rag
   .\setup.ps1 -QueryOnly
   ```
   > If PowerShell blocks the script (`running scripts is disabled` / execution policy), run it as:
   > `powershell -ExecutionPolicy Bypass -File .\setup.ps1 -QueryOnly` (same for `Register-LpRag.ps1`).
4. **Make it available in every Claude Code session** (skill + MCP server, one time):
   ```
   .\tools\Register-LpRag.ps1
   ```

**Done.** Open a **new** Claude Code session in any folder and ask, e.g.:
> *"What fields does the group-performance MongoDB collection have?"*
> *"Which tables reference FP_ASSIGNMENTS and what are its indexes?"*
> *"How is faculty-led group performance computed in the reports API?"*

---

## Path B — Full setup (build the index yourself)

Use this only if you're the index maintainer or need a fresh build. Requires a **Confluence API
token** (create at <https://id.atlassian.com/manage-profile/security/api-tokens>).

```
cd lp-uworld-rag
.\setup.ps1 -Email you@uworld.com -Token <your-token>
.\tools\Register-LpRag.ps1
```

`setup.ps1` creates the venv, installs, ingests the Confluence docs, ingests configured repo code,
then validates. It's **idempotent** and delta-based — re-running only re-embeds what changed. Flags:
`-Full` (rebuild from scratch), `-SkipCode` (docs only), `-QueryOnly` (Path A), `-DryRun` (print
steps only).

---

## Add YOUR repo's code to the RAG (one file, you own it)

So `deep-query` can pull code from your repo, add **one file at your repo root** and check the repo
out **as a sibling** of lp-uworld-rag — it's auto-discovered, with **no change to lp-uworld-rag**:

```jsonc
// <your-repo>/.rag/ingest.json
{ "repoKey": "reports",                       // MUST match the `repo` in the Confluence doc metadata
  "displayName": "uwwebtech.learningplatform.reports.api",
  "language": "csharp",
  "sourceDirs": ["uwwebtech.learningplatform.reports.api", "…application", "…infrastructure"],
  "sourceExclude": ["bin","obj",".g.cs","Migrations","Properties"] }
```

Then (maintainer, or you if you have a Confluence token): `python -m lp_uworld_rag ingest-code --all`.

---

## Maintainer — refresh & publish the index

When docs or code change, rebuild and re-share so Path-A users get the update:

```
cd lp-uworld-rag
.\setup.ps1 -Full                      # re-ingest docs + code
# zip the built index and upload, replacing the file behind the Drive link:
Compress-Archive -Path chroma_functional, chroma_technical, code_stores -DestinationPath lp-uworld-rag-index.zip -Force
```
Upload `lp-uworld-rag-index.zip` to the shared Google Drive folder (`<GOOGLE_DRIVE_SHARE_LINK>`),
keeping the same link so this doc stays valid.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No supported Python found` | Install a 64-bit standard build 3.10–3.14: `winget install Python.Python.3.12`, reopen the terminal |
| C++ / build errors during install | You're on a **free-threaded ("t")** or 32-bit Python — install the standard 64-bit build instead |
| `lp-rag` tools not showing in Claude Code | Run `.\tools\Register-LpRag.ps1`, then start a **new** session |
| `Missing Confluence API token` | Only ingest needs a token; for Path A use `.\setup.ps1 -QueryOnly` (no token) |
| `status` shows 0 chunks | The chroma folders weren't extracted into the repo root — re-download and unzip there |
| Broken `.venv` | Delete the `.venv` folder and re-run `setup.ps1` (it also self-recreates a partial venv) |
| `running scripts is disabled on this system` | Run via `powershell -ExecutionPolicy Bypass -File .\setup.ps1 …`, or once per session `Set-ExecutionPolicy -Scope Process Bypass`, or unblock the file (`Unblock-File .\setup.ps1`) |
| Your repo's code isn't found | It must ship `.rag/ingest.json` **and** be cloned **beside** lp-uworld-rag (same parent folder). Siblings are required — there's no per-repo config to set |
