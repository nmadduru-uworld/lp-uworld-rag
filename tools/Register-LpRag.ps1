<#
.SYNOPSIS
    Register the lp-rag skill + MCP server at USER level, so any Claude Code session
    (any directory, any repo) can query the LP knowledge base.

.DESCRIPTION
    Two steps:
      1. Copy the lp-rag skill to %USERPROFILE%\.claude\skills\lp-rag (user-level skills are
         available in every session, not just inside this repo).
      2. Register the MCP server user-scoped via `claude mcp add --scope user`, pointing at THIS
         clone's venv python (absolute path derived from the repo root -> portable per clone).

    Prerequisite: run setup.ps1 first (creates .venv and builds/ingests the index).

.EXAMPLE
    .\tools\Register-LpRag.ps1
#>
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No venv at $venvPython -- run setup.ps1 first."
}

# 1. skill -> user level
$src = Join-Path $repoRoot ".claude\skills\lp-rag"
$dst = Join-Path $env:USERPROFILE ".claude\skills\lp-rag"
New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
Copy-Item $src $dst -Recurse -Force
Write-Host "OK: skill installed to $dst" -ForegroundColor Green

# 2. MCP server -> user scope (idempotent: remove any previous registration first)
try { claude mcp remove lp-uworld-rag --scope user 2>$null | Out-Null } catch {}
claude mcp add lp-uworld-rag --scope user -- $venvPython -m lp_uworld_rag mcp
Write-Host "OK: MCP server 'lp-uworld-rag' registered user-scoped -> $venvPython" -ForegroundColor Green

Write-Host "`nDone. Open a NEW Claude Code session anywhere and ask an LP question"
Write-Host "(or invoke /lp-rag) -- the skill and mcp__lp-uworld-rag__* tools will be available."
