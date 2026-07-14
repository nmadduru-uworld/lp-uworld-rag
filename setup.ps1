<#
.SYNOPSIS
    One-command setup + ingest for lp-uworld-rag.

.DESCRIPTION
    Provisions the project and ingests everything in a single run:
      1. create the .venv (if missing)          5. set Confluence env vars (persisted + this session)
      2. pip install -e .                        6. ingest Confluence docs
      3. copy config.json.example -> config.json 7. ingest every configured repo's code (ingest-code --all)
      4. (creds, below)                          8. repos --validate, then status

    Idempotent: re-running skips venv creation, never clobbers an existing config.json, and the
    delta ingest only re-embeds what actually changed.

    You supply your own Confluence API token (from https://id.atlassian.com/manage-profile/security/api-tokens).
    It is written to your USER environment via setx and into this process only -- never into any
    file in the repo. config.json keeps its emailValue/apiTokenValue null and reads the env vars.

.PARAMETER Email
    Confluence account email. If omitted and CONFLUENCE_EMAIL isn't already set, you're prompted.

.PARAMETER Token
    Confluence API token. If omitted and CONFLUENCE_API_TOKEN isn't already set, you're prompted
    (hidden input).

.PARAMETER Full
    Pass --full to both ingests (rebuild from scratch instead of delta-syncing).

.PARAMETER SkipCode
    Skip the code-ingest step (use on a machine without the repo checkouts on disk).

.PARAMETER DryRun
    Print each step that would run, without executing anything (and without prompting for secrets).

.EXAMPLE
    .\setup.ps1 -Email you@uworld.com -Token abcd1234
.EXAMPLE
    .\setup.ps1 -Full
.EXAMPLE
    .\setup.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$Email,
    [string]$Token,
    [switch]$Full,
    [switch]$SkipCode,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$fullArg = if ($Full) { @('--full') } else { @() }

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-DryRun([string]$msg) { Write-Host "[dry-run] $msg" -ForegroundColor DarkGray }

# -- 1. venv --------------------------------------------------------------------
if (Test-Path $py) {
    Write-Step "venv already present (.venv) -- skipping creation"
} elseif ($DryRun) {
    Write-DryRun "python -m venv .venv"
} else {
    Write-Step "creating .venv"
    python -m venv .venv
}

# -- 2. install -----------------------------------------------------------------
if ($DryRun) {
    Write-DryRun "$py -m pip install -e . --quiet"
} else {
    Write-Step "installing package + dependencies (pip install -e .)"
    & $py -m pip install -e . --quiet
}

# -- 3. config ------------------------------------------------------------------
if (Test-Path (Join-Path $PSScriptRoot 'config.json')) {
    Write-Step "config.json already exists -- leaving it untouched"
} elseif ($DryRun) {
    Write-DryRun "Copy-Item config.json.example config.json"
} else {
    Write-Step "creating config.json from config.json.example"
    Copy-Item 'config.json.example' 'config.json'
}

# -- 4/5. credentials -----------------------------------------------------------
# Resolve email/token from params, then existing env, then an interactive prompt. Persist to the
# user environment (setx) only for values we actually took in here; always export into THIS process
# so the ingest below can authenticate (setx alone doesn't affect the running session).
if ($DryRun) {
    Write-DryRun "resolve CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN (param -> env -> prompt), setx + set for this session (token never printed)"
} else {
    $persist = $false

    if (-not $Email) {
        if ($env:CONFLUENCE_EMAIL) { $Email = $env:CONFLUENCE_EMAIL }
        else { $Email = Read-Host 'Confluence email'; $persist = $true }
    } else { $persist = $true }

    if (-not $Token) {
        if ($env:CONFLUENCE_API_TOKEN) { $Token = $env:CONFLUENCE_API_TOKEN }
        else {
            $secure = Read-Host 'Confluence API token (input hidden)' -AsSecureString
            $Token = [System.Net.NetworkCredential]::new('', $secure).Password
            $persist = $true
        }
    } else { $persist = $true }

    if (-not $Email -or -not $Token) {
        throw 'Confluence email and token are both required to ingest.'
    }

    $env:CONFLUENCE_EMAIL = $Email
    $env:CONFLUENCE_API_TOKEN = $Token
    if ($persist) {
        Write-Step "persisting CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN to your user environment"
        setx CONFLUENCE_EMAIL $Email | Out-Null
        setx CONFLUENCE_API_TOKEN $Token | Out-Null
    } else {
        Write-Step "using CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN already in your environment"
    }
}

# -- 6. docs ingest -------------------------------------------------------------
if ($DryRun) {
    Write-DryRun "$py -m lp_uworld_rag ingest $($fullArg -join ' ')"
} else {
    Write-Step "ingesting Confluence docs"
    & $py -m lp_uworld_rag ingest @fullArg
}

# -- 7. code ingest -------------------------------------------------------------
# ingest-code --all embeds every repo with a repos/<key>/ingest.json spec whose checkout exists on
# this machine (config.json repos.checkouts). It logs+continues past a missing checkout and exits
# non-zero; we surface that as a warning rather than failing the whole bring-up.
if ($SkipCode) {
    Write-Step "skipping code ingest (-SkipCode)"
} elseif ($DryRun) {
    Write-DryRun "$py -m lp_uworld_rag ingest-code --all $($fullArg -join ' ')"
} else {
    Write-Step "ingesting repo code (ingest-code --all)"
    & $py -m lp_uworld_rag ingest-code --all @fullArg
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "code ingest reported a problem for at least one repo (a checkout may be missing on this machine). Docs ingest still succeeded; re-run '$py -m lp_uworld_rag ingest-code --repo <key>' once its checkout is set in config.json, or pass -SkipCode."
    }
}

# -- 8. validate + status -------------------------------------------------------
if ($DryRun) {
    Write-DryRun "$py -m lp_uworld_rag repos --validate"
    Write-DryRun "$py -m lp_uworld_rag status"
    Write-Host "[dry-run] no changes made." -ForegroundColor DarkGray
    return
}

Write-Step "validating registered repos"
& $py -m lp_uworld_rag repos --validate

Write-Step "index status"
& $py -m lp_uworld_rag status

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "  * MCP server for Claude Code / agents:  $py -m lp_uworld_rag mcp"
Write-Host "  * Ask a cross-repo question:            $py -m lp_uworld_rag deep-query `"<your question>`""
