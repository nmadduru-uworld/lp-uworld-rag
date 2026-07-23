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

.PARAMETER QueryOnly
    Minimal query-only bring-up: create the venv + install, verify prebuilt chroma_*/ index folders
    are present, print status, and stop. Skips credentials and BOTH ingests (querying never touches
    Confluence). Use after downloading the latest chroma folders (see ONBOARDING.md) -- no token needed.

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
    [switch]$QueryOnly,   # minimal path: venv + install only; skip creds + both ingests. Use when
                          # you've dropped prebuilt chroma_*/ folders in (query needs no Confluence token).
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$fullArg = if ($Full) { @('--full') } else { @() }

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-DryRun([string]$msg) { Write-Host "[dry-run] $msg" -ForegroundColor DarkGray }

# -- 1. venv --------------------------------------------------------------------
# Supported Python range for this project's wheel-sensitive deps (torch/cu126, chromadb,
# tree-sitter): 3.10 <= version < 3.15. Validated on 3.14. A bare `python` on PATH may be
# anything (or a Windows Store stub), so resolve a supported interpreter explicitly.
# An interpreter is a hashtable @{ Exe='py'; Args=@('-3.12') } (launcher) or @{ Exe='python'; Args=@() }.
# We invoke via splatting -- & $exe @args -- and NEVER slice arrays by range. `$arr[1..$arr.Length]`
# is an out-of-bounds range: PowerShell 5.1 silently trims it, but pwsh 7 THROWS
# "Index was outside the bounds of the array", which broke fresh-machine setup under PS7. Splatting
# an empty @() passes zero args and is safe on both.
function Test-PythonOk($exe, $exeArgs) {
    # In range 3.10..3.14 AND a regular (GIL) build -- free-threaded "3.14t" builds have no prebuilt
    # wheels for torch/chromadb, so pip tries to compile C++ from source and fails. sys.version
    # carries a "free-threading build" marker on t-builds. Probe passed as array elements (no quoting).
    $probe = @($exeArgs) + @('-c', 'import sys; print(sys.version)')
    $out = & $exe @probe 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
    $line = "$out"
    if ($line -match "free.?threading") { return $false }
    if ($line -notmatch "^(\d+\.\d+\.\d+)") { return $false }
    $v = [version]$Matches[1]
    return ($v -ge [version]"3.10" -and $v -lt [version]"3.15")
}

function Resolve-Python {
    # Preference order = wheel availability: 3.12/3.11 have the broadest prebuilt-wheel coverage for
    # this project's deps; 3.13/3.14 work but are newer; PATH python last so a stray install doesn't
    # win over a known-good launcher version.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in "3.12", "3.11", "3.13", "3.14", "3.10") {
            if (Test-PythonOk 'py' @("-$ver")) { return @{ Exe = 'py'; Args = @("-$ver") } }
        }
    }
    if ((Get-Command python -ErrorAction SilentlyContinue) -and (Test-PythonOk 'python' @())) {
        return @{ Exe = 'python'; Args = @() }
    }
    throw ("No supported Python found: need 3.10-3.14, standard (non-free-threaded) build.`n" +
           "  Install Python 3.12:  winget install Python.Python.3.12`n" +
           "  or download from https://www.python.org/downloads/  (pick the 64-bit, non-'t' build), then re-run.")
}

$venvDir = Join-Path $PSScriptRoot '.venv'
if ((Test-Path $py) -and -not $DryRun) {
    Write-Step "venv already present (.venv) -- skipping creation"
} elseif ($DryRun) {
    Write-DryRun "python -m venv .venv  (interpreter chosen by Resolve-Python, verified after creation)"
} else {
    if ((Test-Path $venvDir) -and -not (Test-Path $py)) {
        Write-Step "found a broken .venv (no Scripts\python.exe) -- recreating"
        Remove-Item -Recurse -Force $venvDir
    }
    $r = Resolve-Python
    Write-Step "creating .venv with: $($r.Exe) $($r.Args -join ' ')"
    & $r.Exe @($r.Args) -m venv .venv
    if (-not (Test-Path $py)) {
        throw ("venv creation did not produce $py (interpreter: $($r.Exe) $($r.Args -join ' ')).`n" +
               "Delete the .venv folder and re-run, or create it manually: $($r.Exe) $($r.Args -join ' ') -m venv .venv")
    }
}

# -- 2. install -----------------------------------------------------------------
if ($DryRun) {
    Write-DryRun "$py -m pip install -e . --quiet"
} else {
    Write-Step "installing package + dependencies (pip install -e .)"
    # pip emits harmless warnings to stderr (e.g. optional-extra notices). Native-command stderr
    # must not abort the run -- switch off Stop for this call and gate on the real exit code only.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $py -m pip install -e . --quiet
    $pipCode = $LASTEXITCODE
    $ErrorActionPreference = $prevEAP
    if ($pipCode -ne 0) { throw "pip install failed (exit $pipCode). See the output above." }
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

# -- QueryOnly short-circuit ----------------------------------------------------
# Minimal path for a teammate who just wants to QUERY using prebuilt chroma folders (downloaded,
# not ingested). Query/MCP never touch Confluence, so no token is needed and both ingests are
# skipped. We only verify the index is present and (optionally) register the skill.
if ($QueryOnly) {
    $funcDir = Join-Path $PSScriptRoot 'chroma_functional'
    $techDir = Join-Path $PSScriptRoot 'chroma_technical'
    if (-not (Test-Path $techDir) -or -not (Test-Path $funcDir)) {
        Write-Warning "chroma_functional/ and/or chroma_technical/ not found. Download the latest index folders (see ONBOARDING.md) and extract them into $PSScriptRoot, then re-run. (Or run full setup to ingest from Confluence.)"
    } else {
        Write-Step "prebuilt index present (chroma_functional + chroma_technical) -- skipping ingest"
    }
    Write-Step "index status"
    & $py -m lp_uworld_rag status
    Write-Host ""
    Write-Host "Query-only setup complete. To make lp-rag available in every Claude Code session:" -ForegroundColor Green
    Write-Host "  .\tools\Register-LpRag.ps1"
    return
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
