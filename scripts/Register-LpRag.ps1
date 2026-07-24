<#
.SYNOPSIS
    Register the lp-uworld-rag MCP server for Claude Code + Cowork AND Claude Desktop chat, using
    this clone's absolute venv path.

.DESCRIPTION
    The script runs from inside the repo, so it knows the absolute path to this clone's venv python
    and writes that everywhere (no ${CLAUDE_PROJECT_DIR}/relative-path resolution issues):
      1. Claude Code + Cowork: `claude mcp add --scope user` (~/.claude.json) -- works from any dir.
      2. Claude Desktop chat: merges the entry into claude_desktop_config.json. Prompts for that
         file's path (with guidance on where to find it in the app), backs it up, and preserves all
         existing settings -- only the "lp-uworld-rag" entry is added/replaced. You confirm first.

    Prerequisite: run setup.ps1 first (creates the .venv).
    After running: open a NEW Claude Code session; fully quit and reopen Claude Desktop; check /mcp.

.PARAMETER Config
    Full path to claude_desktop_config.json. If omitted, you're prompted (with a default offered).

.PARAMETER Force
    Skip the confirmation prompt (for automation).

.EXAMPLE
    .\scripts\Register-LpRag.ps1
.EXAMPLE
    .\scripts\Register-LpRag.ps1 -Config "C:\Users\me\AppData\Roaming\Claude\claude_desktop_config.json" -Force
#>
param(
    [string]$Config,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent

# --- prerequisite: this clone's venv python (the absolute path we register) -------------------
# The script runs from inside the repo, so it KNOWS the absolute venv path -- we write that
# everywhere, avoiding ${CLAUDE_PROJECT_DIR}/relative-path resolution issues entirely.
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "No venv at $venvPython -- run setup.ps1 first."
}

# --- register for Claude Code + Cowork: user scope, absolute path -----------------------------
# Code (and Cowork, which is Code under the hood) read ~/.claude.json user scope, NOT the Desktop
# config. Registering the absolute path here makes it work from ANY directory (no reliance on the
# project .mcp.json's relative path, which only resolves when the session is rooted at the repo).
if (Get-Command claude -ErrorAction SilentlyContinue) {
    try { claude mcp remove lp-uworld-rag --scope user 2>$null | Out-Null } catch {}
    claude mcp add lp-uworld-rag --scope user -- $venvPython -m lp_uworld_rag mcp
    Write-Host "OK: registered for Claude Code + Cowork (user scope) -> $venvPython" -ForegroundColor Green
} else {
    Write-Host "NOTE: 'claude' CLI not found -- skipped Code/Cowork user-scope registration." -ForegroundColor Yellow
    Write-Host "      Run this once Claude Code CLI is installed:"
    Write-Host "      claude mcp add lp-uworld-rag --scope user -- `"$venvPython`" -m lp_uworld_rag mcp"
}

# --- resolve the Desktop config path: helper text -> prompt (default offered) -----------------
if (-not $Config) {
    $default = Join-Path $env:APPDATA 'Claude\claude_desktop_config.json'
    Write-Host "Find your Claude Desktop config file path:" -ForegroundColor Cyan
    Write-Host "  In Claude Desktop:  Settings  ->  Developer  ->  Edit Config"
    Write-Host "  (that opens the folder containing 'claude_desktop_config.json' -- copy its full path)"
    Write-Host ""
    Write-Host "Default location on this machine:" -ForegroundColor DarkGray
    Write-Host "  $default"
    Write-Host ""
    $Config = Read-Host "Paste the config file path (or press Enter to use the default above)"
    if (-not $Config) { $Config = $default }
}

# --- build the entry (absolute venv python) ---------------------------------------------------
$entry = [pscustomobject]@{
    command = $venvPython
    args    = @('-m', 'lp_uworld_rag', 'mcp')
}

Write-Host "`nAbout to add this MCP server to:" -ForegroundColor Cyan
Write-Host "  $Config"
Write-Host "Entry ('mcpServers' -> 'lp-uworld-rag'):" -ForegroundColor Cyan
Write-Host (([pscustomobject]@{ 'lp-uworld-rag' = $entry }) | ConvertTo-Json -Depth 10)

# --- clearance: require confirmation before writing -------------------------------------------
if (-not $Force) {
    $ans = Read-Host "`nProceed? Existing config is backed up and preserved. (y/N)"
    if ($ans -notmatch '^[Yy]') { Write-Host "Aborted -- no changes made." -ForegroundColor Yellow; return }
}

# --- load (or create), back up, merge, write --------------------------------------------------
if (Test-Path $Config) {
    Copy-Item $Config "$Config.backup.json" -Force
    $cfg = Get-Content $Config -Raw | ConvertFrom-Json
} else {
    New-Item -ItemType Directory -Force (Split-Path $Config) | Out-Null
    $cfg = [pscustomobject]@{}
}
if (-not ($cfg.PSObject.Properties.Name -contains 'mcpServers')) {
    $cfg | Add-Member -NotePropertyName mcpServers -NotePropertyValue ([pscustomobject]@{})
}
if ($cfg.mcpServers.PSObject.Properties.Name -contains 'lp-uworld-rag') {
    $cfg.mcpServers.'lp-uworld-rag' = $entry
} else {
    $cfg.mcpServers | Add-Member -NotePropertyName 'lp-uworld-rag' -NotePropertyValue $entry
}
# Depth 100 preserves the app's deeply-nested settings; write UTF-8 without BOM.
[System.IO.File]::WriteAllText($Config, ($cfg | ConvertTo-Json -Depth 100))

Write-Host "`nOK: lp-uworld-rag written to $Config" -ForegroundColor Green
if (Test-Path "$Config.backup.json") {
    Write-Host "     backup: $Config.backup.json"
    Write-Host "     revert: Copy-Item `"$Config.backup.json`" `"$Config`" -Force"
}
Write-Host "`nNext:" -ForegroundColor Cyan
Write-Host "  * Claude Code / Cowork: open a NEW session (any folder) -- lp-uworld-rag is registered (user scope, absolute path)."
Write-Host "  * Claude Desktop chat:  fully quit and reopen the app."
Write-Host "  * Verify either with /mcp, or: claude mcp get lp-uworld-rag"
