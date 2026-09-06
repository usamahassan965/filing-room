# Windows stand-in for the Makefile. Usage: pwsh -File make.ps1 smoke
param([Parameter(Position = 0)][string]$Target = "help")

$py = Join-Path $PSScriptRoot ".conda\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

switch ($Target) {
    "up"    { docker compose up -d }
    "down"  { docker compose down }
    "smoke" { & $py -m filing.cli smoke }
    "probe" { & $py -m filing.cli probe }
    "ingest" { & $py -m filing.cli ingest }
    "corpus" { & $py -m filing.cli corpus }
    "corpus-offline" { & $py -m filing.cli corpus --offline }
    "test"  { & $py -m pytest }
    "lint"  { & $py -m ruff check src tests }
    "fmt"   { & $py -m ruff format src tests }
    "clean" { & $py -m filing.cli cache --clear }
    default {
        Write-Host "up | down | smoke | probe | ingest | corpus | corpus-offline | test | lint | fmt | clean"
    }
}
