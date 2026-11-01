# Create the project virtual environment (.venv) with uv, from the committed lockfile.
#   scripts\setup_env.ps1          core + notebook tooling
#   scripts\setup_env.ps1 -Rag     also the legacy semantic-search extras (sentence-transformers, anthropic)
# Set $env:UV_PROJECT_ENVIRONMENT to put the environment somewhere other than .\.venv.
param([switch]$Rag)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error @"
uv is not installed. Install it first (https://docs.astral.sh/uv/getting-started/installation/), e.g.:
    winget install --id=astral-sh.uv -e
then re-run this script.
"@
}

$extras = @("--extra", "notebook")
if ($Rag) { $extras += @("--extra", "rag") }

uv sync --locked @extras                                    # Python 3.11 is fetched automatically if missing
if ($LASTEXITCODE -ne 0) { throw "uv sync failed" }

uv run --locked python -c "import sys; sys.path.insert(0, 'src'); import gg_labels, gg_tests, gg_econ, guide_and_go; print('environment OK: project modules import with', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) { throw "import check failed" }

@"

Next:
    uv run python tests/test_gg.py                 # unit tests and simulations (no data or WRDS access needed)
    # WRDS credentials go in .env (see README) before running the pipeline, e.g.
    uv run python src/guide_and_go.py build --start 2004 --end 2025
"@ | Write-Host
