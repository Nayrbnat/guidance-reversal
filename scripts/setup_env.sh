#!/usr/bin/env bash
# Create the project virtual environment (.venv) with uv, from the committed lockfile.
#   scripts/setup_env.sh          core + notebook tooling
#   scripts/setup_env.sh --rag    also the legacy semantic-search extras (sentence-transformers, anthropic)
# Set UV_PROJECT_ENVIRONMENT=/some/path to put the environment somewhere other than ./.venv.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
uv is not installed. Install it first (https://docs.astral.sh/uv/getting-started/installation/), e.g.:
    macOS / Linux:  brew install uv        or   pipx install uv
    Windows:        winget install --id=astral-sh.uv -e
then re-run this script.
EOF
    exit 1
fi

extras=(--extra notebook)
[[ "${1:-}" == "--rag" ]] && extras+=(--extra rag)

uv sync --locked "${extras[@]}"      # Python 3.11 is fetched automatically if missing

uv run --locked python -c "
import sys; sys.path.insert(0, 'src')
import gg_labels, gg_tests, gg_econ, guide_and_go
print('environment OK: project modules import with', sys.version.split()[0])"

cat <<'EOF'

Next:
    uv run python tests/test_gg.py                 # unit tests and simulations (no data or WRDS access needed)
    # WRDS credentials go in .env (see README) before running the pipeline, e.g.
    uv run python src/guide_and_go.py build --start 2004 --end 2025
EOF
