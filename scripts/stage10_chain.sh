#!/usr/bin/env bash
# Stage 10 unattended chain (docs/STAGE10_SPEC.md, Amendment 1). Waits for the two long pulls, then: prep -> label 2026 filings -> triage and phases ->
# score -> assemble -> run the two tests once. Stops at the first failure and says where. Never deletes anything.
cd "$(dirname "$0")/.." || exit 1
LOG=gg_cache/stage10_chain.log
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail() { step "FAILED at: $*"; exit 1; }

step "waiting for the insider stream and the Compustat price pull"
until [ -f gg_cache/tr_insiders_2025_06_plus.pkl ] && [ -f outputs/results/secd_vs_crsp_v2.json ]; do sleep "${STAGE10_POLL:-60}"; done
step "pulls finished"

step "prep (validation gate, linking, CRSP for new firms, splice, filings list)"
uv run python scripts/prep_stage10.py >> gg_cache/prep_stage10.log 2>&1 || { tail -n 5 gg_cache/prep_stage10.log | tee -a "$LOG"; fail "prep_stage10"; }

N=$(($(wc -l < gg_cache/phase_needed_2026.csv) - 1))
step "filings to label: $N"
if [ "$N" -gt 0 ]; then
  for pass in 1 2; do
    uv run python src/gg_phase_run.py gg_cache/phase_needed_2026.csv gg_cache/phase_labels_2026.jsonl 8 >> gg_cache/phase_run_2026.log 2>&1 || fail "label pass $pass"
  done
  step "labels done: $(wc -l < gg_cache/phase_labels_2026.jsonl) lines"
  uv run python scripts/phases_2026.py >> gg_cache/phases_2026.log 2>&1 || { tail -n 5 gg_cache/phases_2026.log | tee -a "$LOG"; fail "phases_2026"; }
  uv run python - <<'EOF' || fail "needed list"
import pandas as pd
a = pd.read_csv("gg_cache/phase_needed_all.csv", dtype=str)
b = pd.read_csv("gg_cache/phase_needed_2026.csv", dtype=str)
c = pd.concat([a, b]).drop_duplicates("accession")
c.to_csv("gg_cache/phase_needed_all2026.csv", index=False)
print(len(a), len(b), len(c))
EOF
  step "scoring new phase starts with the frozen scorer"
  uv run python scripts/score_phases.py 4 phase_table_2026.csv phase_needed_all2026.csv >> gg_cache/score_phases_2026.log 2>&1 || fail "score_phases"
  step "assembling scores (refuses to write if any existing score changes)"
  uv run python scripts/assemble_scores.py --extension2026 >> gg_cache/assemble_2026.log 2>&1 || { tail -n 5 gg_cache/assemble_2026.log | tee -a "$LOG"; fail "assemble_scores"; }
else
  uv run python - <<'EOF' || fail "copy scores"
import pandas as pd
import sys
sys.path.insert(0, "src")
import guide_and_go as G
pd.read_pickle(G.DATA / "stage5_scores_2025.pkl").to_pickle(G.DATA / "stage5_scores_2026.pkl")
EOF
fi

step "running Stage 10 (once)"
uv run python scripts/run_stage10.py > gg_cache/run_stage10.log 2>&1 || { tail -n 8 gg_cache/run_stage10.log | tee -a "$LOG"; fail "run_stage10"; }
step "CHAIN DONE"
