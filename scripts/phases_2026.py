"""Extend the phase table (already extended to 2025) to the 2026 filings, WITHOUT touching the existing files: combine the old and new revenue-label reads, triage only the
new MAJOR changes, build the combined phase table, and refuse to write unless every previously existing filing keeps its level, phase and phase_first.

python scripts/phases_2026.py            (writes gg_cache/phase_table_2026.csv and gg_cache/phase_triage_2026.jsonl)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import guide_and_go as G
import gg_phase_run as R
import gg_transparency as T

C = G.CACHE


def read(p):
    return pd.DataFrame([json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()])


old = pd.concat([read(C / "phase_labels.jsonl"), read(C / "phase_labels_2025.jsonl")], ignore_index=True)
new = read(C / "phase_labels_2026.jsonl")
lab = pd.concat([old, new], ignore_index=True).drop_duplicates("accession", keep="first")
lab["gvkey"] = lab["gvkey"].astype(str).str.zfill(6)
print(f"labels: old {len(old)}, new {len(new)}, combined {len(lab)}", flush=True)

old_tri = {**R.load_triage(C / "phase_triage.jsonl"), **R.load_triage(C / "phase_triage_2025.jsonl")}
pairs = R.triage_pairs(lab)
todo = pairs[~pairs.accession.isin(old_tri)]
print(f"MAJOR changes in the combined set: {len(pairs)}, already triaged: {len(pairs) - len(todo)}, new to triage: {len(todo)}", flush=True)
out = C / "phase_triage_2026.jsonl"
if len(todo):
    stats = R.run_triage(todo, out, T.claude_runner("claude-haiku-5-5", T.DEFAULT_FLAGS), workers=3)
    print("triage:", stats, flush=True)
tri = dict(old_tri)
if out.exists():
    tri.update(R.load_triage(out))

pt = R.phase_table(lab, tri)
pt["gvkey"] = pt["gvkey"].astype(str).str.zfill(6)
prev = pd.read_csv(C / "phase_table_2025.csv", dtype={"gvkey": str, "accession": str, "phase_first": str}).set_index("accession")
import gg_stage10 as S10
bad = S10.changed_phases(pt, prev.reset_index())
m = prev
print(f"previously existing filings: {len(m)}, changed by the extension: {len(bad)}", flush=True)
if len(bad):
    print(bad[:10])
    sys.exit("REFUSED: the extension changed an existing filing's phase; not writing")
pt.to_csv(C / "phase_table_2026.csv", index=False)
fresh = pt[~pt.accession.isin(prev.index)]
print(f"new filings: {len(fresh)}, levels {fresh.level.value_counts().to_dict()}, new phase starts to score: "
      f"{fresh.phase_first.nunique()} phase_first units ({int((~fresh.phase_first.isin(prev.phase_first.unique())).sum())} filings under units not scored before)", flush=True)
print("triage pending/unusable among new:", int(fresh.triage_pending.sum()), int(fresh.triage_unusable.sum()))
