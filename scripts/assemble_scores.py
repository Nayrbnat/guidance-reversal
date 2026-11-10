"""Build outputs/data/stage5_scores.pkl (one row per filing, the form gg_stage5.load_inputs reads) from the phase scoring run.

python scripts/assemble_scores.py"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_phase_run as R
import gg_transparency as T

OUT = G.ROOT / "validation" / "transp_scores"
freeze = json.load(open(G.RESULTS / "stage5_freeze.json"))
reads = []
for p in sorted(OUT.glob("shard*.jsonl")):
    reads += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
rd = pd.DataFrame(reads)
rd = rd[rd["read"] == 1]
st = {}
for line in (OUT / "statuses.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    st[r["unit_id"]] = r["status"]                                         # the latest status per unit
cur = {}
for line in (OUT / "current_hashes.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    cur[r["unit_id"]] = r["sha"]
ps = R.choose_reads(rd, cur, st, freeze["config_hash"])                    # the read on the current text, never the highest score
(OUT / "stale_or_missing_units.json").write_text(json.dumps(ps.loc[ps["status"].isin(["stale_input", "not_scored"]), "accession"].tolist()), encoding="utf-8")

EXT2 = len(sys.argv) > 1 and sys.argv[1] == "--extension2026"      # builds on the 2025 extension; writes stage5_scores_2026.pkl
EXT = len(sys.argv) > 1 and sys.argv[1] in ("--extension", "--extension2026")     # python scripts/assemble_scores.py --extension : writes stage5_scores_2025.pkl, never the original
pt = pd.read_csv(G.CACHE / ("phase_table_2026.csv" if EXT2 else "phase_table_2025.csv" if EXT else "phase_table.csv"), dtype={"gvkey": str, "accession": str, "phase_first": str})
sc = R.scores_from_phases(pt, ps)
if EXT:
    old = pd.read_pickle(G.DATA / ("stage5_scores_2025.pkl" if EXT2 else "stage5_scores.pkl")).set_index("accession")
    import gg_stage10 as S10
    diff = S10.changed_scores(sc, old.reset_index())
    print(f"existing filings {len(old)}, scores changed by the extension: {len(diff)}")
    if len(diff):
        sys.exit("REFUSED: the extension changed an existing score; not writing")
    sc.to_pickle(G.DATA / ("stage5_scores_2026.pkl" if EXT2 else "stage5_scores_2025.pkl"))
else:
    sc.to_pickle(G.DATA / "stage5_scores.pkl")
firsts = pt["phase_first"].nunique()
print(f"phase-first units {firsts}: scored {int(ps['t_final'].notna().sum())}, status counts {ps['status'].value_counts().to_dict()}")
print(f"filings {len(sc)}: with a score {int(sc['t_final'].notna().sum())} ({sc['t_final'].notna().mean():.1%})")
t = sc.drop_duplicates("unit_id")["t_final"].dropna()
print("T distribution (phases):", {int(k): int(v) for k, v in t.value_counts().sort_index().items()}, "mean %.2f sd %.2f" % (t.mean(), t.std()))
print("adjacent-level share:", round(T.bunching(t), 3))
print("config hashes:", sc["config_hash"].dropna().unique())
