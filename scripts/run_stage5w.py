"""Run Stage 5W (docs/STAGE5W_SPEC.md): the Stage 5 test T1 (low-T minus high-T excess alpha) on the widened flag, returns through 2025.

python scripts/run_stage5w.py [--reps 5000]      # writes outputs/results/stage5w_results.json
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_extend as EXT
import gg_lossrate as LR
import gg_stage5 as S5
import gg_tests as GT
import gg_variants as GV

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5000)
ap.add_argument("--waive-viability", action="store_true", help="Amendment 1: waive the Stage 5 placebo-pool viability rule")
a = ap.parse_args()
t0 = time.time()
mins = lambda: f"{(time.time() - t0) / 60:.1f} min"

ev = LR.flags(GV.load_eligible_events(2004, 2025, "validation/candidate_labels.csv"))
db = G.WRDS(refresh=False)
crsp, mkt, _ = G.load_crsp(db, ev.permno.unique(), 2004, 2025)
crsp, mkt = EXT.extend_returns(db, crsp, mkt, ev.permno.unique())
fac_all = GT.load_factors(db)
cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
cal = cal[np.isin(cal, fac_all.index.values)]
fac = fac_all.reindex(pd.DatetimeIndex(cal))
use = ev[ev.elig].copy()
win = GT.build_windows(use, crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
use["has_win"] = use.event_id.isin(win.keys())
flagged = set(use[(use.flag == 1) & use.has_win].event_id)
print(f"eligible {len(use):,} | flagged with window {len(flagged)} ({mins()})", flush=True)

scores = pd.read_pickle(S5.default_paths()["scores"])
meta = S5.score_meta(scores)
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
base = use[["event_id", "year", "ann_found"]].copy()
events = S5.attach_event_T(base, use[["event_id", "gvkey", "dep_ann_final"]], scores, idx, meta)
print(f"events with T: {int(events['T'].notna().sum()):,} of {len(events):,}; flagged with T: {int(events[events.event_id.isin(flagged)]['T'].notna().sum())} ({mins()})", flush=True)

if a.waive_viability:
    import gg_transparency as _T
    _real = _T.group_viability
    _T.group_viability = lambda *args, **kw: {**_real(*args, **kw), 'ok': True, 'reasons': []}
cuts = (2.0, 3.0)                                              # the Stage 5 cut points, kept fixed (spec section 1)
r = S5.run_t1(win, events, fac, len(cal), flagged, cuts=cuts, reps=a.reps, seed=0, n_boot=1000, with_spread=True, with_sens=False)
print({k: (v if not isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
fz = use[use.exec_name.str.contains("Bisignano", na=False) & (use.conm == "FISERV INC")]
r["fiserv"] = {"flagged": int((fz.flag == 1).sum()) if len(fz) else 0, "T": float(events[events.event_id.isin(fz.event_id)]["T"].iloc[0]) if len(fz) and events[events.event_id.isin(fz.event_id)]["T"].notna().any() else None}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


(G.RESULTS / "stage5w_results.json").write_text(json.dumps(clean({"spec": "docs/STAGE5W_SPEC.md", "T1_widened": r}), indent=1), encoding="utf-8")
print("wrote stage5w_results.json", mins(), flush=True)
