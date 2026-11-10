"""Run Stage 12b (docs/STAGE12B_SPEC.md): the Stage 12 out-of-sample test with XGBoost, a neural network and support vector machines.

python scripts/run_stage12b.py --kind xgb [--reps 200]        # one model; writes outputs/results/stage12b_<kind>.json
python scripts/run_stage12b.py --combine                      # Holm across the four models; writes outputs/results/stage12b_results.json
The sample, features, windows and test are identical to scripts/run_stage12.py (the first part of this file is copied from it).
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import guide_and_go as G
import gg_lossrate as LR
import gg_stage5 as S5
import gg_stage11 as S11
import gg_stage12 as S12
import gg_tests as GT
import gg_variants as GV

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["xgb", "mlp", "svm_lin", "svm_poly", "logit", "hgb"])
ap.add_argument("--compact", action="store_true", help="Section 6 addendum: the six-feature set R6")
ap.add_argument("--both", action="store_true", help="run the 19-feature and the six-feature sets in one process")
ap.add_argument("--extend", action="store_true", help="returns through 2025-12-31 (docs/STAGE12B_SPEC.md, Amendment 1)")
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--combine", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
RES = G.RESULTS
mins = lambda: f"{(time.time() - t0) / 60:.1f} min"

# ------------------------------------------------------------------------------------------- sample (as the large-loss test) plus the leak guard
ev = LR.flags(GV.load_eligible_events(2004, 2025, "validation/candidate_labels.csv"))
db = G.WRDS(refresh=False)
crsp, mkt, _ = G.load_crsp(db, ev.permno.unique(), 2004, 2025)
if a.extend:
    import gg_extend as EXT
    crsp, mkt = EXT.extend_returns(db, crsp, mkt, ev.permno.unique())
cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
mk = mkt.drop_duplicates("date").set_index("date").mkt.reindex(pd.DatetimeIndex(cal)).astype(float).values
elig = ev[ev.elig].copy()
win = GT.build_windows(elig, crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
use = elig.join(LR.bhars(win, mk), on="event_id").dropna(subset=["bhar"]).copy()
use["year"] = use.dep_ann_final.dt.year
n0 = len(use)
use = use[(use.year <= 2024) & (use.ndays >= 200)]
late = (use.dep_ann - use.dep_ann_final).dt.days > 1                       # leak guard: first-identified date later than the corrected one
n_late, n_late_flag = int(late.sum()), int((late & (use.flag == 1)).sum())
use = use[~late].copy()
use["y"] = (use.bhar < -0.30).astype(int)
print(f"sample: {n0:,} eligible with a window -> {len(use):,} after year<=2024, >=200 days, leak guard (dropped {n_late}, of which {n_late_flag} flagged) | "
      f"large-loss share {use.y.mean():.1%} | flagged {int(use.flag.sum())} ({mins()})", flush=True)

# ------------------------------------------------------------------------------------------- features
use = S12.market_features(use, crsp, "dep_ann_final")
mcap = np.exp(use.f_lmcap)
F = pd.DataFrame(index=use.index)
for c in ("f_lmcap", "f_ret12", "f_vol"):
    F[c] = use[c]
F["is_ceo"] = (use.role == "CEO").astype(float)
for s in sorted(use.sector.dropna().unique()):
    F["sec_" + s] = (use.sector == s).astype(float)
B = list(F.columns)
R = pd.DataFrame(index=use.index)
R["n_guid_12m"] = use.n_guid_12m.fillna(0)
R["days_since_guid"] = use.days_since_guid.fillna(999)
R["n_unfulfilled"] = use.n_unfulfilled.fillna(0)
R["opt_last"], R["opt_max"], R["guid_raise_last"] = use.opt_last, use.opt_max, use.guid_raise_last
R["sold_post_guid_pct"] = use.sell_val_post_guid.fillna(0) / mcap
R["sold_12m_pct"] = use.sell_val_12m.fillna(0) / mcap
R["n_sells_12m"] = use.n_sells_12m.fillna(0)
R["opp_sell_pct"] = use.opp_sell_val.fillna(0) / mcap
R["abrupt_rule"] = use.abrupt_rule.fillna(0)
for t in ("retirement_planned", "resigned_abrupt", "terminated_forced", "left_for_other_job", "health_personal", "other_unclear", "not_found"):
    R["exit_" + t] = (use.dep_type_rule == t).astype(float)
R["flag_f"] = use.flag.astype(float)
Rc = list(R.columns)
df = pd.concat([use[["year", "y", "bhar", "flag", "guided_unfulfilled", "dep_ann_final", "permno", "gvkey", "event_id"]], F, R], axis=1)
df = df.dropna(subset=["f_lmcap", "f_ret12", "f_vol"]).reset_index(drop=True)
df["q"] = df.dep_ann_final.dt.to_period("Q").astype(str)
print(f"modelling sample {len(df):,} events | baseline features {len(B)} | reversal features {len(Rc)} | test years 2010-2024 ({mins()})", flush=True)
TY = range(2010, 2025)

if a.combine:
    import glob
    res = {}
    for k in ("xgb", "mlp", "svm_lin", "svm_poly"):
        f = RES / f"stage12b_{k}.json"
        if f.exists():
            res[k] = json.loads(f.read_text())
    ps = {k: v["p_one_sided"] for k, v in res.items()}
    holm = GT.holm(ps)
    for k in res:
        res[k]["holm_p"] = float(holm[k])
    (RES / "stage12b_results.json").write_text(json.dumps({"spec": "docs/STAGE12B_SPEC.md", "models": res}, indent=1, default=float))
    for k, v in res.items():
        print(f"{k}: AUC {v['auc_base']:.3f} -> {v['auc_base_plus_r']:.3f}, delta {v['delta']:+.4f} {v['ci95_boot']} | p {v['p_one_sided']:.3f} Holm {v['holm_p']:.3f}")
    sys.exit(0)

Rc_full = list(Rc)
COMPACT = ["n_guid_12m", "sold_post_guid_pct", "opt_last", "n_unfulfilled", "abrupt_rule", "flag_f"]
etag = "ext_" if a.extend else ""
for compact in ([False, True] if a.both else [a.compact]):
    Rc = COMPACT if compact else Rc_full
    tag = etag + ("compact_" if compact else "")
    r = S12.perm_delta_auc(df, B, Rc, "y", a.kind, TY, a.reps, a.seed)
    r["ci95_boot"] = S12.boot_delta_auc(r["pred_base"], r["pred_br"], df.q.values, 1000, a.seed)
    pb, pbr = r.pop("pred_base"), r.pop("pred_br")
    r["n_test_events"] = int(len(pb)); r["n_test_years"] = int(pb.year.nunique())
    (RES / f"stage12b_{tag}{a.kind}.json").write_text(json.dumps(r, indent=1, default=float))
    print(f"P12b {tag}{a.kind} ({len(Rc)} reversal features): AUC base {r['auc_base']:.3f} -> base+R {r['auc_base_plus_r']:.3f}, delta {r['delta']:+.4f} "
          f"(95% boot [{r['ci95_boot'][0]:+.4f}, {r['ci95_boot'][1]:+.4f}]) | permutation p {r['p_one_sided']:.3f} ({a.reps} reps; null mean {r['null_mean']:+.4f}, "
          f"null 95th pct {r['null_q95']:+.4f}) | test events {r['n_test_events']} ({mins()})", flush=True)
    print("by-year delta:", {int(k): round(v, 3) for k, v in r["by_year"].items()}, flush=True)
