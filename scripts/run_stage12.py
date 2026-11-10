"""Run Stage 12 (docs/STAGE12_SPEC.md): a rolling-window out-of-sample test with a machine-learning model, and rolling-window stability.

python scripts/run_stage12.py [--hgb-reps 200] [--logit-reps 1000]      # run from the project root; writes outputs/results/stage12_results.json
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
ap.add_argument("--hgb-reps", type=int, default=200)
ap.add_argument("--logit-reps", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--extend", action="store_true", help="returns through 2025-12-31 (docs/STAGE12_SPEC.md, Amendment 1)")
ap.add_argument("--primary-only", action="store_true", help="P12a only; skip the secondary analyses")
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
_tag = "_ext" if a.extend else ""
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
out = {"spec": "docs/STAGE12_SPEC.md", "sample": {"n": int(len(df)), "flagged": int(df.flag.sum()), "large_loss_share": float(df.y.mean()),
                                                    "dropped_leak_guard": n_late, "dropped_leak_guard_flagged": n_late_flag, "n_base_features": len(B), "n_reversal_features": len(Rc)}}

# ------------------------------------------------------------------------------------------- P12a: the confirmatory test
res = {}
for kind, reps in (("hgb", a.hgb_reps), ("logit", a.logit_reps)):
    r = S12.perm_delta_auc(df, B, Rc, "y", kind, TY, reps, a.seed)
    r["ci95_boot"] = S12.boot_delta_auc(r["pred_base"], r["pred_br"], df.q.values, 1000, a.seed)
    pb, pbr = r.pop("pred_base"), r.pop("pred_br")
    r["n_test_events"] = int(len(pb)); r["n_test_years"] = int(pb.year.nunique())
    res[kind] = r
    print(f"P12a {kind}: AUC base {r['auc_base']:.3f} -> base+R {r['auc_base_plus_r']:.3f}, delta {r['delta']:+.4f} (95% boot [{r['ci95_boot'][0]:+.4f}, {r['ci95_boot'][1]:+.4f}]) "
          f"| permutation p {r['p_one_sided']:.3f} ({reps} reps; null mean {r['null_mean']:+.4f}, null 95th pct {r['null_q95']:+.4f}) | test events {r['n_test_events']} ({mins()})", flush=True)
    if kind == "hgb":
        pos_by_year = df[df.year.isin(TY)].groupby("year").y.sum()
        print("   by-year delta:", {int(k): round(v, 3) for k, v in r["by_year"].items()}, flush=True)
out["P12a"] = res
if a.primary_only:
    (RES / f"stage12_results{_tag}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"wrote {RES / ('stage12_results' + _tag + '.json')} ({mins()})")
    sys.exit(0)

# ------------------------------------------------------------------------------------------- secondary
sec = {}
# (1) the flag alone as a score
fa = []
for t in TY:
    g = df[df.year == t]
    pos = int(g.y.sum())
    if pos >= 5 and len(g) - pos >= 5 and g.flag.nunique() > 1:
        fa.append(roc_auc_score(g.y, g.flag))
sec["flag_alone_mean_auc"] = float(np.mean(fa)); sec["flag_alone_years"] = len(fa)
print(f"secondary: the flag alone as a score, mean within-year AUC {np.mean(fa):.3f} over {len(fa)} years (0.5 = no information)", flush=True)
# (2) expanding window
for kind, reps in (("logit", 200), ("hgb", max(50, a.hgb_reps // 2))):
    r = S12.perm_delta_auc(df, B, Rc, "y", kind, TY, reps, a.seed, expanding=True)
    r.pop("pred_base"), r.pop("pred_br")
    sec[f"expanding_{kind}"] = {k: r[k] for k in ("auc_base", "auc_base_plus_r", "delta", "p_one_sided", "reps")}
    print(f"secondary expanding window {kind}: delta {r['delta']:+.4f}, p {r['p_one_sided']:.3f} ({reps} reps) ({mins()})", flush=True)
# (3) opacity block on the events with a score
scores = pd.read_pickle(G.DATA / "stage5_scores.pkl"); idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
d2 = S5.attach_T(df.copy(), scores, idx, S5.score_meta(scores), date_col="dep_ann_final")
d2 = d2[d2["T"].notna()].reset_index(drop=True)
for kind, reps in (("logit", 200), ("hgb", 60)):
    r = S12.perm_delta_auc(d2, B + Rc, ["T"], "y", kind, TY, reps, a.seed)
    r.pop("pred_base"), r.pop("pred_br")
    sec[f"opacity_{kind}"] = {"n": int(len(d2)), **{k: r[k] for k in ("auc_base", "auc_base_plus_r", "delta", "p_one_sided", "reps")}}
    print(f"secondary opacity T added to B+R ({len(d2)} events with a score), {kind}: delta {r['delta']:+.4f}, p {r['p_one_sided']:.3f} ({mins()})", flush=True)
# (4) the return itself: out-of-sample R-squared against the training mean
lo, hi = df.bhar.quantile([0.01, 0.99])
df["yr"] = df.bhar.clip(lo, hi)


def rolling_r2(cols, model):
    X, yv, yrs = df[cols].to_numpy(float), df.yr.to_numpy(), df.year.to_numpy()
    sse = sst = 0.0
    for t in TY:
        tr, te = np.isin(yrs, list(S12.split_years(t))), yrs == t
        if tr.sum() < 100 or te.sum() == 0:
            continue
        A, Bm = S12._prep(X[tr], X[te], scale=(model == "ridge"))
        m = (Ridge(alpha=10.0) if model == "ridge" else HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=0)).fit(A, yv[tr])
        sse += np.sum((yv[te] - m.predict(Bm)) ** 2); sst += np.sum((yv[te] - yv[tr].mean()) ** 2)
    return 1 - sse / sst


for model in ("ridge", "hgb"):
    rb, rbr = rolling_r2(B, model), rolling_r2(B + Rc, model)
    sec[f"return_r2_{model}"] = {"base": float(rb), "base_plus_r": float(rbr), "delta": float(rbr - rb)}
    print(f"secondary return as outcome ({model}): out-of-sample R2 base {rb:+.4f}, base+R {rbr:+.4f}, delta {rbr - rb:+.4f}", flush=True)
out["secondary"] = sec

# ------------------------------------------------------------------------------------------- P12b: rolling-window stability
fl, comp = df[df.flag == 1], df[(df.guided_unfulfilled == 1) & (df.flag == 0)]
g12 = pd.concat([fl.assign(g=1), comp.assign(g=0)]).copy()
gp = pd.read_pickle(G.DATA / "panel_guid.pkl")
rel = {int(p): np.sort(pd.DatetimeIndex(s.dropna().unique()).values) for p, s in gp.groupby("permno").t}


def first_release(permno, d):
    arr = rel.get(int(permno))
    if arr is None:
        return pd.NaT
    i = arr.searchsorted(np.datetime64(d + pd.Timedelta(days=2)), side="left")
    return pd.Timestamp(arr[i]) if i < len(arr) and arr[i] <= np.datetime64(d + pd.Timedelta(days=365)) else pd.NaT


fac = GT.load_factors(db)
g12["rel_date"] = [first_release(p, d) for p, d in zip(g12.permno, g12.dep_ann_final)]
cw = S11.car_window(g12.dropna(subset=["rel_date"]), crsp, fac, "rel_date", (-1, 1), col="car").dropna(subset=["car"])
cw["react"] = 100 * cw.car.clip(*cw.car.quantile([0.01, 0.99]))
g12["react"] = np.nan
g12.loc[cw.index, "react"] = cw.react
g12["bhar_w"] = 100 * g12.bhar.clip(lo, hi)
g12["loss"] = 100 * g12.y
wins = [(s, s + 4) for s in range(2004, 2021)]
blocks = [(2004, 2008), (2009, 2013), (2014, 2018), (2019, 2024)]


def gaps(sub, col):
    s = sub.dropna(subset=[col])
    return S12.gap_boot(s[col].values, s.g.values, s.q.values, 1000, a.seed)


p12b = {"windows": {}, "blocks": {}}
for col, label in (("loss", "large-loss share gap (pts)"), ("react", "first-guidance reaction gap CAR(-1,+1) (pts)"), ("bhar_w", "12-month BHAR gap (pts)")):
    rows = {}
    for lo_y, hi_y in wins:
        rows[f"{lo_y}-{hi_y}"] = gaps(g12[(g12.year >= lo_y) & (g12.year <= hi_y)], col)
    br = {f"{lo_y}-{hi_y}": gaps(g12[(g12.year >= lo_y) & (g12.year <= hi_y)], col) for lo_y, hi_y in blocks}
    q = S12.cochran_q([v["gap"] for v in br.values()], [v["se"] for v in br.values()])
    p12b["windows"][col] = rows
    p12b["blocks"][col] = {"by_block": br, "cochran_q": q}
    harm = (lambda v: v > 0) if col == "loss" else (lambda v: v < 0)               # the hypothesis: more losses, worse returns, worse reaction
    n_harm = sum(1 for v in rows.values() if np.isfinite(v["gap"]) and harm(v["gap"]))
    print(f"P12b {label}: {n_harm} of {len(rows)} rolling windows point in the hypothesised direction | blocks "
          + ", ".join(f"{k} {v['gap']:+.1f} (se {v['se']:.1f})" for k, v in br.items()) + f" | Cochran Q p {q['p']:.3f}", flush=True)
out["P12b"] = p12b
(RES / f"stage12_results{_tag}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(f"wrote {RES / ('stage12_results' + _tag + '.json')} ({mins()})")
