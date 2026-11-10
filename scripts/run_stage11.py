"""Run Stage 11 (docs/STAGE11_SPEC.md): P11a first-guidance reaction, P11b sparse-effect tests, P11c evidence for the null, P11d score correction.

python scripts/run_stage11.py [--reps 5000]          # run from the project root; writes outputs/results/stage11_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_lossrate as LR
import gg_stage5 as S5
import gg_stage11 as S11
import gg_tests as GT
import gg_variants as GV

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
RES = G.RESULTS
GH = RES / "github_main"

# ------------------------------------------------------------------------------------------- sample (as Stage 10)
ev = LR.flags(GV.load_eligible_events(2004, 2025, "validation/candidate_labels.csv"))
lab = pd.read_csv("validation/candidate_labels.csv").drop_duplicates("event_id").set_index("event_id")
ev["strict"] = ((ev.gr_candidate == 1) & (ev.event_id.map(lab.abrupt_ann_label) == 1)).astype(int)
db = G.WRDS(refresh=False)
crsp, mkt, _ = G.load_crsp(db, ev.permno.unique(), 2004, 2025)
cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
mk = mkt.drop_duplicates("date").set_index("date").mkt.reindex(pd.DatetimeIndex(cal)).astype(float).values
elig = ev[ev.elig].copy()
win = GT.build_windows(elig, crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
use = elig.join(LR.bhars(win, mk), on="event_id").dropna(subset=["bhar"]).copy()
use["year"] = use.dep_ann_final.dt.year
flagged = use[use.flag == 1]
comp = use[(use.guided_unfulfilled == 1) & (use.flag == 0)]
pool = use[use.flag == 0]
print(f"sample: eligible with BHAR {len(use):,} | flagged {len(flagged)} | comparator (guided, no sale) {len(comp)} | pool {len(pool):,} | strict {int(use.strict.sum())} ({(time.time()-t0)/60:.1f} min)", flush=True)

out = {"spec": "docs/STAGE11_SPEC.md", "reps": a.reps, "sample": {"eligible_with_bhar": int(len(use)), "flagged": int(len(flagged)),
       "comparator": int(len(comp)), "pool": int(len(pool)), "strict": int(use.strict.sum())}}

# ------------------------------------------------------------------------------------------- P11b: sparse tests on the 12-month return
sp = S11.sparse_calibrate(flagged.bhar.values, flagged.year.values, pool.bhar.values, pool.year.values, a.reps, a.seed)
sp_strict = S11.sparse_calibrate(use[use.strict == 1].bhar.values, use[use.strict == 1].year.values, pool.bhar.values, pool.year.values, a.reps, a.seed)
out["P11b"] = {"primary_bhar_flagged": sp, "strict_flag": sp_strict}
print(f"P11b (BHAR, {sp['n']} flagged): HC p={sp['p_hc']:.3f} | Cauchy p={sp['p_cauchy']:.3f} | mean-based p={sp['p_mean']:.3f} | "
      f"events with p<0.05: {sp['n_events_p_below_005']} (expected {sp['expected_below_005']:.0f}) ({(time.time()-t0)/60:.1f} min)", flush=True)

# ------------------------------------------------------------------------------------------- P11a: reaction to the first guidance after the exit
g = pd.read_pickle(G.DATA / "panel_guid.pkl")
rel = {int(p): np.sort(pd.DatetimeIndex(s.dropna().unique()).values) for p, s in g.groupby("permno").t}


def first_release(permno, d):
    arr = rel.get(int(permno))
    if arr is None:
        return pd.NaT
    lo, hi = np.datetime64(d + pd.Timedelta(days=2)), np.datetime64(d + pd.Timedelta(days=365))
    i = arr.searchsorted(lo, side="left")
    return pd.Timestamp(arr[i]) if i < len(arr) and arr[i] <= hi else pd.NaT


fac = GT.load_factors(db)
grp = pd.concat([flagged.assign(g=1), comp.assign(g=0)]).copy()
grp["rel_date"] = [first_release(p, d) for p, d in zip(grp.permno, grp.dep_ann_final)]
print(f"first post-exit guidance found for {int(grp.rel_date.notna().sum())} of {len(grp)} (flagged {int(grp[grp.g==1].rel_date.notna().sum())} of {len(flagged)})", flush=True)
res_a = {}
for name, w in (("primary_m1p1", (-1, 1)), ("w_0p1", (0, 1)), ("w_m1p5", (-1, 5))):
    x = S11.car_window(grp.dropna(subset=["rel_date"]), crsp, fac, "rel_date", w, col="car").dropna(subset=["car"]).copy()
    lo, hi = x.car.quantile([0.01, 0.99])
    x["y"] = 100 * x.car.clip(lo, hi)
    x["q"] = x.dep_ann_final.dt.to_period("Q").astype(str)
    r = S11.cluster_boot_diff(x.y.values, x.g.values, x.q.values, a.reps, a.seed)
    r["mean_flagged"] = float(x[x.g == 1].y.mean())
    r["mean_comparator"] = float(x[x.g == 0].y.mean())
    res_a[name] = r
    if name == "primary_m1p1":
        xp = x
print(f"P11a primary CAR(-1,+1): flagged {res_a['primary_m1p1']['mean_flagged']:+.2f} vs comparator {res_a['primary_m1p1']['mean_comparator']:+.2f} pts | "
      f"diff {res_a['primary_m1p1']['diff']:+.2f} (se {res_a['primary_m1p1']['se']:.2f}) one-sided p {res_a['primary_m1p1']['p_one_sided_neg']:.3f} | "
      f"detectable {res_a['primary_m1p1']['mde']:.2f} pts | n {res_a['primary_m1p1']['n1']}/{res_a['primary_m1p1']['n0']} ({(time.time()-t0)/60:.1f} min)", flush=True)
# secondary: against placebo groups of all non-flagged eligible departures with a post-exit release, and sparse tests on the CARs
allx = pool.copy()
allx["rel_date"] = [first_release(p, d) for p, d in zip(allx.permno, allx.dep_ann_final)]
allx = S11.car_window(allx.dropna(subset=["rel_date"]), crsp, fac, "rel_date", (-1, 1), col="car").dropna(subset=["car"])
allx["y"] = 100 * allx.car.clip(*allx.car.quantile([0.01, 0.99]))
fl_c = xp[xp.g == 1]
sp_car = S11.sparse_calibrate(fl_c.y.values, fl_c.dep_ann_final.dt.year.values, allx.y.values, allx.dep_ann_final.dt.year.values, a.reps, a.seed)
out["P11a"] = {"windows": res_a, "vs_all_nonflagged": {"n_flagged": int(len(fl_c)), "pool_n": int(len(allx))}, "sparse_on_car": sp_car}
print(f"P11a vs all departures (mean-based p {sp_car['p_mean']:.3f}); sparse tests on CARs: HC p={sp_car['p_hc']:.3f} Cauchy p={sp_car['p_cauchy']:.3f}", flush=True)

# Holm over the two Stage 11 primaries (project-wide count becomes 40)
pa, pb = res_a["primary_m1p1"]["p_one_sided_neg"], sp["p_hc"]
srt = sorted([("P11a", pa), ("P11b_HC", pb)], key=lambda t: t[1])
adj, prev = {}, 0.0
for k, (n, p) in enumerate(srt):
    prev = max(prev, min(1.0, (len(srt) - k) * p))
    adj[n] = prev
out["holm"] = adj
out["project_wide_tests"] = 40

# ------------------------------------------------------------------------------------------- P11c: how large an effect is ruled out
s8 = json.load(open(RES / "stage8_results.json"))["primary"]
s9 = json.load(open(RES / "lossrate_results.json"))["within_departures"]
s1 = json.load(open(RES / "study_results.json"))["primary"]
s5 = json.load(open(GH / "stage5_results.json"))["tests"]
s6 = json.load(open(GH / "stage6_results.json"))["primary"]
s7 = json.load(open(GH / "stage7_results.json"))
rows = []


def add(name, est, se, harm_sign, sesoi=None, unit=""):
    r = {"test": name, "estimate": est, "se": se, "unit": unit, "harm_sign": harm_sign,
         "bound95_harm_side": harm_sign * est + 1.645 * se}
    if sesoi is not None:
        r["sesoi"] = sesoi
        r["noninferiority"] = S11.noninferiority(est, se, sesoi, harm_sign)
        r["bf01"] = {f"{m}x": S11.bf01_halfnormal(est, se, sesoi * m, harm_sign) for m in (0.5, 1, 2)}
    rows.append(r)


add("Stage 8 opacity mechanism (pts per SD of T)", s8["Z"]["coef"], s8["Z"]["se"], -1, 0.5, "pts per SD")
add("Stage 10 large-loss share, flagged vs no-sale (pts)", 100 * s9["diff"], 100 * (s9["ci95"][1] - s9["ci95"][0]) / (2 * 1.96), +1, 5.0, "pts")
add("Stage 1 alpha (% a year)", 100 * s1["alpha_ann"], 100 * abs(s1["alpha_ann"] / s1["t"]), -1, 5.0, "% a year")
for nm, k, key in (("Stage 5 T2", "T2", "TOS"), ("Stage 5 T3", "T3", "TOSX")):
    add(nm, s5[k]["result"][key]["coef"], s5[k]["result"][key]["se"], -1, None)
add("Stage 5 T4", s5["T4"]["result"]["primary"]["coef"], s5["T4"]["result"]["primary"]["se"], -1, None)
add("Stage 6 primary (Z x G x X)", s6["ZGX"]["coef"], s6["ZGX"]["se"], -1, None)
add("Stage 7 P7a (Z x G x X)", s7["P7a"]["ZGX"]["coef"], s7["P7a"]["ZGX"]["se"], -1, None)
add("Stage 7 P7b (Z x G x X)", s7["P7b"]["ZGX"]["coef"], s7["P7b"]["ZGX"]["se"], -1, None)
out["P11c"] = rows
for r in rows:
    if "bf01" in r:
        print(f"P11c {r['test']}: est {r['estimate']:+.3f} se {r['se']:.3f} | rules out harm >= {r['bound95_harm_side']:+.2f} {r['unit']} | BF01 {r['bf01']['1x']:.1f} (0.5x {r['bf01']['0.5x']:.1f}, 2x {r['bf01']['2x']:.1f})", flush=True)

# ------------------------------------------------------------------------------------------- P11d: correct the opacity results for the noisy score
aud = pd.read_excel(ROOT / "validation" / "AUDIT_100_Representative_complete.xlsx", sheet_name="Representative", header=3)
aud = aud.rename(columns={"YOUR SCORE 0-5": "human", "id (hidden)": "unit_id"})[["unit_id", "human"]].dropna()
key = pd.read_csv(ROOT / "validation" / "transp_pilot" / "audit_KEY_do_not_open.csv")[["unit_id", "model_T"]]
pairs = aud.merge(key, on="unit_id").dropna()
sc = pd.read_pickle(G.DATA / "stage5_scores.pkl")
sd_m = S5.score_meta(sc)["sd"]
print(f"audit pairs {len(pairs)} | corr {np.corrcoef(pairs.human, pairs.model_T)[0,1]:.3f} | mean human {pairs.human.mean():.2f} vs model {pairs.model_T.mean():.2f}", flush=True)
corr = {}
for nm, est, se in (("Stage 8 primary (Z)", s8["Z"]["coef"], s8["Z"]["se"]), ("Stage 6 primary (Z x G x X)", s6["ZGX"]["coef"], s6["ZGX"]["se"]),
                    ("Stage 7 P7a (Z x G x X)", s7["P7a"]["ZGX"]["coef"], s7["P7a"]["ZGX"]["se"])):
    corr[nm] = S11.regression_calibration(pairs.human.values, pairs.model_T.values, sd_m, est, se, a.reps, a.seed)
    c = corr[nm]
    print(f"P11d {nm}: model {est:+.3f} (se {se:.3f})", flush=True)
    print(f"      as registered:          x{c['scale_factor']:.2f} -> {c['theta_corrected']:+.3f}, 95% [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p(effect<0) {c['p_lt0']:.3f}", flush=True)
    print(f"      population-consistent:  x{c['scale_factor_pop']:.2f} -> {c['theta_corrected_pop']:+.3f}, 95% [{c['ci95_pop'][0]:+.3f}, {c['ci95_pop'][1]:+.3f}], p(effect<0) {c['p_lt0_pop']:.3f}  (R2 {c['r2']:.2f}; audit sd of model score {c['sd_m_audit']:.2f} vs population {c['sd_global_m']:.2f})", flush=True)
out["P11d"] = corr
(RES / "stage11_results.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(f"Holm-adjusted: {adj}\nwrote {RES / 'stage11_results.json'} ({(time.time()-t0)/60:.1f} min)")
