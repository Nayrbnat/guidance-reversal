"""Run Stage 6 (docs/STAGE6_SPEC.md): H5 at the level of the individual CEO insider sale.

python scripts/run_stage6.py [--power-reps 40]

Order fixed by the spec: build the panel, print the sample, run the power simulation, THEN estimate. Writes outputs/data/stage6_panel.pkl and
outputs/results/stage6_results.json."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_stage3 as S3
import gg_stage5 as S5
import gg_stage6 as S6
import gg_tests as GT

ap = argparse.ArgumentParser()
ap.add_argument("--power-reps", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
HORIZONS = (21, 63, 126, 252)

# ---------------------------------------------------------------- inputs
ep = pd.read_pickle(G.DATA / "panel_episodes.pkl")
sales = pd.read_pickle(G.DATA / "panel_sales.pkl")
g = pd.read_pickle(G.DATA / "panel_guid.pkl")
Q0 = pd.read_pickle(G.DATA / "panel_A_quarters.pkl")
permno_map = Q0.dropna(subset=["permno"]).drop_duplicates("ep_id").set_index("ep_id")["permno"].astype("int64").to_dict()
db = G.WRDS(refresh=False)
crsp, mkt, _ = G.load_crsp(db, sorted(g.permno.unique()), S3.START, S3.END)
fac = GT.load_factors(db)
tab, age_tab = S3.market_table(crsp), S3.load_age(db)
scores = pd.read_pickle(G.DATA / "stage5_scores.pkl")
meta = S5.score_meta(scores)
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
print(f"inputs loaded ({(time.time() - t0) / 60:.1f} min)", flush=True)

# ---------------------------------------------------------------- panel
d = S6.sale_days(sales, ep, permno_map, min_value=0.0)
print("sale-days (any value):", len(d), "| permno-matched episodes:", d.ep_id.nunique(), flush=True)
d = S6.attach_guidance(d, g, 180)
d = S6.car_table(d, crsp, fac, horizons=HORIZONS)
for h in HORIZONS:
    d[f"Y{h}"] = S3.wins((-100.0 * d[f"car{h}"]).dropna()).reindex(d.index)
d["lval"] = np.log(d["val"])
d = S3.add_market_controls(d, tab, "trandate")
d = S3.add_exec_controls(d, ep, age_tab, "trandate")
d = d.assign(gvkey=d["gvkey"].astype(str).str.zfill(6))
d = S5.attach_T(d, scores, idx, meta, date_col="trandate", strict=True)
d["Z"] = d["T_z"]
d["HighT"] = (d["T"] >= 3).astype(float).where(d["T"].notna())
d["ym"] = d["trandate"].dt.strftime("%Y-%m")
d["yr"] = d["trandate"].dt.year
d.to_pickle(G.DATA / "stage6_panel.pkl")
base = d[(d["val"] >= 100_000) & d["Z"].notna() & d["Y63"].notna()].copy()
cover = {"sale_days_any": int(len(d)), "sale_days_ge100k": int((d.val >= 100_000).sum()), "with_T": int(((d.val >= 100_000) & d.Z.notna()).sum()),
         "primary_sample": int(len(base)), "firms": int(base.gvkey.nunique()), "G": int(base.G.sum()), "X": int(base.X.sum()),
         "GX": int((base.G * base.X).sum()), "OX": int((base.O * base.X).sum()), "missing_T_reasons": d.loc[d.val >= 100_000, "T_missing_reason"].value_counts().to_dict()}
print("SAMPLE:", json.dumps(cover), flush=True)

# ---------------------------------------------------------------- power, BEFORE the estimates are printed
r0 = S6.run_primary(base, "Y63")
se = r0["ZGX"]["se"]
mde = 2.49 * se
print(f"POWER (before estimates): SE of Z x G x X = {se:.3f} pp per sd of T; minimum detectable effect (one-sided 5%, 80% power) = {mde:.2f} pp per sd", flush=True)
pbase = base[["gvkey", "ym", "Y63", "G", "X", "Z"] + S6.CONTROLS].reset_index(drop=True)
pw = S6.power_sim(pbase, "Y63", deltas=(0.0, mde / 2, mde, 2 * mde), reps=a.power_reps, seed=a.seed)
print("POWER simulation (share rejecting at 5%) by planted effect (pp per sd of T):", {round(k, 2): v for k, v in pw.items()}, flush=True)

# ---------------------------------------------------------------- estimates
res = {"spec": "docs/STAGE6_SPEC.md", "coverage": cover, "power": {"se": se, "mde": mde, "sim": {str(round(k, 3)): v for k, v in pw.items()}, "reps": a.power_reps}}
res["primary"] = r0
print("PRIMARY: Z x G x X =", round(r0["ZGX"]["coef"], 3), "se", round(se, 3), "one-sided p", round(r0["p_one"], 3), "| G x X (c) =", round(r0["GX"]["coef"], 3), flush=True)
sec = {}
sec["S1_all_sales_Z_only"] = None
res["secondary"] = sec


def attempt(name, fn):
    try:
        sec[name] = fn()
        print("  ", name, "done", flush=True)
    except Exception as e:
        sec[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("  ", name, "FAILED", sec[name], flush=True)


import gg_econ as E


def s1():
    x = d[d["Z"].notna() & d["Y63"].notna() & (d["val"] >= 100_000)]
    r = E.fe_ols(x, "Y63", ["Z"] + [c for c in S6.CONTROLS], firm="gvkey", year="ym", cluster="gvkey", cluster2="ym")
    return S3.summarise(r, "Z", -1)   # H5: coefficient on Z predicted negative


attempt("S1_all_sales_Z_only", s1)
attempt("S2_optimistic_O", lambda: S6.run_primary(base, "Y63", g="O"))
attempt("S3_high_T_dummy", lambda: S6.run_primary(base, "Y63", z="HighT"))
for h in (21, 126, 252):
    attempt(f"S4_h{h}", lambda h=h: S6.run_primary(d[(d.val >= 100_000) & d.Z.notna() & d[f"Y{h}"].notna()], f"Y{h}"))
attempt("S5_ge1m", lambda: S6.run_primary(base[base.val >= 1_000_000], "Y63"))
attempt("S5_all_values", lambda: S6.run_primary(d[d.Z.notna() & d.Y63.notna()], "Y63"))
first = base.assign(q=base.trandate.dt.to_period("Q")).sort_values("trandate").drop_duplicates(["ep_id", "q"])
attempt("S6_first_per_quarter", lambda: S6.run_primary(first, "Y63"))
attempt("S7_2004_2014", lambda: S6.run_primary(base[base.yr <= 2014], "Y63"))
attempt("S7_2015_2024", lambda: S6.run_primary(base[base.yr >= 2015], "Y63"))
out = G.RESULTS / "stage6_results.json"


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


out.write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("wrote", out, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
