"""Run Stage 7 (docs/STAGE7_SPEC.md): H5 at the sale level with industry-by-month fixed effects (P7a) and with abrupt exits and non-routine
sales as well (P7b). Reads the saved Stage 6 panel. Order fixed by the spec: samples and counts, power for both primary tests, THEN estimates.

python scripts/run_stage7.py [--power-reps 30]"""
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
import gg_stage6 as S6
import gg_stage7 as S7
import gg_tests as GT

ap = argparse.ArgumentParser()
ap.add_argument("--power-reps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()

d = pd.read_pickle(G.DATA / "stage6_panel.pkl")
ep = pd.read_pickle(G.DATA / "panel_episodes.pkl").set_index("ep_id")
sales = pd.read_pickle(G.DATA / "panel_sales.pkl")
ev = pd.read_csv(G.DATA / "events.csv", parse_dates=["dep_ann"], dtype={"gvkey": str, "execid": str})
ev["gvkey"] = ev["gvkey"].str.zfill(6)
lab = pd.read_csv(G.ROOT / "validation" / "candidate_labels.csv")
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")

base = d[(d["val"] >= 100_000) & d["Z"].notna() & d["Y63"].notna()].copy().reset_index(drop=True)
base["gvkey"] = base["gvkey"].astype(str).str.zfill(6)
base["execid"] = base["ep_id"].map(ep["execid"]).astype(str)
sic = idx.sort_values("filing_date").drop_duplicates("gvkey", keep="last").set_index("gvkey")["sic"]
base["sic2"] = S7.sic2(base["gvkey"].map(sic))
base = S7.exit_flags(base, ev, lab)
base = S7.routine_class(base, sales)

# volatility control
db = G.WRDS(refresh=False)
gd = pd.read_pickle(G.DATA / "panel_guid.pkl")
crsp, mkt, _ = G.load_crsp(db, sorted(gd.permno.unique()), S3.START, S3.END)
fac = GT.load_factors(db)
base = S6.pre_sale_vol(base, crsp, fac)
ok = base["vol"].notna() & base["sic2"].notna()
base["V"] = (base["vol"] - base.loc[ok, "vol"].mean()) / base.loc[ok, "vol"].std()
base = base[ok].reset_index(drop=True)
print(f"panel ready ({(time.time() - t0) / 60:.1f} min): {len(base)} sale-days", flush=True)
CTRL = S6.CONTROLS + ["V"]

base["Xa"] = (base["abrupt_class"] == 1).astype(float)                         # abrupt exit announced within 365 days
base["Xp"] = ((base["X_ann"] == 1) & (base["abrupt_class"] == 0)).astype(float)  # planned or other exit
unclass_exit = (base["X_ann"] == 1) & base["abrupt_class"].isna()
nonroutine = base["cls"] != "routine"

samples = {
    "P7a": base.assign(X=base["X"]),
    "P7b": base[nonroutine & ~unclass_exit].assign(X=lambda x: x["Xa"]),
    "S2_placebo": base[nonroutine & ~unclass_exit & (base["Xa"] == 0)].assign(X=lambda x: x["Xp"]),
    "S3_all_sales": base[~unclass_exit].assign(X=lambda x: x["Xa"]),
    "S3_opportunistic": base[(base["cls"] == "opportunistic") & ~unclass_exit].assign(X=lambda x: x["Xa"]),
}
cover = {k: {"sale_days": int(len(v)), "G": int(v.G.sum()), "X": int(v.X.sum()), "GX": int((v.G * v.X).sum()), "firms": int(v.gvkey.nunique()),
             "GX_firms": int(v.loc[(v.G * v.X) == 1, "gvkey"].nunique())} for k, v in samples.items()}
cover["classes"] = base["cls"].value_counts().to_dict()
cover["exits"] = {"X_ann": int(base.X_ann.sum()), "abrupt": int((base.abrupt_class == 1).sum()), "planned_or_other": int((base.abrupt_class == 0).sum()),
                  "unclassified": int(unclass_exit.sum())}
print("SAMPLES:", json.dumps(cover), flush=True)

# ---------------------------------------------------------------- power, BEFORE the estimates are printed
res = {"spec": "docs/STAGE7_SPEC.md", "coverage": cover, "power": {}}
for name in ("P7a", "P7b"):
    s = samples[name]
    r = S6.run_primary(s, "Y63", firm="sic2", controls=CTRL)
    se = r["ZGX"]["se"]
    mde = 2.49 * se
    pcols = ["gvkey", "sic2", "ym", "Y63", "G", "X", "Z"] + CTRL
    pw = S6.power_sim(s[pcols].reset_index(drop=True), "Y63", deltas=(0.0, mde / 2, mde, 2 * mde), reps=a.power_reps, seed=a.seed, firm="sic2", controls=CTRL)
    res["power"][name] = {"se": se, "mde": mde, "sim": {str(round(k, 3)): v for k, v in pw.items()}, "reps": a.power_reps}
    print(f"POWER {name} (before estimates): SE of Z x G x X = {se:.3f}; minimum detectable effect = {mde:.2f} pp per sd (Stage 6: SE 1.69, MDE 4.21); "
          f"simulated rejection at planted {[round(k, 2) for k in pw]}: {list(pw.values())}", flush=True)
    res[name] = r

# ---------------------------------------------------------------- estimates and Holm over the two primary tests
fam = {k: res[k]["p_one"] for k in ("P7a", "P7b")}
holm = GT.holm(fam)
res["family_p"], res["holm"] = fam, holm
for k in ("P7a", "P7b"):
    print(f"PRIMARY {k}: Z x G x X = {res[k]['ZGX']['coef']:.3f} (se {res[k]['ZGX']['se']:.3f}), one-sided p {res[k]['p_one']:.3f}, Holm {holm[k]:.3f} | G x X = {res[k]['GX']['coef']:.3f}", flush=True)

sec = {}
res["secondary"] = sec


def attempt(name, fn):
    try:
        sec[name] = fn()
        r = sec[name]
        print("  ", name, "ZGX", round(r["ZGX"]["coef"], 3), "se", round(r["ZGX"]["se"], 3), "p1", round(r["p_one"], 3), "n", r["n"], "GX", r.get("n_GX"), flush=True)
    except Exception as e:
        sec[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("  ", name, "FAILED", sec[name], flush=True)


for k in ("P7a", "P7b"):
    attempt(f"S1_{k}_firm_FE", lambda k=k: S6.run_primary(samples[k], "Y63", firm="gvkey", controls=CTRL))
attempt("S2_placebo_planned_exit", lambda: S6.run_primary(samples["S2_placebo"], "Y63", firm="sic2", controls=CTRL))
attempt("S3_P7b_all_sales", lambda: S6.run_primary(samples["S3_all_sales"], "Y63", firm="sic2", controls=CTRL))
attempt("S3_P7b_opportunistic", lambda: S6.run_primary(samples["S3_opportunistic"], "Y63", firm="sic2", controls=CTRL))
attempt("S4_P7b_optimistic_O", lambda: S6.run_primary(samples["P7b"], "Y63", g="O", firm="sic2", controls=CTRL))
for h in (21, 126):
    attempt(f"S5_P7b_h{h}", lambda h=h: S6.run_primary(samples["P7b"].dropna(subset=[f"Y{h}"]), f"Y{h}", firm="sic2", controls=CTRL))
first = samples["P7b"].assign(q=samples["P7b"].trandate.dt.to_period("Q")).sort_values("trandate").drop_duplicates(["ep_id", "q"])
attempt("S6_P7b_first_per_quarter", lambda: S6.run_primary(first, "Y63", firm="sic2", controls=CTRL))


def division(s2):
    n = int(s2)
    for lo, hi, name in ((1, 9, "A"), (10, 14, "B"), (15, 17, "C"), (20, 39, "D"), (40, 49, "E"), (50, 51, "F"), (52, 59, "G"), (60, 67, "H"), (70, 89, "I"), (91, 99, "J")):
        if lo <= n <= hi:
            return name
    return "Z"


p7b_div = samples["P7b"].assign(div=samples["P7b"]["sic2"].map(division))
attempt("S7_P7b_sic_division", lambda: S6.run_primary(p7b_div, "Y63", firm="div", controls=CTRL))


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


out = G.RESULTS / "stage7_results.json"
out.write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("wrote", out, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
