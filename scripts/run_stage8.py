"""Run Stage 8 (docs/STAGE8_SPEC.md): does the avoided loss after insider sales depend on transparency? All officers and directors, 2004-2024.
Order fixed by the spec: build the panel and print the sample, run power, THEN estimate.

python scripts/run_stage8.py [--power-reps 12]"""
import argparse
import glob
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
import gg_stage7 as S7
import gg_stage8 as S8
import gg_tests as GT

ap = argparse.ArgumentParser()
ap.add_argument("--power-reps", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
HORIZONS = (21, 63, 126)
CTRL = ["lmcap", "ret12", "lval", "V", "role_CEO", "role_CFO", "role_DIRECTOR"]


def last(pattern):
    return sorted(glob.glob(str(G.CACHE / pattern)))[-1]


ins = pd.read_pickle(last("tr_insiders_2004_2025_74394e0cd2.pkl"))
names = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(G.CACHE / "crsp_stocknames_v2_*.pkl")))], ignore_index=True)
names = names.drop_duplicates(["permno", "ncusip", "namedt"])
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
for c in ("linkdt", "linkenddt"):
    link[c] = pd.to_datetime(link[c], errors="coerce")
names["namedt"], names["nameenddt"] = pd.to_datetime(names["namedt"]), pd.to_datetime(names["nameenddt"])
guid = pd.read_pickle(G.DATA / "panel_guid.pkl")
scores = pd.read_pickle(G.DATA / "stage5_scores.pkl")
meta = S5.score_meta(scores)
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
db = G.WRDS(refresh=False)
crsp, mkt, _ = G.load_crsp(db, sorted(guid.permno.unique()), S3.START, S3.END)
fac = GT.load_factors(db)
tab = S3.market_table(crsp)
print(f"inputs loaded ({(time.time() - t0) / 60:.1f} min)", flush=True)

d = S8.insider_sale_days(ins, names, link, min_value=100_000.0)
n_linked = len(d)
print("insider-days >= $100k, officers and directors, linked to permno and gvkey:", n_linked, "| by role:", d.role.value_counts().to_dict(), flush=True)
d = S6.attach_guidance(d, guid, 180)
d = S6.car_table(d, crsp, fac, horizons=HORIZONS)
for h in HORIZONS:
    d[f"Y{h}"] = S3.wins((-100.0 * d[f"car{h}"]).dropna()).reindex(d.index)
d["lval"] = np.log(d["val"])
d = S3.add_market_controls(d, tab, "trandate")
d = S6.pre_sale_vol(d, crsp, fac)
d["V"] = (d["vol"] - d["vol"].mean()) / d["vol"].std()
for r in ("CEO", "CFO", "DIRECTOR"):
    d[f"role_{r}"] = (d["role"] == r).astype(float)
sic = idx.sort_values("filing_date").drop_duplicates("gvkey", keep="last").set_index("gvkey")["sic"]
d["sic2"] = S7.sic2(d["gvkey"].map(sic))
d = S5.attach_T(d, scores, idx, meta, date_col="trandate", strict=True)
d["Z"] = d["T_z"]
d["HighT"] = (d["T"] >= 3).astype(float).where(d["T"].notna())
d["yq"] = d["trandate"].dt.to_period("Q").astype(str)
d["yr"] = d["trandate"].dt.year
sales_all = ins[(ins["trancode"] == "S") & ins["rolecode1"].map(S8.role_group).notna()][["personid", "trandate"]].rename(columns={"personid": "ep_id"})
d = S7.routine_class(d.rename(columns={"personid": "ep_id"}), sales_all).rename(columns={"ep_id": "personid"})
d.to_pickle(G.DATA / "stage8_panel.pkl")
base = d[d["Y63"].notna() & d["Z"].notna() & d["V"].notna() & d["sic2"].notna()].copy().reset_index(drop=True)
cover = {"linked": int(n_linked), "with_car": int(d["Y63"].notna().sum()), "with_T": int((d["Y63"].notna() & d["Z"].notna()).sum()), "primary_sample": int(len(base)),
         "firms": int(base.gvkey.nunique()), "people": int(base.personid.nunique()), "by_role": base.role.value_counts().to_dict(), "classes": base["cls"].value_counts().to_dict(),
         "G_share": float(base.G.mean()), "missing_T_reasons": d.loc[d["Y63"].notna(), "T_missing_reason"].value_counts().to_dict()}
print("SAMPLE:", json.dumps(cover), f"({(time.time() - t0) / 60:.1f} min)", flush=True)

# ---------------------------------------------------------------- power, BEFORE the estimates are printed
r0 = S8.run_mechanism(base, "Y63", controls=CTRL)
se = r0["Z"]["se"]
mde = 2.49 * se
print(f"POWER (before estimates): SE of Z = {se:.3f} pp per sd of T; minimum detectable effect = {mde:.2f} pp per sd", flush=True)
pcols = ["gvkey", "sic2", "yq", "Y63", "Z"] + CTRL
pw = S8.power_sim(base[pcols].reset_index(drop=True), "Y63", deltas=(0.0, mde / 2, mde, 2 * mde), reps=a.power_reps, seed=a.seed, controls=CTRL)
print("POWER simulation (share rejecting at 5%) by planted effect:", {round(k, 3): v for k, v in pw.items()}, flush=True)

# ---------------------------------------------------------------- estimates
res = {"spec": "docs/STAGE8_SPEC.md", "coverage": cover, "power": {"se": se, "mde": mde, "sim": {str(round(k, 4)): v for k, v in pw.items()}, "reps": a.power_reps}, "primary": r0}
print(f"PRIMARY: Z = {r0['Z']['coef']:.3f} (se {se:.3f}), one-sided p {r0['p_one']:.3f}, two-sided p {r0['Z']['p_two']:.3f}, n {r0['n']}", flush=True)
sec = {}
res["secondary"] = sec


def attempt(name, fn):
    try:
        r = fn()
        sec[name] = r
        key = r.get("key", "Z")
        print("  ", name, key, round(r[key]["coef"], 3), "se", round(r[key]["se"], 3), "p1", round(r["p_one"], 3), "n", r["n"], flush=True)
    except Exception as e:
        sec[name] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print("  ", name, "FAILED", sec[name], flush=True)


attempt("S1_non_routine", lambda: S8.run_mechanism(base[base["cls"] != "routine"], "Y63", controls=CTRL))
attempt("S1_opportunistic", lambda: S8.run_mechanism(base[base["cls"] == "opportunistic"], "Y63", controls=CTRL))
for role in ("CEO", "CFO", "OFFICER", "DIRECTOR"):
    attempt(f"S2_{role}", lambda role=role: S8.run_mechanism(base[base["role"] == role], "Y63", controls=CTRL))
attempt("S3_Z_x_G", lambda: S8.run_mechanism(base, "Y63", g="G", controls=CTRL))
for h in (21, 126):
    attempt(f"S4_h{h}", lambda h=h: S8.run_mechanism(base[base[f"Y{h}"].notna()], f"Y{h}", controls=CTRL))
attempt("S5_high_T_dummy", lambda: S8.run_mechanism(base, "Y63", z="HighT", controls=CTRL))
attempt("S6_firm_FE", lambda: S8.run_mechanism(base, "Y63", controls=CTRL, firm="gvkey"))
attempt("S7_ge1m", lambda: S8.run_mechanism(base[base["val"] >= 1_000_000], "Y63", controls=CTRL))
first = base.assign(q=base["yq"]).sort_values("trandate").drop_duplicates(["personid", "q"])
attempt("S8_first_per_person_quarter", lambda: S8.run_mechanism(first, "Y63", controls=CTRL))
med = base["lmcap"].median()
attempt("S9_small_firms", lambda: S8.run_mechanism(base[base["lmcap"] <= med], "Y63", controls=CTRL))
attempt("S9_large_firms", lambda: S8.run_mechanism(base[base["lmcap"] > med], "Y63", controls=CTRL))


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


out = G.RESULTS / "stage8_results.json"
out.write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("wrote", out, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
