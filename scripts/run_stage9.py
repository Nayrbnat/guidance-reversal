"""Run Stage 9 (docs/STAGE9_SPEC.md): the Stage 8 mechanism test on held-out 2025 insider sales. Order fixed by the spec: build the sample and print it,
print power, THEN estimate. Run it once.

python scripts/run_stage9.py [--power-reps 30]"""
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
import gg_crsp_ciz as C

ap = argparse.ArgumentParser()
ap.add_argument("--power-reps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
CTRL = ["lmcap", "ret12", "lval", "V", "role_CEO", "role_CFO", "role_DIRECTOR"]
START, END = "2025-01-01", "2025-09-30"


def last(pattern):
    return sorted(glob.glob(str(G.CACHE / pattern)))[-1]


ins = pd.read_pickle(last("tr_insiders_2004_2025_*.pkl"))
names = pd.read_pickle(G.CACHE / "ciz_names_merged.pkl")
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
for c in ("linkdt", "linkenddt"):
    link[c] = pd.to_datetime(link[c], errors="coerce")
guid = pd.read_pickle(G.DATA / "panel_guid.pkl")
old_scores = pd.read_pickle(G.DATA / "stage5_scores.pkl")
meta = S5.score_meta(old_scores)                                   # Z keeps the Stage 5 scale (mu, sigma from the scores as they were)
scores = pd.read_pickle(G.DATA / "stage5_scores_2025.pkl")
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
crsp = pd.read_pickle(G.CACHE / "ciz_dsf_2023_2025.pkl")
crsp["prc"] = crsp["prc"].abs()
f = pd.read_pickle(G.CACHE / "ciz_ff_daily.pkl")
f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
fac = f.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce")
if fac["mktrf"].abs().mean() > 0.02:
    fac = fac / 100.0
tab = S3.market_table(crsp)
print(f"inputs loaded ({(time.time() - t0) / 60:.1f} min); crsp {crsp.date.min().date()} to {crsp.date.max().date()}, factors to {fac.index.max().date()}", flush=True)

d = S8.insider_sale_days(ins, names, link, min_value=100_000.0, start=START, end=END)
n_linked = len(d)
d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
print("insider-days >= $100k, officers and directors, linked, 2025-01 to 2025-09:", n_linked, "| by role:", d.role.value_counts().to_dict(), flush=True)
d = S6.attach_guidance(d, guid, 180)
d = S6.car_table(d, crsp, fac, horizons=(63,))
d["Y63"] = S3.wins((-100.0 * d["car63"]).dropna()).reindex(d.index)
d["lval"] = np.log(d["val"])
d = S3.add_market_controls(d, tab, "trandate")
d = S6.pre_sale_vol(d, crsp, fac)
d["V"] = (d["vol"] - d["vol"].mean()) / d["vol"].std()
for r in ("CEO", "CFO", "DIRECTOR"):
    d[f"role_{r}"] = (d["role"] == r).astype(float)
sic = idx.assign(gvkey=idx["gvkey"].astype(str).str.zfill(6)).sort_values("filing_date").drop_duplicates("gvkey", keep="last").set_index("gvkey")["sic"]
d["sic2"] = S7.sic2(d["gvkey"].map(sic))
d = S5.attach_T(d, scores, idx, meta, date_col="trandate", strict=True)
d["Z"] = d["T_z"]
d["yq"] = d["trandate"].dt.to_period("Q").astype(str)
base = d[d["Y63"].notna() & d["Z"].notna() & d["V"].notna() & d["sic2"].notna()].copy().reset_index(drop=True)
cover = {"linked": int(n_linked), "with_car": int(d["Y63"].notna().sum()), "with_T": int((d["Y63"].notna() & d["Z"].notna()).sum()), "primary_sample": int(len(base)),
         "firms": int(base.gvkey.nunique()), "people": int(base.personid.nunique()), "by_role": base.role.value_counts().to_dict(), "G_share": float(base.G.mean()),
         "quarters": sorted(base.yq.unique().tolist()), "missing_T_reasons": d.loc[d["Y63"].notna(), "T_missing_reason"].value_counts().to_dict()}
print("SAMPLE:", json.dumps(cover), f"({(time.time() - t0) / 60:.1f} min)", flush=True)

# ---------------------------------------------------------------- power, BEFORE the estimate
r0 = S8.run_mechanism(base, "Y63", controls=CTRL, two_way=False)
se = r0["Z"]["se"]
mde = 2.49 * se
print(f"POWER (before estimate): SE of Z = {se:.3f} pp per sd of T; minimum detectable effect = {mde:.2f} pp per sd", flush=True)
pw = S8.power_sim(base[["gvkey", "sic2", "yq", "Y63", "Z"] + CTRL].reset_index(drop=True), "Y63", deltas=(0.0, mde / 2, mde, 2 * mde), reps=a.power_reps, seed=a.seed,
                  controls=CTRL, two_way=False)
print("POWER simulation (share rejecting at 5%) by planted effect:", {round(k, 3): v for k, v in pw.items()}, flush=True)

# ---------------------------------------------------------------- the one confirmatory test
res = {"spec": "docs/STAGE9_SPEC.md", "coverage": cover, "power": {"se": se, "mde": mde, "sim": {str(round(k, 4)): v for k, v in pw.items()}, "reps": a.power_reps}, "primary": r0}
z, se_, p1 = r0["Z"]["coef"], r0["Z"]["se"], r0["p_one"]
print(f"PRIMARY: Z = {z:.3f} (se {se_:.3f}), one-sided p {p1:.3f}, two-sided p {r0['Z']['p_two']:.3f}, 90% interval [{z - 1.645 * se_:.3f}, {z + 1.645 * se_:.3f}], n {r0['n']}", flush=True)
print("  Stage 8 estimate -0.129 is", "INSIDE" if z - 1.645 * se_ <= -0.129 <= z + 1.645 * se_ else "OUTSIDE", "the 90% interval", flush=True)

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


attempt("S1_other_officers", lambda: S8.run_mechanism(base[base["role"] == "OFFICER"], "Y63", controls=CTRL, two_way=False))
med = base["lmcap"].median()
attempt("S1_large_firms", lambda: S8.run_mechanism(base[base["lmcap"] > med], "Y63", controls=CTRL, two_way=False))
attempt("S2_Z_x_G", lambda: S8.run_mechanism(base, "Y63", g="G", controls=CTRL, two_way=False))
d.to_pickle(G.DATA / "stage9_panel.pkl")

# ---------------------------------------------------------------- exploratory walk-forward on 2004-2024 (years already seen; carries no confirmatory weight)
panel = pd.read_pickle(G.DATA / "stage8_panel.pkl")
w = panel[panel["Y63"].notna() & panel["Z"].notna() & panel["V"].notna() & panel["sic2"].notna()].copy()
w["yr"] = w["trandate"].dt.year
SUBSETS = {"all": lambda x, m: x, "officers": lambda x, m: x[x["role"] == "OFFICER"], "large": lambda x, m: x[x["lmcap"] > m],
           "opportunistic": lambda x, m: x[x["cls"] == "opportunistic"]}
walk = []
for yr in range(2009, 2025):
    tr, te = w[(w.yr >= yr - 5) & (w.yr <= yr - 1)], w[w.yr == yr]
    m_tr, m_te = tr["lmcap"].median(), te["lmcap"].median()
    row = {"test_year": yr, "n_test": int(len(te))}
    best, best_t = None, 0.0
    for nm, fn in SUBSETS.items():
        try:
            rt = S8.run_mechanism(fn(tr, m_tr), "Y63", controls=CTRL, two_way=False)
            t = rt["Z"]["coef"] / rt["Z"]["se"]
            row[f"train_t_{nm}"] = float(t)
            if t < best_t:
                best, best_t = nm, t
        except Exception:
            row[f"train_t_{nm}"] = None
    try:
        row["test_all"] = S8.run_mechanism(te, "Y63", controls=CTRL, two_way=False)["Z"]["coef"]
    except Exception:
        row["test_all"] = None
    row["selected"] = best
    try:
        row["test_selected"] = S8.run_mechanism(SUBSETS[best](te, m_te), "Y63", controls=CTRL, two_way=False)["Z"]["coef"] if best else None
    except Exception:
        row["test_selected"] = None
    walk.append(row)
wf = pd.DataFrame(walk)
print("WALK-FORWARD (exploratory):")
print(wf[["test_year", "n_test", "test_all", "selected", "test_selected"]].round(3).to_string(index=False), flush=True)
print("  mean test_all", round(float(wf.test_all.mean()), 3), "| years negative", int((wf.test_all < 0).sum()), "of", int(wf.test_all.notna().sum()),
      "| mean test_selected", round(float(wf.test_selected.mean()), 3), flush=True)
res["walk_forward"] = walk


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


out = G.RESULTS / "stage9_results.json"
out.write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("wrote", out, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
