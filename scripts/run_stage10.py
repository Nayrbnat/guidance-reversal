"""Run Stage 10 as amended (docs/STAGE10_SPEC.md, Amendment 1): T-A (large firms, new months) and T-B (pooled, all firms, 2025-01 to 2026-06), design C
(SIC2 x year-quarter effects, firm-clustered errors), Holm across the two. Order fixed by the spec: samples, power, THEN estimates. Run once.

python scripts/run_stage10.py [--power-reps 30]"""
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
import gg_stage7 as S7
import gg_stage8 as S8
import gg_stage10 as S10

ap = argparse.ArgumentParser()
ap.add_argument("--power-reps", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="write the results here instead of outputs/results/stage10_results.json (used to check refactors)")
a = ap.parse_args()
t0 = time.time()
CTRL = ["lmcap", "ret12", "lval", "V", "role_CEO", "role_CFO", "role_DIRECTOR"]
START, END, NEW_FROM, LARGE = "2025-01-01", "2026-06-30", "2025-10-01", 22.298     # log market cap of $4.83 billion, fixed in the spec

ins = pd.read_pickle(G.CACHE / "tr_insiders_combined_2026.pkl")
names = pd.read_pickle(G.CACHE / "ciz_names_merged.pkl")
names.loc[names["nameenddt"] == pd.Timestamp("2025-12-31"), "nameenddt"] = pd.Timestamp("2026-12-31")
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
for c in ("linkdt", "linkenddt"):
    link[c] = pd.to_datetime(link[c], errors="coerce")
guid = pd.read_pickle(G.DATA / "panel_guid.pkl")
meta = S5.score_meta(pd.read_pickle(G.DATA / "stage5_scores.pkl"))               # Z keeps the Stage 5 scale
scores = pd.read_pickle(G.DATA / "stage5_scores_2026.pkl")
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
crsp = pd.read_pickle(G.CACHE / "spliced_prices_2023_2026.pkl")
crsp["prc"] = crsp["prc"].abs()
f = pd.read_pickle(G.CACHE / "ciz_ff_daily.pkl")
f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
fac = f.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce")
if fac["mktrf"].abs().mean() > 0.02:
    fac = fac / 100.0
tab = S3.market_table(crsp)
print(f"inputs loaded ({(time.time() - t0) / 60:.1f} min); prices to {crsp.date.max().date()}, factors to {fac.index.max().date()}", flush=True)

d = S8.insider_sale_days(ins, names, link, min_value=100_000.0, start=START, end=END)
d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
n_linked = len(d)
print("insider-days >= $100k, officers and directors, linked, 2025-01 to 2026-06:", n_linked, "| by role:", d.role.value_counts().to_dict(), flush=True)
d = S6.attach_guidance(d, guid, 180)
d = S6.car_table(d, crsp, fac, horizons=(63,))
d["Yraw"] = -100.0 * d["car63"]
d["lval"] = np.log(d["val"])
d = S3.add_market_controls(d, tab, "trandate")
d = S6.pre_sale_vol(d, crsp, fac)
for r in ("CEO", "CFO", "DIRECTOR"):
    d[f"role_{r}"] = (d["role"] == r).astype(float)
sic = idx.assign(gvkey=idx["gvkey"].astype(str).str.zfill(6)).sort_values("filing_date").drop_duplicates("gvkey", keep="last").set_index("gvkey")["sic"]
d["sic2"] = S7.sic2(d["gvkey"].map(sic))
d = S5.attach_T(d, scores, idx, meta, date_col="trandate", strict=True)
d["Z"] = d["T_z"]
d["yq"] = d["trandate"].dt.to_period("Q").astype(str)
d["sicq"] = d["sic2"].astype(str) + "_" + d["yq"]
d["post2025"] = d["trandate"] > pd.Timestamp("2025-09-30")
d.to_pickle(G.DATA / "stage10_panel.pkl")


pooled, new_all, large = S10.stage_samples(d, new_from=NEW_FROM, large_cut=LARGE)
cover = {"linked": int(n_linked), "with_T": int(d["Z"].notna().sum()), "pooled": int(len(pooled)), "pooled_firms": int(pooled.gvkey.nunique()),
         "new_months_all": int(len(new_all)), "new_months_firms": int(new_all.gvkey.nunique()), "T_A_large_new": int(len(large)), "T_A_firms": int(large.gvkey.nunique()),
         "by_quarter_pooled": pooled.yq.value_counts().sort_index().to_dict(), "missing_T_reasons": d["T_missing_reason"].value_counts().to_dict(),
         "linked_without_return_window": int(d["Yraw"].isna().sum())}
print("SAMPLES:", json.dumps(cover), f"({(time.time() - t0) / 60:.1f} min)", flush=True)


def design_c(df, y="Y63", **kw):
    return S8.run_mechanism(df, y, controls=CTRL, firm="sicq", period=None, two_way=False, **kw)


# ---------------------------------------------------------------- power for BOTH tests, before either estimate
res = {"spec": "docs/STAGE10_SPEC.md Amendment 1", "coverage": cover}
est = {}
for name, df in (("T_A_large_new", large), ("T_B_pooled", pooled)):
    r = design_c(df)
    se = r["Z"]["se"]
    mde = 2.49 * se
    print(f"POWER {name}: n {r['n']}, SE of Z = {se:.3f}; minimum detectable effect = {mde:.2f} pp per sd of T", flush=True)
    pw = S8.power_sim(df[["gvkey", "sicq", "Y63", "Z"] + CTRL].reset_index(drop=True), "Y63", deltas=(0.0, mde / 2, mde, 2 * mde), reps=a.power_reps, seed=a.seed,
                      controls=CTRL, firm="sicq", period=None, two_way=False)
    print(f"  simulated rejection by planted effect: { {round(k, 3): round(v, 3) for k, v in pw.items()} }", flush=True)
    est[name] = r
    res[name] = {"power": {"se": se, "mde": mde, "sim": {str(round(k, 4)): v for k, v in pw.items()}, "reps": a.power_reps}}

# ---------------------------------------------------------------- the two estimates and Holm
holm = S10.holm_adjust({k: est[k]["p_one"] for k in est})
for k, r in est.items():
    z, se = r["Z"]["coef"], r["Z"]["se"]
    print(f"{k}: Z = {z:.3f} (se {se:.3f}), one-sided p {r['p_one']:.3f}, Holm-adjusted {holm[k]:.3f}, 90% interval [{z - 1.645 * se:.3f}, {z + 1.645 * se:.3f}], n {r['n']}", flush=True)
    res[k].update({"estimate": r, "holm_p": holm[k]})
rn = design_c(new_all)
print(f"reported beside T-B, new months only, all firms: Z = {rn['Z']['coef']:.3f} (se {rn['Z']['se']:.3f}), one-sided p {rn['p_one']:.3f}, n {rn['n']}", flush=True)
res["new_months_all_firms"] = rn


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


out = Path(a.out) if a.out else G.RESULTS / "stage10_results.json"
out.write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("wrote", out, f"({(time.time() - t0) / 60:.1f} min)", flush=True)
