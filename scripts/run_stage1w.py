"""Run Stage 1W (docs/STAGE1W_SPEC.md): the Stage 1 calendar-time Carhart test on the widened flag.

python scripts/run_stage1w.py --part primary  [--reps 2000]     # writes outputs/results/stage1w_primary.json  (also the funnel and the Fiserv check)
python scripts/run_stage1w.py --part variants [--reps 1000]     # writes outputs/results/stage1w_variants.json (eight variants, Holm)
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
import gg_lossrate as LR
import gg_tests as GT
import gg_variants as GV

ap = argparse.ArgumentParser()
ap.add_argument("--part", choices=["primary", "variants"], required=True)
ap.add_argument("--reps", type=int, default=None)
ap.add_argument("--extend", action="store_true", help="Amendment 1: extend returns through 2025-12-31 with the CRSP v2 file")
a = ap.parse_args()
reps = a.reps or (2000 if a.part == "primary" else 1000)
tag = "_ext" if a.extend else ""
t0 = time.time()
mins = lambda: f"{(time.time() - t0) / 60:.1f} min"
RES = G.RESULTS

# ------------------------------------------------------------------ sample: the Stage 10 flag on eligible departures, with 12-month return windows
ev = LR.flags(GV.load_eligible_events(2004, 2025, "validation/candidate_labels.csv"))
db = G.WRDS(refresh=False)
crsp, mkt, _ = G.load_crsp(db, ev.permno.unique(), 2004, 2025)
fac_all = GT.load_factors(db)
cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
if a.extend:
    import gg_crsp_ciz  # noqa: F401  (the v2 cache is already in the project layout: permno, date, ret, prc, shrout)
    v2 = pd.read_pickle(ROOT / "gg_cache" / "ciz_dsf_2023_2025.pkl")
    v2["date"] = pd.to_datetime(v2["date"])
    new = v2[(v2.date > "2024-12-31") & v2.permno.isin(ev.permno.dropna().astype(int).unique())][["permno", "date", "ret"]]
    crsp = pd.concat([crsp[crsp.date <= "2024-12-31"][["permno", "date", "ret"]], new], ignore_index=True)
    cal = np.sort(pd.to_datetime(crsp.date).unique().astype("datetime64[ns]"))
    print(f"extended returns: {len(new):,} stock-days after 2024; calendar to {pd.Timestamp(cal[-1]):%Y-%m-%d}", flush=True)
cal = cal[np.isin(cal, fac_all.index.values)]
fac = fac_all.reindex(pd.DatetimeIndex(cal))
T_ = len(cal)
use = ev[ev.elig].copy()
win252 = GT.build_windows(use, crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
win126 = GT.build_windows(use, crsp, cal, date_col="dep_ann_final", skip=2, horizon=126)
use["has_win"] = use.event_id.isin(win252.keys())
fl = use[(use.flag == 1) & use.has_win]
nonfl = use[(use.flag == 0) & use.has_win]
print(f"eligible departures {len(use):,} | flagged {int((use.flag == 1).sum())} | flagged with window {len(fl)} | non-flagged with window {len(nonfl):,} ({mins()})", flush=True)


def run_test(name, ids, windows, factors, pool_base, reps, seed=7):
    ids = [i for i in ids if i in windows]
    port, cnt = GT.portfolio(windows, ids, T_)
    r = GT.alpha_test(port, cnt, fac, factors, 10)
    r["n_events"] = len(ids)
    pool = pd.concat([pool_base[["event_id", "year"]], use[use.event_id.isin(ids)][["event_id", "year"]]]).drop_duplicates("event_id")
    null = GT.placebo_alphas(windows, ids, pool, fac, T_, strata_col="year", reps=reps, seed=seed, factors=factors, maxlags=10)
    r["placebo_p"] = GT.placebo_p(r["alpha_ann"], null, "lower")
    r["placebo_mean"] = float(np.nanmean(null))
    r["placebo_sd"] = float(np.nanstd(null))
    r["placebo_q05"], r["placebo_q95"] = [float(x) for x in np.nanpercentile(null, [5, 95])]
    r["reps"] = reps
    r["name"] = name
    print(f"  {name:<44s} n={r['n_events']:>4d} alpha {r['alpha_ann']:+7.1%}  t={r['t']:+.2f}  placebo p={r['placebo_p']:.3f}  (placebo mean {r['placebo_mean']:+.1%}, sd {r['placebo_sd']:.1%}) ({mins()})", flush=True)
    return r


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


ids_all = fl.event_id.tolist()
out = {"spec": "docs/STAGE1W_SPEC.md", "reps": reps}

if a.part == "primary":
    # funnel (all departures -> eligible -> guided 365d -> sold after first -> period ends after exit -> usable window)
    raw = GV.load_eligible_events(2004, 2025, "validation/candidate_labels.csv")
    guided = use.n_guid_12m.fillna(0) >= 1
    sold = guided & (use.sell_val_post_guid.fillna(0) > 0)
    flag = use.flag == 1
    out["funnel"] = {"departures": int(len(raw)), "eligible": int(len(use)), "guided_365d": int(guided.sum()), "sold_after_first": int(sold.sum()),
                     "flag": int(flag.sum()), "flag_with_window": int(len(fl))}
    print("funnel:", out["funnel"], flush=True)
    # the Fiserv check
    fz = use[use.exec_name.str.contains("Bisignano", na=False) & (use.conm == "FISERV INC")]
    out["fiserv"] = {"in_eligible": int(len(fz)), "flagged": int((fz.flag == 1).sum()) if len(fz) else 0,
                     "has_window": bool(fz.event_id.isin(win252.keys()).any()) if len(fz) else False}
    print("Fiserv:", out["fiserv"], flush=True)
    out["primary"] = run_test("PRIMARY: widened flag, 12m, four-factor", ids_all, win252, GT.FACTORS_FF4, nonfl, reps)
    out["role_counts"] = {k: int(v) for k, v in fl.role.value_counts().items()}
    out["sector_counts"] = {k: int(v) for k, v in fl.sector.value_counts().head(5).items()}
    (RES / f"stage1w{tag}_primary.json").write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
else:
    crisis = {2008, 2009, 2020}
    ceo_base = nonfl[nonfl.role == "CEO"]
    sec = []
    sec.append(run_test("V1 CEOs only", fl[fl.role == "CEO"].event_id.tolist(), win252, GT.FACTORS_FF4, ceo_base, reps))
    sec.append(run_test("V2 six-month hold", ids_all, win126, GT.FACTORS_FF4, nonfl, reps))
    sec.append(run_test("V3 six-factor model", ids_all, win252, GT.FACTORS_FF6, nonfl, reps))
    sec.append(run_test("V4 excluding 2008-09 and 2020", fl[~fl.year.isin(crisis)].event_id.tolist(), win252, GT.FACTORS_FF4, nonfl[~nonfl.year.isin(crisis)], reps))
    sec.append(run_test("V5 announcements 2004-2014", fl[fl.year <= 2014].event_id.tolist(), win252, GT.FACTORS_FF4, nonfl[nonfl.year <= 2014], reps))
    sec.append(run_test("V6 announcements 2015-2025", fl[fl.year >= 2015].event_id.tolist(), win252, GT.FACTORS_FF4, nonfl[nonfl.year >= 2015], reps))
    sec.append(run_test("V7 guidance within 180 days", fl[fl.days_since_guid.fillna(9999) <= 180].event_id.tolist(), win252, GT.FACTORS_FF4, nonfl, reps))
    v8 = use[(use.n_guid_12m.fillna(0) >= 1) & (use.n_unfulfilled.fillna(0) >= 1) & (use.sell_val_post_lastguid.fillna(0) > 0) & use.has_win]
    sec.append(run_test("V8 sale after the latest guidance", v8.event_id.tolist(), win252, GT.FACTORS_FF4, nonfl, reps))
    adj = GT.holm({s["name"]: s["placebo_p"] for s in sec})
    for s in sec:
        s["holm_p"] = adj.get(s["name"], np.nan)
    out["variants"] = sec
    (RES / f"stage1w{tag}_variants.json").write_text(json.dumps(clean(out), indent=1), encoding="utf-8")
print("done", mins(), flush=True)
