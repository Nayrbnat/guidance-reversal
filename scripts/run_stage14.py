"""Run Stage 14 (docs/STAGE14_SPEC.md): valuation and turnaround controls for the flagged-firm returns. Exploratory; run once after the spec was fixed.
Order: samples, then A (announcement reaction), B (matched comparison), C (event-level regression), D (successor type), for the old and the new flag.

python scripts/run_stage14.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_stage14 as S14

ev = pd.read_pickle(G.DATA / "stage13_events_counts.pkl")
funda = pd.read_pickle(G.CACHE / "funda_events_2001_plus.pkl")
ec = pd.read_pickle(G.CACHE / "execucomp_2004_2025.pkl")
ev = ev.reset_index(drop=True)
ev["gvkey"] = ev["gvkey"].astype(str).str.zfill(6)
ev["dep_ann"] = pd.to_datetime(ev["dep_ann"])
ev["year"] = ev["dep_ann"].dt.year
ev["q"] = ev["dep_ann"].dt.to_period("Q").astype(str)
ev["log_mcap"] = np.log(ev["mcap"].where(ev["mcap"] > 0))
ev = S14.add_value_vars(ev, funda, lag_days=90)
ev["log_bm0"] = ev["log_bm"].fillna(0.0)                                     # missing log BM is zero with the indicator bm_missing in the regression
ev["flag_old"] = ((ev["gr_candidate_old"] == 1) & (ev["abrupt_ann"] == 1)).astype(int)
ev["flag_new"] = ((ev["gr_candidate9"] == 1) & (ev["abrupt_ann"] == 1)).astype(int)


def wins(s):
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)


for src, dst in (("bhar_252", "bhar_w"), ("car_m1p1", "car_w"), ("pre_ret", "pre_ret_w"), ("ep", "ep_w"), ("roa", "roa_w"), ("droa", "droa_w")):
    ev[dst] = wins(ev[src])
ev["role_ceo"] = (ev["role"] == "CEO").astype(float)
ev = S14.successor_outside(ev, ec)

SIZE_RET = ["log_mcap", "pre_ret_w"]
FULL = ["log_mcap", "pre_ret_w", "log_bm0", "bm_missing", "ep_w", "roa_w", "droa_w", "neg_earn", "role_ceo"]
MATCH_FULL = ["log_mcap", "pre_ret_w", "log_bm", "ep_w"]
sample = ev[ev["bhar_w"].notna()].copy()
cover = {"events": int(len(ev)), "with_bhar": int(len(sample)),
         "flag_old_with_bhar": int(sample.flag_old.sum()), "flag_new_with_bhar": int(sample.flag_new.sum()),
         "with_valuation_vars": int(sample[FULL].dropna().shape[0]),
         "flag_old_with_valuation": int(sample.loc[sample[FULL].notna().all(axis=1), "flag_old"].sum()),
         "flag_new_with_valuation": int(sample.loc[sample[FULL].notna().all(axis=1), "flag_new"].sum()),
         "funda_firm_years": int(len(funda)), "events_with_book_equity": int(ev["bm"].notna().sum()), "events_with_earnings": int(ev["ep"].notna().sum())}
print("SAMPLES:", json.dumps(cover), flush=True)
res = {"spec": "docs/STAGE14_SPEC.md", "coverage": cover}

for flag in ("flag_old", "flag_new"):
    r = {}
    ids = sample.loc[sample[flag] == 1, "event_id"].tolist()
    # A: announcement-window reaction, flagged against all other departures
    a = ev[ev["car_w"].notna()].copy()
    r["A_announcement_car"] = {**S14.flag_regression(a, "car_w", flag, [], fe=None, cluster="q"), "mean_flagged": float(a.loc[a[flag] == 1, "car_w"].mean()), "mean_other": float(a.loc[a[flag] == 0, "car_w"].mean()),
                               "n_flagged": int(a[flag].sum())}
    # B: matched comparison, with and without valuation
    r["B_matched_size_ret"] = S14.matched_diff(sample, "bhar_w", ids, SIZE_RET, k=5, reps=2000, seed=0)
    r["B_matched_full"] = S14.matched_diff(sample, "bhar_w", ids, MATCH_FULL, k=5, reps=2000, seed=0)
    # C: event-level regression; (i)/(ii) also on the same complete-case sample as (iii)
    same = sample.dropna(subset=FULL)
    r["C_i_full_sample"] = S14.flag_regression(sample, "bhar_w", flag, [], fe="year", cluster="q")
    r["C_i_same_sample"] = S14.flag_regression(same, "bhar_w", flag, [], fe="year", cluster="q")
    r["C_ii_size_ret"] = S14.flag_regression(same, "bhar_w", flag, SIZE_RET, fe="year", cluster="q")
    r["C_iii_full_controls"] = S14.flag_regression(same, "bhar_w", flag, FULL, fe="year", cluster="q")
    # D: successor type (descriptive only)
    ceo = ev[(ev["role"] == "CEO") & ev["outside"].notna()]
    fl = ceo[ceo[flag] == 1]
    r["D_successor"] = {"share_outside_flagged": float(fl["outside"].mean()) if len(fl) else None, "n_flagged_ceo_with_successor": int(len(fl)),
                        "share_outside_other": float(ceo.loc[ceo[flag] == 0, "outside"].mean()),
                        "bhar_flagged_outside": float(fl.loc[(fl["outside"] == 1) & fl["bhar_w"].notna(), "bhar_w"].mean()) if (fl["outside"] == 1).any() else None,
                        "bhar_flagged_inside": float(fl.loc[(fl["outside"] == 0) & fl["bhar_w"].notna(), "bhar_w"].mean()) if (fl["outside"] == 0).any() else None,
                        "n_outside_with_bhar": int(((fl["outside"] == 1) & fl["bhar_w"].notna()).sum()), "n_inside_with_bhar": int(((fl["outside"] == 0) & fl["bhar_w"].notna()).sum())}
    # descriptives of the flagged firms against the others (valuation and distress)
    r["descriptives"] = {c: {"flagged": float(sample.loc[sample[flag] == 1, c].mean()), "other": float(sample.loc[sample[flag] == 0, c].mean())} for c in ("log_mcap", "pre_ret_w", "log_bm", "ep_w", "roa_w", "neg_earn", "bm_missing")}
    res[flag] = r
    f = lambda d: f"coef {d['coef']:+.4f} (se {d['se']:.4f}, p2 {d['p_two']:.3f}, n {d['n']})"
    print(f"\n=== {flag}: {len(ids)} flagged events with a 12-month BHAR ===", flush=True)
    print("  A announcement CAR[-1,+1]:", f(r["A_announcement_car"]), f"| flagged mean {r['A_announcement_car']['mean_flagged']:+.4f}, other mean {r['A_announcement_car']['mean_other']:+.4f}, n flagged {r['A_announcement_car']['n_flagged']}", flush=True)
    for k in ("B_matched_size_ret", "B_matched_full"):
        m = r[k]
        print(f"  {k}: diff {m['diff']:+.4f}  95% [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]  n {m['n']} (dropped {m['n_dropped']})", flush=True)
    for k in ("C_i_full_sample", "C_i_same_sample", "C_ii_size_ret", "C_iii_full_controls"):
        print(f"  {k}:", f(r[k]), flush=True)
    print("  D successor:", json.dumps(r["D_successor"]), flush=True)
    print("  descriptives (flagged vs other):", {c: (round(v["flagged"], 3), round(v["other"], 3)) for c, v in r["descriptives"].items()}, flush=True)


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


(G.RESULTS / "stage14_results.json").write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("\nwrote", G.RESULTS / "stage14_results.json", flush=True)
