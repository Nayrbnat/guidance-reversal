"""
gg_variants.py -- Stage-2 tests (PREREGISTRATION.md, "Stage 2 (REVISED ...)").

Hypothesis H2: executives who issued guidance, sold shares and then left (any exit type except death) earn negative
risk-adjusted returns, more so when the guidance was optimistic. No abruptness condition.

    python src/gg_variants.py --start 2004 --end 2025 --reps 5000

Family (Holm-corrected together):
  F1: 4 links (L1-L4) x 2 universes (U1, U2)  -> calendar-time FF4 alpha vs placebo      (8 tests)
  F2: 3 links (L1-L3) x 2 universes           -> alpha(optimistic guidance) - alpha(not), within-year permutation (6 tests)
Writes outputs/results/stage2_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg_tests as GT
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS
LINKS = {
    "L1": "guidance <=180d, sale after latest guidance",
    "L2": "guidance <=365d, sale after latest guidance",
    "L3": "guidance <=365d, any sale in prior 365d",
    "L4": "any sale in prior 365d (no guidance requirement)",
}


def link_masks(ev: pd.DataFrame) -> dict:
    d = ev.days_since_guid
    post = ev.sell_val_post_lastguid.fillna(0) > 0
    anysale = ev.sell_val_12m.fillna(0) > 0
    return {"L1": (d <= 180) & post, "L2": (d <= 365) & post, "L3": (d <= 365) & anysale, "L4": anysale}


def load_eligible_events(start: int, end: int, labels: str = "validation/candidate_labels.csv") -> pd.DataFrame:
    """Same event construction as main() (used by Stage 3): columns dep_ann_final, has_ann, is_death, is_health, year, elig."""
    ev = pd.read_csv(DATA / "events.csv", parse_dates=["dep_ann", "leftofc", "first_guid_date"], dtype={"gvkey": str})
    ev["dep_ann_final"] = ev.dep_ann
    ev["has_ann"] = ev.ann_found == 1
    dep_type_ann = pd.Series(index=ev.index, dtype=object)
    lab_path = Path(labels)
    if lab_path.exists():
        lab = pd.read_csv(lab_path, parse_dates=["ann_date"]).drop_duplicates("event_id").set_index("event_id")
        j = ev[["event_id"]].join(lab[["ann_date", "ann_found_ann", "dep_type_ann"]], on="event_id")
        use = j.ann_date.notna() & (j.ann_found_ann == 1)
        ev.loc[use, "dep_ann_final"] = j.loc[use, "ann_date"]
        ev.loc[use, "has_ann"] = True
        dep_type_ann = j.dep_type_ann
    ev["is_death"] = (ev.dep_type_rule == "death") | (dep_type_ann == "death")
    ev["is_health"] = (ev.dep_type_rule == "health_personal") | (dep_type_ann == "health_personal")
    ev["year"] = ev.dep_ann_final.dt.year
    ev = ev[(ev.year >= start) & (ev.year <= end)].reset_index(drop=True)
    ev["elig"] = ev.has_ann & ~ev.is_death
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2004)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--reps-explore", type=int, default=2000)
    ap.add_argument("--labels", default="validation/candidate_labels.csv")
    a = ap.parse_args()

    ev = pd.read_csv(DATA / "events.csv", parse_dates=["dep_ann", "leftofc", "first_guid_date"], dtype={"gvkey": str})
    ev["dep_ann_final"] = ev.dep_ann
    ev["has_ann"] = ev.ann_found == 1
    dep_type_ann = pd.Series(index=ev.index, dtype=object)
    lab_path = Path(a.labels)
    if lab_path.exists():
        lab = pd.read_csv(lab_path, parse_dates=["ann_date"]).drop_duplicates("event_id").set_index("event_id")
        j = ev[["event_id"]].join(lab[["ann_date", "ann_found_ann", "dep_type_ann"]], on="event_id")
        use = j.ann_date.notna() & (j.ann_found_ann == 1)
        ev.loc[use, "dep_ann_final"] = j.loc[use, "ann_date"]
        ev.loc[use, "has_ann"] = True
        dep_type_ann = j.dep_type_ann
    ev["is_death"] = (ev.dep_type_rule == "death") | (dep_type_ann == "death")
    ev["is_health"] = (ev.dep_type_rule == "health_personal") | (dep_type_ann == "health_personal")
    ev["year"] = ev.dep_ann_final.dt.year
    ev = ev[(ev.year >= a.start) & (ev.year <= a.end)].reset_index(drop=True)
    elig = ev.has_ann & ~ev.is_death
    print(f"events {len(ev):,} | with identified announcement {int(ev.has_ann.sum()):,} | deaths excluded {int((ev.has_ann & ev.is_death).sum())} | eligible {int(elig.sum()):,}")

    db = G.WRDS(refresh=False)
    crsp, mkt, _ = G.load_crsp(db, ev.permno.unique(), a.start, a.end)
    fac_all = GT.load_factors(db)
    cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
    cal = cal[np.isin(cal, fac_all.index.values)]
    fac = fac_all.reindex(pd.DatetimeIndex(cal))
    T_ = len(cal)
    win = GT.build_windows(ev[elig], crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
    ev["has_win"] = ev.event_id.isin(win.keys())
    elig = elig & ev.has_win
    print(f"eligible events with a usable 12m window: {int(elig.sum()):,}")

    L = link_masks(ev)
    U = {"U1": pd.Series(True, index=ev.index), "U2": ev.role_last.isin(["CEO", "CFO"])}
    res = {"n_events": int(len(ev)), "n_eligible": int(elig.sum()), "links": LINKS, "F1": [], "F2": [], "exploratory": {}}

    def alpha_block(ids):
        port, cnt = GT.portfolio(win, ids, T_)
        r = GT.alpha_test(port, cnt, fac, GT.FACTORS_FF4, 10)
        return {k: r[k] for k in ("alpha_ann", "t", "n_days", "mean_pos")}

    # ------------------------------------------------------------------ F1
    print("\nF1: does guidance -> sale -> exit predict negative alpha?")
    for lk in LINKS:
        for un in U:
            m = elig & L[lk] & U[un]
            ids = ev.loc[m, "event_id"].tolist()
            pool = ev.loc[elig & U[un], ["event_id", "year"]]
            r = {"name": f"F1 {lk} x {un}", "link": lk, "universe": un, "n_events": len(ids)}
            if len(ids) >= 5:
                r.update(alpha_block(ids))
                null = GT.placebo_alphas(win, ids, pool, fac, T_, reps=a.reps, seed=11)
                r["placebo_p"] = GT.placebo_p(r["alpha_ann"], null, "lower")
                r["placebo_mean"], r["placebo_sd"] = float(np.nanmean(null)), float(np.nanstd(null))
                r["placebo_q05"] = float(np.nanpercentile(null, 5))
            r["in_family"] = bool(len(ids) >= 40 and r.get("placebo_p") is not None and np.isfinite(r.get("placebo_p", np.nan)))
            res["F1"].append(r)
            print(f"  {r['name']:<12s} n={r['n_events']:>5d}  alpha {r.get('alpha_ann', float('nan')):+7.1%}  t={r.get('t', float('nan')):+.2f}  placebo p={r.get('placebo_p', float('nan')):.3f}  "
                  f"(placebo sd {r.get('placebo_sd', float('nan')):.1%})")

    # ------------------------------------------------------------------ F2
    print("\nF2: among them, do OPTIMISTIC guiders (guidance above consensus) do worse?")
    for lk in ("L1", "L2", "L3"):
        for un in U:
            m = elig & L[lk] & U[un] & ev.opt_last.notna()
            sub = ev[m]
            flags = (sub.opt_last > 0).values
            r = {"name": f"F2 {lk} x {un}", "link": lk, "universe": un, "n_events": int(m.sum()), "n_optimistic": int(flags.sum()), "n_other": int((~flags).sum())}
            if flags.sum() >= 5 and (~flags).sum() >= 5:
                obs, aO, aN, null = GT.split_permutation(win, sub.event_id.values, flags, sub.year.values, fac, T_, reps=a.reps, seed=21)
                r.update({"alpha_optimistic": aO, "alpha_other": aN, "diff": obs, "perm_p": GT.placebo_p(obs, null, "lower"),
                          "perm_sd": float(np.nanstd(null))})
            r["in_family"] = bool(min(r["n_optimistic"], r["n_other"]) >= 20 and np.isfinite(r.get("perm_p", np.nan)))
            res["F2"].append(r)
            print(f"  {r['name']:<12s} n={r['n_events']:>4d} (opt {r['n_optimistic']}, other {r['n_other']})  alpha O {r.get('alpha_optimistic', float('nan')):+7.1%}  N {r.get('alpha_other', float('nan')):+7.1%}  "
                  f"diff {r.get('diff', float('nan')):+7.1%}  perm p={r.get('perm_p', float('nan')):.3f}")

    # ------------------------------------------------------------------ Holm over the family
    fam = {}
    for r in res["F1"]:
        if r["in_family"]:
            fam[r["name"]] = r["placebo_p"]
    for r in res["F2"]:
        if r["in_family"]:
            fam[r["name"]] = r["perm_p"]
    adj = GT.holm(fam)
    for r in res["F1"] + res["F2"]:
        r["holm_p"] = adj.get(r["name"])
    res["family_size"] = len(fam)
    res["smallest_holm_p"] = min(adj.values()) if adj else None
    print(f"\nHolm family size {len(fam)}; smallest Holm-adjusted p = {res['smallest_holm_p']}")

    # ------------------------------------------------------------------ exploratory (NOT in the family)
    print("\nEXPLORATORY (not part of the family)")
    ex = res["exploratory"]
    # E1: raised guidance as the intent split
    ex["raise_split"] = []
    for lk in ("L1", "L2", "L3"):
        for un in U:
            m = elig & L[lk] & U[un] & ev.guid_raise_last.notna()
            sub = ev[m]; flags = (sub.guid_raise_last == 1).values
            r = {"name": f"raise {lk} x {un}", "n": int(m.sum()), "n_raised": int(flags.sum())}
            if flags.sum() >= 10 and (~flags).sum() >= 10:
                obs, aO, aN, null = GT.split_permutation(win, sub.event_id.values, flags, sub.year.values, fac, T_, reps=a.reps_explore, seed=31)
                r.update({"alpha_raised": aO, "alpha_not": aN, "diff": obs, "perm_p": GT.placebo_p(obs, null, "lower")})
            ex["raise_split"].append(r)
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    # E2: ex-post guidance misses (periods ending after the exit) for the guide-sell-leave group vs other departures
    ex["post_miss"] = []
    for lk in LINKS:
        g1 = ev[elig & L[lk] & ev.miss_unfulfilled.notna()].miss_unfulfilled
        g0 = ev[elig & ~L[lk] & ev.miss_unfulfilled.notna()].miss_unfulfilled
        if len(g1) >= 10:
            p = st.mannwhitneyu(g1, g0, alternative="less").pvalue
            ex["post_miss"].append({"link": lk, "n_group": int(len(g1)), "median_miss_group": float(g1.median()), "median_miss_other": float(g0.median()), "p_group_more_negative": float(p)})
    print("   post-exit guidance miss:", ex["post_miss"])
    # E3: continuous optimism vs 12m BHAR within L3 x U1
    sub = ev[elig & L["L3"] & ev.opt_last.notna() & ev.bhar_252.notna()]
    if len(sub) > 20:
        rho, p = st.spearmanr(sub.opt_last, sub.bhar_252)
        ex["optimism_vs_bhar"] = {"link": "L3", "n": int(len(sub)), "spearman": float(rho), "p": float(p)}
        print("   spearman(optimism, 12m BHAR), L3:", ex["optimism_vs_bhar"])
    # E4: excluding health/personal exits as well
    ex["no_health"] = []
    for lk in LINKS:
        m = elig & L[lk] & U["U2"] & ~ev.is_health
        ids = ev.loc[m, "event_id"].tolist()
        pool = ev.loc[elig & U["U2"] & ~ev.is_health, ["event_id", "year"]]
        if len(ids) >= 20:
            r = alpha_block(ids)
            null = GT.placebo_alphas(win, ids, pool, fac, T_, reps=a.reps_explore, seed=41)
            r.update({"link": lk, "n": len(ids), "placebo_p": GT.placebo_p(r["alpha_ann"], null, "lower")})
            ex["no_health"].append(r)
    print("   excluding health exits (U2):", [(r["link"], r["n"], round(r["alpha_ann"], 3), round(r["placebo_p"], 3)) for r in ex["no_health"]])
    # descriptive: composition of the main widened group
    g = ev[elig & L["L3"] & U["U2"]]
    ex["describe_L3_U2"] = {"n": int(len(g)), "years": {int(k): int(v) for k, v in g.year.value_counts().sort_index().items()},
                            "role_last": g.role_last.value_counts().to_dict(), "median_mcap_bn": float(g.mcap.median() / 1e9),
                            "above_consensus_share": float((g.opt_last > 0).mean()), "mean_bhar252": float(g.bhar_252.mean()), "median_bhar252": float(g.bhar_252.median())}

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    (RESULTS / "stage2_results.json").write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
    print("\nwrote", RESULTS / "stage2_results.json")


if __name__ == "__main__":
    main()
