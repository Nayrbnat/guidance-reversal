"""
STAGE 13 COPY of src/gg_study.py (docs/STAGE13_SPEC.md): identical tests with the wider sale condition (gr_candidate9) swapped in; S3 dropped; original outputs untouched.

gg_study.py -- run the pre-specified Guidance Reversal tests described in PREREGISTRATION.md.

    python src/gg_study.py --start 2004 --end 2025                  # real run (needs validation/candidate_labels.csv)
    python src/gg_study.py --start 2004 --end 2025 --dry-run       # mechanics check on a RANDOM flag; says so in the output

Reads outputs/data/events.csv (from `guide_and_go.py build`), the cached CRSP/WRDS pulls and Fama-French factors,
and writes outputs/results/study_results.json and outputs/data/study_windows.pkl (used by the analysis notebook).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import gg_tests as GT
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS


def load_events(labels_path: Path | None, dry_run: bool) -> tuple[pd.DataFrame, str]:
    ev = pd.read_csv(DATA / "events.csv", parse_dates=["dep_ann", "leftofc", "first_guid_date"])
    ev["dep_ann_final"] = ev.dep_ann
    ev["abrupt_ann"] = np.nan
    src = "rule"
    if labels_path is not None and labels_path.exists():
        lab = pd.read_csv(labels_path, parse_dates=["ann_date"])
        lab = lab.drop_duplicates("event_id").set_index("event_id")
        ev = ev.join(lab[["ann_date", "abrupt_ann_label", "ann_found_ann"]], on="event_id")
        has = ev.abrupt_ann_label.notna()
        ev.loc[has, "abrupt_ann"] = ev.loc[has, "abrupt_ann_label"].astype(float)
        use_date = ev.ann_date.notna() & (ev.ann_found_ann == 1)
        ev.loc[use_date, "dep_ann_final"] = ev.loc[use_date, "ann_date"]
        src = "annotator"
    ev["year"] = ev.dep_ann_final.dt.year
    return ev, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2004)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--labels", default="validation/candidate_labels.csv")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--reps-secondary", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ev, src = load_events(Path(a.labels), a.dry_run)
    c13 = pd.read_pickle(DATA / "stage13_events_counts.pkl").set_index("event_id")           # Stage 13: wider sale condition
    ev["gr_candidate_orig"] = ev.gr_candidate
    ev["gr_candidate"] = ev.event_id.map(c13.gr_candidate9).fillna(0).astype(int)
    ev["abrupt_ann"] = ev.abrupt_ann.fillna(ev.abrupt_rule)                                  # annotator label where one exists, rule classifier otherwise
    ev["gr_rule"] = ev.event_id.map(c13.gr_rule9).fillna(0).astype(int)
    print(f"events: {len(ev):,} | candidate (guidance + sale) events: {int(ev.gr_candidate.sum())} | abrupt source: {src}")
    if src == "annotator":
        ann_flag = (ev.gr_candidate == 1) & (ev.abrupt_ann == 1)
        missing = int(((ev.gr_candidate == 1) & ev.abrupt_ann.isna()).sum())
        if missing:
            print(f"  [warn] {missing} candidates have no annotator label and are NOT flagged")
        ev["gr_flag"] = ann_flag.astype(int)
    else:
        ev["gr_flag"] = ev.gr_rule
    ev["gr_flag_rule"] = ev.gr_rule

    db = G.WRDS(refresh=False)
    crsp, mkt, _names = G.load_crsp(db, ev.permno.unique(), a.start, a.end)
    fac_all = GT.load_factors(db)
    cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
    cal = cal[np.isin(cal, fac_all.index.values)]
    fac = fac_all.reindex(pd.DatetimeIndex(cal))
    T_ = len(cal)
    print(f"calendar: {pd.Timestamp(cal[0]):%Y-%m-%d} to {pd.Timestamp(cal[-1]):%Y-%m-%d} ({T_:,} trading days)")

    use = ev[(ev.dep_ann_final.dt.year >= a.start) & (ev.dep_ann_final.dt.year <= a.end)].copy()
    win252 = GT.build_windows(use, crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
    win126 = GT.build_windows(use, crsp, cal, date_col="dep_ann_final", skip=2, horizon=126)
    use["has_win"] = use.event_id.isin(win252.keys())
    print(f"events with usable 12m windows: {int(use.has_win.sum()):,} of {len(use):,}")

    rng = np.random.default_rng(0)
    pool_ok = use[use.has_win & (use.ann_found == 1)]                  # placebo pool: announcement identified
    flagged = use[(use.gr_flag == 1) & use.has_win]
    if a.dry_run:                                                     # mechanics check: a RANDOM flag, not the real one
        flagged = pool_ok.sample(min(66, len(pool_ok)), random_state=1)
        use["gr_flag"] = use.event_id.isin(flagged.event_id).astype(int)
        print("*** DRY RUN: the 'flagged' set below is RANDOM; these numbers say nothing about Guidance Reversal ***")
    fl_ids = flagged.event_id.tolist()
    print(f"flagged events with windows: {len(fl_ids)} (flag total {int(use.gr_flag.sum())})")

    results = {"dry_run": bool(a.dry_run), "abrupt_source": src, "n_events": int(len(use)), "n_with_window": int(use.has_win.sum()),
               "n_candidates": int(use.gr_candidate.sum()), "n_flagged": int(use.gr_flag.sum()), "n_flagged_with_window": len(fl_ids),
               "calendar_days": int(T_)}

    def run_test(name, ids, windows, factors, pool, reps, strata="year", maxlags=10, side="lower"):
        ids = [i for i in ids if i in windows]
        port, cnt = GT.portfolio(windows, ids, T_)
        r = GT.alpha_test(port, cnt, fac, factors, maxlags)
        r["n_events"] = len(ids)
        if len(ids) >= 5:
            null = GT.placebo_alphas(windows, ids, pool, fac, T_, strata_col=strata, reps=reps, seed=7, factors=factors, maxlags=maxlags)
            r["placebo_p"] = GT.placebo_p(r["alpha_ann"], null, side)
            r["placebo_mean"] = float(np.nanmean(null))
            r["placebo_sd"] = float(np.nanstd(null))
            r["placebo_q05"], r["placebo_q95"] = [float(x) for x in np.nanpercentile(null, [5, 95])]
            r["_null"] = null
        else:
            r["placebo_p"] = np.nan
        r["name"] = name
        return r

    pool_df = pool_ok[["event_id", "year"]].copy()
    pool_df = pd.concat([pool_df, flagged[["event_id", "year"]]]).drop_duplicates("event_id")
    # --- primary
    print("running primary test ...")
    prim = run_test("PRIMARY: flagged, 12m, FF4", fl_ids, win252, GT.FACTORS_FF4, pool_df, a.reps)
    results["primary"] = {k: v for k, v in prim.items() if k != "_null"}
    results["primary_pass"] = bool(prim["n_events"] >= 40 and np.isfinite(prim["placebo_p"]) and prim["placebo_p"] < 0.05)
    print(f"  alpha {prim['alpha_ann']:+.1%}/yr  t={prim['t']:.2f}  placebo p={prim['placebo_p']:.3f}  n={prim['n_events']}")
    # sensitivity: placebo from all other departures
    all_pool = use[use.has_win][["event_id", "year"]]
    results["primary_allpool"] = {k: v for k, v in run_test("primary vs all-departures pool", fl_ids, win252, GT.FACTORS_FF4,
                                                              pd.concat([all_pool, flagged[["event_id", "year"]]]).drop_duplicates("event_id"),
                                                              a.reps_secondary).items() if k != "_null"}

    if a.dry_run:
        # A dry run must never compute returns for the REAL subsets (rule-based flag, candidates, ...): doing so
        # would reveal results before the pre-specified primary test. Mechanics check on the random flag only.
        results["secondary"] = []
        G.make_output_dirs()
        (RESULTS / "stage13_DRYRUN.json").write_text(json.dumps({k: v for k, v in results.items() if k in
                                                                   ("dry_run", "primary", "n_events", "n_with_window")}, indent=1, default=float), encoding="utf-8")
        print("dry run complete (random flag only; secondary tests deliberately skipped)")
        return

    # --- secondary family (Holm-adjusted together)
    sec = []
    ceo = flagged[flagged.role == "CEO"].event_id.tolist()
    sec.append(run_test("S1 CEO only", ceo, win252, GT.FACTORS_FF4, pool_df[pool_df.event_id.isin(use[use.role == "CEO"].event_id) | pool_df.event_id.isin(ceo)], a.reps_secondary))
    sec.append(run_test("S2 6-month hold", fl_ids, win126, GT.FACTORS_FF4, pool_df, a.reps_secondary))
    rule_ids = use[(use.gr_flag_rule == 1) & use.has_win].event_id.tolist()
    sec.append(run_test("S4 rule-based abrupt instead of annotators", rule_ids, win252, GT.FACTORS_FF4, pool_df, a.reps_secondary))
    sec.append(run_test("S5 six-factor model", fl_ids, win252, GT.FACTORS_FF6, pool_df, a.reps_secondary))
    cand_ids = use[(use.gr_candidate == 1) & use.has_win].event_id.tolist()
    sec.append(run_test("S6 guidance + sale, any exit type (no abruptness)", cand_ids, win252, GT.FACTORS_FF4, pool_df, a.reps_secondary))
    crisis = {2008, 2009, 2020}
    nc = flagged[~flagged.year.isin(crisis)].event_id.tolist()
    sec.append(run_test("S7 excluding 2008-09 and 2020", nc, win252, GT.FACTORS_FF4, pool_df[~pool_df.year.isin(crisis)], a.reps_secondary))
    h1 = flagged[flagged.year <= 2014].event_id.tolist()
    h2 = flagged[flagged.year >= 2015].event_id.tolist()
    sec.append(run_test("S8a announcements 2004-2014", h1, win252, GT.FACTORS_FF4, pool_df[pool_df.year <= 2014], a.reps_secondary))
    sec.append(run_test("S8b announcements 2015-2025", h2, win252, GT.FACTORS_FF4, pool_df[pool_df.year >= 2015], a.reps_secondary))
    adj = GT.holm({s["name"]: s.get("placebo_p", np.nan) for s in sec})
    for s in sec:
        s["holm_p"] = adj.get(s["name"], np.nan)
        print(f"  {s['name']:<58s} n={s['n_events']:>3d} alpha {s['alpha_ann']:+7.1%}  placebo p={s.get('placebo_p', float('nan')):.3f}  Holm p={s['holm_p']:.3f}")
    results["secondary"] = [{k: v for k, v in s.items() if k != "_null"} for s in sec]

    # --- exploratory (not part of the pre-specified family)
    results["exploratory"] = {}
    ex = use[use.has_win].copy()
    if ex.bhar_252.notna().sum() > 100 and len(fl_ids) >= 10:
        results["exploratory"]["matched_bhar"] = GT.matched_difference(ex, fl_ids)
    # portfolio series for plotting
    port, cnt = GT.portfolio(win252, fl_ids, T_)
    results["_port_dates"] = None

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    results.pop("_port_dates")
    G.make_output_dirs()
    (RESULTS / "stage13_results.json").write_text(json.dumps(clean(results), indent=1), encoding="utf-8")
    print("wrote", RESULTS / "stage13_results.json")


if __name__ == "__main__":
    main()
