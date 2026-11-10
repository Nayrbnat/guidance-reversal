"""
gg_lossrate.py -- the registered large-loss-rate test (docs/LOSS_RATE_SPEC.md).

    python src/gg_lossrate.py --start 2004 --end 2025 --reps 5000

Question: among departures, are firms where the executive guided, sold and left before the guidance came due more likely
to lose a large amount relative to the market than other departures? Outcome: 12-month market-adjusted BHAR (days +2 to
+253 after the announcement) below -30 percentage points (sensitivities: -20, -40, complete windows only, the strict
42-firm flag). Inference: placebo groups of non-flagged departures, matched on announcement-year counts.

Reads outputs/data/events.csv and the cached CRSP pulls; writes outputs/results/lossrate_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg_tests as GT
import gg_variants as GV
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS
THRESHOLDS = (-0.30, -0.20, -0.40)          # primary first, then the two registered sensitivities


def flags(ev: pd.DataFrame) -> pd.DataFrame:
    """Registered flag: guided in the prior 365 days, sold after the first of those guidance releases, and at least one
    of them covers a period that ends after the announcement. Adds 'flag' and 'guided_unfulfilled' (the comparator)."""
    guided = ev.n_guid_12m.fillna(0) >= 1
    unf = ev.n_unfulfilled.fillna(0) >= 1
    sold = ev.sell_val_post_guid.fillna(0) > 0
    ev["guided_unfulfilled"] = (guided & unf).astype(int)
    ev["flag"] = (guided & unf & sold).astype(int)
    return ev


def bhars(windows: dict, mkt_by_idx: np.ndarray) -> pd.DataFrame:
    """event_id -> (BHAR, number of observed days). BHAR = prod(1+r_firm) - prod(1+r_market) over the same days."""
    rows = {}
    for eid, (idx, r) in windows.items():
        m = mkt_by_idx[idx]
        ok = ~np.isnan(m)
        if ok.sum() < 20:
            continue
        rows[eid] = (np.prod(1 + r[ok]) - np.prod(1 + m[ok]), int(ok.sum()))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["bhar", "ndays"])


def placebo_rates(pool: pd.DataFrame, need: pd.Series, thr: float, reps: int, seed: int) -> np.ndarray:
    """Share with BHAR < thr in random groups of pool events, with `need` events per announcement year."""
    rng = np.random.default_rng(seed)
    by_y = {y: (g.bhar.values < thr) for y, g in pool.groupby("year")}
    out = np.empty(reps)
    for r in range(reps):
        hits = tot = 0
        for y, k in need.items():
            arr = by_y.get(y)
            if arr is None or len(arr) == 0:
                continue
            pick = rng.choice(len(arr), size=min(k, len(arr)), replace=False)
            hits += int(arr[pick].sum())
            tot += len(pick)
        out[r] = hits / tot if tot else np.nan
    return out


def test_one(use: pd.DataFrame, flag_col: str, thr: float, reps: int, seed: int = 0, min_days: int = 20) -> dict:
    u = use[use.ndays >= min_days]
    fl = u[u[flag_col] == 1]
    pool = u[(u.flag == 0) & u.elig]                      # placebo pool: non-flagged eligible departures (as in Stage 1)
    if len(fl) == 0:
        return {"n_flagged": 0}
    need = fl.groupby("year").size()
    obs = float((fl.bhar < thr).mean())
    dr = placebo_rates(pool, need, thr, reps, seed)
    p = (1 + int((dr >= obs).sum())) / (1 + reps)
    return {"threshold": thr, "n_flagged": int(len(fl)), "n_loss_flagged": int((fl.bhar < thr).sum()),
            "rate_flagged": obs, "placebo_mean": float(np.nanmean(dr)), "placebo_sd": float(np.nanstd(dr)),
            "diff": obs - float(np.nanmean(dr)), "p_one_sided": p,
            "detectable_diff": float(2.5 * np.nanstd(dr)), "pool_n": int(len(pool)),
            "placebo_q05": float(np.nanquantile(dr, 0.05)), "placebo_q95": float(np.nanquantile(dr, 0.95))}


def within_comparator(use: pd.DataFrame, thr: float, boot: int, seed: int = 0) -> dict:
    """Flagged vs departures with outstanding guidance but no sale: difference in large-loss shares, CI by quarter bootstrap."""
    a = use[(use.flag == 1)]
    b = use[(use.guided_unfulfilled == 1) & (use.flag == 0) & use.elig]
    if len(a) == 0 or len(b) == 0:
        return {}
    d = pd.concat([a.assign(g=1), b.assign(g=0)])
    d["loss"] = (d.bhar < thr).astype(int)
    d["q"] = d.dep_ann_final.dt.to_period("Q").astype(str)
    diff = lambda x: x[x.g == 1].loss.mean() - x[x.g == 0].loss.mean()
    est = float(diff(d))
    qs = d.q.unique()
    by_q = {q: g for q, g in d.groupby("q")}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(boot):
        x = pd.concat([by_q[q] for q in rng.choice(qs, size=len(qs), replace=True)])
        if x.g.nunique() == 2:
            bs.append(diff(x))
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"threshold": thr, "n_flagged": int(len(a)), "n_nosale": int(len(b)),
            "rate_flagged": float(a.bhar.lt(thr).mean()), "rate_nosale": float(b.bhar.lt(thr).mean()),
            "diff": est, "ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2004)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--labels", default="validation/candidate_labels.csv")
    ap.add_argument("--extend", action="store_true", help="returns through 2025-12-31 (docs/LOSS_RATE_SPEC.md, Amendment 1)")
    a = ap.parse_args()

    ev = GV.load_eligible_events(a.start, a.end, a.labels)
    ev = flags(ev)
    # strict original flag (annotator-judged abrupt), for the sensitivity
    lab = pd.read_csv(a.labels).drop_duplicates("event_id").set_index("event_id")
    ev["strict"] = ((ev.gr_candidate == 1) & (ev.event_id.map(lab.abrupt_ann_label) == 1)).astype(int)

    db = G.WRDS(refresh=False)
    crsp, mkt, _ = G.load_crsp(db, ev.permno.unique(), a.start, a.end)
    if a.extend:
        import gg_extend as EXT
        crsp, mkt = EXT.extend_returns(db, crsp, mkt, ev.permno.unique())
    cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
    mk = mkt.drop_duplicates("date").set_index("date").mkt.reindex(pd.DatetimeIndex(cal)).astype(float).values

    elig = ev[ev.elig].copy()
    win = GT.build_windows(elig, crsp, cal, date_col="dep_ann_final", skip=2, horizon=252)
    bh = bhars(win, mk)
    use = elig.join(bh, on="event_id").dropna(subset=["bhar"])
    print(f"eligible departures {len(elig):,} | with a BHAR {len(use):,} | flagged {int(use.flag.sum())} | "
          f"strict flag {int(use.strict.sum())} | guided with outstanding guidance, no sale {int(((use.guided_unfulfilled==1)&(use.flag==0)).sum())}")

    res = {"spec": "docs/LOSS_RATE_SPEC.md", "reps": a.reps, "n_eligible": int(len(elig)), "n_with_bhar": int(len(use)),
           "overall_loss_share": {str(t): float((use.bhar < t).mean()) for t in THRESHOLDS},
           "primary": test_one(use, "flag", THRESHOLDS[0], a.reps),
           "sensitivity_threshold": {str(t): test_one(use, "flag", t, a.reps) for t in THRESHOLDS[1:]},
           "sensitivity_complete_windows": test_one(use, "flag", THRESHOLDS[0], a.reps, min_days=200),
           "sensitivity_strict_flag": test_one(use, "strict", THRESHOLDS[0], a.reps),
           "within_departures": within_comparator(use, THRESHOLDS[0], a.boot),
           "within_departures_t20": within_comparator(use, THRESHOLDS[1], a.boot),
           "within_departures_t40": within_comparator(use, THRESHOLDS[2], a.boot)}
    out = RESULTS / ("lossrate_results_ext.json" if a.extend else "lossrate_results.json")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    p = res["primary"]
    print(f"PRIMARY (BHAR < -30 pts): flagged {p['n_loss_flagged']}/{p['n_flagged']} = {p['rate_flagged']:.1%} | "
          f"placebo {p['placebo_mean']:.1%} (sd {p['placebo_sd']:.1%}) | diff {p['diff']*100:+.1f} pts | one-sided p {p['p_one_sided']:.3f} "
          f"| detectable {p['detectable_diff']*100:.1f} pts")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
