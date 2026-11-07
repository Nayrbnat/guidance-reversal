"""
Tests for gg_lossrate (the registered large-loss-rate test, docs/LOSS_RATE_SPEC.md): flag logic, BHAR arithmetic, and the
placebo test on planted data. No data or WRDS access needed.
Run:  python tests/test_gg_lossrate.py    (from the repo root; no pytest needed)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # modules live in src/
import gg_lossrate as LR

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


def test_flag():
    ev = pd.DataFrame({"n_guid_12m": [2, 2, 2, 0, np.nan, 3],
                       "n_unfulfilled": [1, 0, 2, 1, 1, np.nan],
                       "sell_val_post_guid": [5e5, 5e5, 0.0, 5e5, 5e5, 5e5]})
    out = LR.flags(ev.copy())
    check("flag needs guidance, a sale after it, and guidance still outstanding", out.flag.tolist() == [1, 0, 0, 0, 0, 0],
          str(out.flag.tolist()))
    check("comparator = guided with outstanding guidance, whatever the sale", out.guided_unfulfilled.tolist() == [1, 0, 1, 0, 0, 0],
          str(out.guided_unfulfilled.tolist()))


def test_bhar():
    n = 30
    mk = np.full(n + 2, 0.01)
    idx = np.arange(1, n + 1)
    win = {"a": (idx, np.full(n, -0.02)),                                   # firm falls 2% a day, market rises 1% a day
           "b": (idx, np.full(n, 0.01))}                                    # tracks the market
    b = LR.bhars(win, mk)
    check("BHAR = product(1+r_firm) - product(1+r_mkt)", abs(b.loc["a", "bhar"] - ((0.98 ** n) - (1.01 ** n))) < 1e-12, str(b.bhar.to_dict()))
    check("a stock that tracks the market has BHAR 0", abs(b.loc["b", "bhar"]) < 1e-12)
    check("observed days are counted", int(b.loc["a", "ndays"]) == n)
    check("windows shorter than 20 observed days are dropped", LR.bhars({"c": (np.array([1, 2]), np.array([0.0, 0.0]))}, mk).empty)


def _pool(n_per_year=400, rate=0.20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2005, 2015):
        loss = rng.random(n_per_year) < rate
        rows.append(pd.DataFrame({"year": y, "bhar": np.where(loss, -0.5, 0.1), "ndays": 250,
                                  "flag": 0, "elig": True, "event_id": [f"{y}_{i}" for i in range(n_per_year)]}))
    return pd.concat(rows, ignore_index=True)


def _flagged(rate, n_per_year=20, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2005, 2015):
        loss = rng.random(n_per_year) < rate
        rows.append(pd.DataFrame({"year": y, "bhar": np.where(loss, -0.5, 0.1), "ndays": 250, "flag": 1, "elig": True,
                                  "event_id": [f"f{y}_{i}" for i in range(n_per_year)]}))
    return pd.concat(rows, ignore_index=True)


def test_placebo_planted():
    pool = _pool()
    same = pd.concat([pool, _flagged(0.20)], ignore_index=True)
    worse = pd.concat([pool, _flagged(0.45)], ignore_index=True)
    better = pd.concat([pool, _flagged(0.05)], ignore_index=True)
    r_same = LR.test_one(same, "flag", -0.30, 1000)
    r_worse = LR.test_one(worse, "flag", -0.30, 1000)
    r_better = LR.test_one(better, "flag", -0.30, 1000)
    check("planted excess losses are detected (one-sided p < 0.05)", r_worse["p_one_sided"] < 0.05, str(r_worse["p_one_sided"]))
    check("fewer losses than the pool give p near 1 (wrong direction)", r_better["p_one_sided"] > 0.95, str(r_better["p_one_sided"]))
    check("the same loss rate as the pool is not significant", r_same["p_one_sided"] > 0.05, str(r_same["p_one_sided"]))
    check("placebo mean recovers the pool rate", abs(r_same["placebo_mean"] - 0.20) < 0.03, str(r_same["placebo_mean"]))
    check("counts reported", r_worse["n_flagged"] == 200 and r_worse["pool_n"] == 4000)


def test_within_comparator():
    a = _flagged(0.30, 60)
    a["guided_unfulfilled"] = 1
    a["dep_ann_final"] = pd.to_datetime(a.year.astype(str) + "-06-01")
    b = _pool(60, 0.10)
    b["guided_unfulfilled"] = 1
    b["dep_ann_final"] = pd.to_datetime(b.year.astype(str) + "-06-01")
    w = LR.within_comparator(pd.concat([a, b], ignore_index=True), -0.30, 200)
    check("within-departure comparator recovers a planted gap", abs(w["diff"] - 0.20) < 0.06, str(w["diff"]))
    check("its bootstrap interval brackets the estimate", w["ci95"][0] < w["diff"] < w["ci95"][1], str(w["ci95"]))


if __name__ == "__main__":
    for fn in (test_flag, test_bhar, test_placebo_planted, test_within_comparator):
        fn()
    print("\n%d failures" % len(FAILS), FAILS)
    sys.exit(1 if FAILS else 0)
