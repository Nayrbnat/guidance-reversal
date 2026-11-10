"""
Tests for gg_stage12 (docs/STAGE12_SPEC.md): leak-free time splits, within-year AUC, the permutation test (size and power), the
pre-event market features, the gap bootstrap and Cochran's Q. No data or WRDS access needed.
Run:  python tests/test_gg12.py    (from the repo root; no pytest needed)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gg_stage12 as S

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


def synth(signal: float, n_per_year=150, years=range(2004, 2025), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        b1, b2 = rng.normal(size=n_per_year), rng.normal(size=n_per_year)
        r1, r2 = rng.normal(size=n_per_year), rng.normal(size=n_per_year)
        lp = -1.3 + 0.5 * b1 + signal * r1
        rows.append(pd.DataFrame({"year": y, "b1": b1, "b2": b2, "r1": r1, "r2": r2, "y": (rng.random(n_per_year) < 1 / (1 + np.exp(-lp))).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_split_is_leak_free():
    for t in (2010, 2015, 2024):
        tr = list(S.split_years(t))
        check(f"training years for {t} stop two years before it (embargo year left out)", max(tr) == t - 2 and min(tr) == t - 6, str(tr))
        check(f"the test year {t} and the embargo year are never in training", t not in tr and (t - 1) not in tr)
    check("expanding window starts in 2004", min(S.split_years(2015, expanding=True)) == 2004)
    df = synth(0.0)
    pred = S.rolling_oos(df, ["b1", "b2"], "y", "logit", range(2010, 2025))
    seen = {}
    ok = True
    for t in range(2010, 2025):
        tr = set(S.split_years(t))
        ok &= t not in tr and (t - 1) not in tr
    check("rolling_oos makes a prediction for every test year and each event once", pred.idx.is_unique and pred.year.nunique() == 15)


def test_year_auc():
    pred = pd.DataFrame({"year": [2010] * 20 + [2011] * 20, "y": ([1] * 8 + [0] * 12) * 2,
                         "p": list(np.r_[np.linspace(0.9, 0.6, 8), np.linspace(0.5, 0.1, 12)]) + list(np.r_[np.linspace(0.5, 0.1, 8), np.linspace(0.9, 0.6, 12)])})
    a = S.year_aucs(pred)
    check("a perfectly ranked year has AUC 1 and a perfectly inverted year has AUC 0", abs(a[2010] - 1) < 1e-9 and abs(a[2011]) < 1e-9, str(a.to_dict()))
    check("mean AUC averages the years", abs(S.mean_auc(pred) - 0.5) < 1e-9)
    few = pd.DataFrame({"year": [2010] * 10, "y": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0], "p": np.linspace(0, 1, 10)})
    check("years with fewer than 5 positives are skipped", len(S.year_aucs(few)) == 0)


def test_signal_detected_and_null_not():
    ty = range(2010, 2025)
    s = S.perm_delta_auc(synth(1.0, seed=1), ["b1", "b2"], ["r1", "r2"], "y", "logit", ty, reps=60)
    n = S.perm_delta_auc(synth(0.0, seed=2), ["b1", "b2"], ["r1", "r2"], "y", "logit", ty, reps=60)
    print(f"   planted signal: delta {s['delta']:+.3f}, p {s['p_one_sided']:.3f} | null: delta {n['delta']:+.3f}, p {n['p_one_sided']:.3f}")
    check("a planted signal in the R block raises out-of-sample AUC", s["delta"] > 0.03, str(s["delta"]))
    check("and the permutation test detects it", s["p_one_sided"] < 0.05, str(s["p_one_sided"]))
    check("with no signal the permutation test does not reject", n["p_one_sided"] > 0.05, str(n["p_one_sided"]))
    check("with no signal the AUC difference is near zero", abs(n["delta"]) < 0.03, str(n["delta"]))
    h = S.perm_delta_auc(synth(1.0, seed=3), ["b1", "b2"], ["r1", "r2"], "y", "hgb", range(2012, 2024, 3), reps=8)
    check("the gradient-boosting model also picks up the planted signal", h["delta"] > 0.02, str(h["delta"]))
    ci = S.boot_delta_auc(s["pred_base"], s["pred_br"], (synth(1.0, seed=1).year * 4 + np.random.default_rng(0).integers(0, 4, 21 * 150)).to_numpy(), reps=200)
    check("the bootstrap interval for the AUC difference is above zero when the signal is real", ci[0] > 0, str(ci))


def test_market_features():
    cal = pd.bdate_range("2018-01-01", periods=700)
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 700)
    crsp = pd.DataFrame({"permno": 1, "date": cal, "ret": r, "prc": 50.0, "shrout": 1000.0})
    t = cal[500]
    out = S.market_features(pd.DataFrame({"permno": [1], "d": [t]}), crsp, "d")
    i0 = 500
    exp_ret = np.prod(1 + r[i0 - 2 - 251:i0 - 2 + 1]) - 1
    check("12-month return ends two trading days before the announcement", abs(out.f_ret12.iloc[0] - exp_ret) < 1e-12, f"{out.f_ret12.iloc[0]} vs {exp_ret}")
    check("volatility is annualised daily volatility", abs(out.f_vol.iloc[0] - r[i0 - 2 - 251:i0 - 2 + 1].std(ddof=1) * np.sqrt(252)) < 1e-12)
    check("market cap is price times shares at that day", abs(out.f_lmcap.iloc[0] - np.log(50.0 * 1000.0 * 1000.0)) < 1e-12)
    crsp2 = crsp.copy()
    crsp2.loc[crsp2.date >= cal[499], "ret"] = 5.0                    # nothing at or after day -1 may enter a feature
    out2 = S.market_features(pd.DataFrame({"permno": [1], "d": [t]}), crsp2, "d")
    check("a huge return after day -2 does not change any feature (no look-ahead)", abs(out2.f_ret12.iloc[0] - out.f_ret12.iloc[0]) < 1e-12)
    check("an event with too little history gets NaN", np.isnan(S.market_features(pd.DataFrame({"permno": [1], "d": [cal[100]]}), crsp, "d").f_ret12.iloc[0]))


def test_gap_and_q():
    rng = np.random.default_rng(1)
    n = 2000
    cl = rng.integers(0, 40, n)
    g = (rng.random(n) < 0.3).astype(int)
    y = rng.normal(size=n) + 0.5 * g
    r = S.gap_boot(y, g, cl, reps=500)
    check("gap bootstrap recovers a planted gap", abs(r["gap"] - 0.5) < 0.15 and r["se"] > 0, str(r))
    q_same = S.cochran_q([0.5, 0.52, 0.48, 0.5], [0.1, 0.1, 0.1, 0.1])
    q_diff = S.cochran_q([0.1, 0.9, 0.1, 0.9], [0.1, 0.1, 0.1, 0.1])
    check("Cochran's Q does not reject for consistent estimates", q_same["p"] > 0.5, str(q_same))
    check("Cochran's Q rejects for very different estimates", q_diff["p"] < 0.001, str(q_diff))


def test_stage12b_models():
    """Every Stage 12b model family detects a planted signal in the R block and returns a finite score (spec Section 3)."""
    ty = range(2012, 2024, 3)
    for kind in ("xgb", "mlp", "svm_lin", "svm_poly"):
        s = S.perm_delta_auc(synth(1.0, seed=3), ["b1", "b2"], ["r1", "r2"], "y", kind, ty, reps=4)
        check(f"{kind}: picks up a planted signal out of sample", s["delta"] > 0.01 and np.isfinite(s["p_one_sided"]), str(s["delta"]))
    m = S.make_model("svm_poly")
    check("the polynomial SVM is not radial", m.kernel == "poly")


if __name__ == "__main__":
    for fn in (test_split_is_leak_free, test_year_auc, test_market_features, test_gap_and_q, test_signal_detected_and_null_not, test_stage12b_models):
        fn()
    print("\n%d failures" % len(FAILS), FAILS)
    sys.exit(1 if FAILS else 0)
