"""
Tests for gg_stage11 (docs/STAGE11_SPEC.md): sparse-effect tests, Bayes factor, non-inferiority, regression calibration,
the event-window abnormal return and the cluster bootstrap. No data or WRDS access needed.
Run:  python tests/test_gg11.py    (from the repo root; no pytest needed)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gg_stage11 as S

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


def test_pvals_and_stats():
    rng = np.random.default_rng(0)
    pool = rng.normal(size=3000)
    years = np.repeat([2005, 2006, 2007], 1000)
    vals, vy = np.array([-5.0, 0.0, 5.0]), np.array([2005, 2006, 2007])
    p = S.lower_tail_pvals(vals, vy, pool, years)
    check("lower-tail p-value is tiny for a very bad outcome and near 1 for a very good one", p[0] < 0.01 and p[2] > 0.99, str(p))
    check("p-value of a median outcome is near 0.5", 0.35 < p[1] < 0.65, str(p))
    check("HC grows when many p-values are small", S.hc_stat(np.r_[np.full(10, 1e-4), rng.uniform(size=90)]) > S.hc_stat(rng.uniform(size=100)))
    check("positive Cauchy grows when many p-values are small", S.cauchy_pos_stat(np.r_[np.full(10, 1e-4), rng.uniform(size=90)]) > S.cauchy_pos_stat(rng.uniform(size=100)))
    check("HC of uniform p-values is modest", S.hc_stat(rng.uniform(size=500)) < 6)


def test_sparse_power_and_size():
    """Under the null the placebo-calibrated p-values reject about 5%; with a few large losses the sparse tests beat the mean test."""
    rng = np.random.default_rng(1)
    years_pool = np.repeat(np.arange(2005, 2013), 400)
    n_per_year = 12
    yrs = np.repeat(np.arange(2005, 2013), n_per_year)
    rej = {"hc": 0, "cauchy": 0, "mean": 0}
    sparse = {"hc": 0, "cauchy": 0, "mean": 0}
    R = 80
    for r in range(R):
        pool = rng.normal(size=len(years_pool))
        null = rng.normal(size=len(yrs))
        alt = rng.normal(size=len(yrs))
        alt[rng.choice(len(alt), 5, replace=False)] -= 3.5              # 5 of 96 firms lose a lot, the rest are unchanged
        a = S.sparse_calibrate(null, yrs, pool, years_pool, reps=250, seed=r)
        b = S.sparse_calibrate(alt, yrs, pool, years_pool, reps=250, seed=r)
        for k in rej:
            rej[k] += a[f"p_{k}"] < 0.05
            sparse[k] += b[f"p_{k}"] < 0.05
    size = {k: v / R for k, v in rej.items()}
    power = {k: v / R for k, v in sparse.items()}
    print("   size:", size, "| power under 5-of-96 large losses:", power)
    check("false-positive rate under the null is near 5% for every statistic", all(v < 0.15 for v in size.values()), str(size))
    check("higher criticism beats the mean test when only a few firms are hit hard", power["hc"] > power["mean"] + 0.10, str(power))
    check("the positive Cauchy test also beats the mean test", power["cauchy"] > power["mean"] + 0.10, str(power))


def test_bayes_factor():
    est, se, scale = -0.13, 0.15, 0.5
    # numerical marginal likelihood under H1: half-normal in the negative direction
    f = lambda th: 2 * stats.norm.pdf(th, 0, scale) * stats.norm.pdf(est, -th, se)      # theta_harm = -effect
    m1, _ = integrate.quad(f, 0, np.inf)
    num = stats.norm.pdf(est, 0, se) / m1
    check("closed-form Bayes factor equals numerical integration", abs(S.bf01_halfnormal(est, se, scale, harm_sign=-1) - num) < 1e-6,
          f"{S.bf01_halfnormal(est, se, scale, -1)} vs {num}")
    check("an estimate of zero favours the null (BF01 > 1)", S.bf01_halfnormal(0.0, 0.2, 0.5) > 1)
    check("a large estimate in the harmful direction favours the effect (BF01 < 1/3)", S.bf01_halfnormal(1.5, 0.2, 0.5) < 1 / 3)
    check("a large estimate in the protective direction favours the null", S.bf01_halfnormal(-1.5, 0.2, 0.5) > 1)
    check("a more precise estimate gives stronger evidence for the null", S.bf01_halfnormal(0.0, 0.05, 0.5) > S.bf01_halfnormal(0.0, 0.5, 0.5))


def test_noninferiority():
    r = S.noninferiority(est=-3.4, se=2.2, sesoi=5.0, harm_sign=1)
    check("harm bound = estimate + 1.645 se", abs(r["bound95"] - (-3.4 + 1.645 * 2.2)) < 1e-9)
    check("a bound inside the SESOI rejects harm", r["reject_harm_at_sesoi"] and r["p"] < 0.05, str(r))
    r2 = S.noninferiority(est=-0.6, se=5.5, sesoi=-5.0 * -1, harm_sign=1)
    check("an imprecise estimate cannot reject a moderate SESOI", not S.noninferiority(est=1.0, se=5.5, sesoi=5.0)["reject_harm_at_sesoi"])


def test_regression_calibration():
    rng = np.random.default_rng(2)
    N, theta_true = 200_000, -0.8                       # effect of one SD of the true score on the outcome
    truth = rng.normal(size=N)
    model = 0.8 * truth + 0.6 * rng.normal(size=N)      # model score = noisy proxy of the truth (reliability 0.64)
    y = theta_true * truth + 3 * rng.normal(size=N)
    zm = (model - model.mean()) / model.std()
    theta_m = np.polyfit(zm, y, 1)[0]
    aud = rng.choice(N, 300, replace=False)
    rc = S.regression_calibration(truth[aud], model[aud], model.std(), theta_m, 0.01, reps=1000)
    check("the raw coefficient on the noisy score is attenuated", abs(theta_m) < abs(theta_true) * 0.9, str(theta_m))
    check("regression calibration recovers the true effect of a standard deviation of the true score",
          abs(rc["theta_corrected"] - theta_true) < 0.12, f"{rc['theta_corrected']} vs {theta_true}")
    check("its interval covers the truth", rc["ci95"][0] < theta_true < rc["ci95"][1], str(rc["ci95"]))
    check("a perfect proxy needs no correction", abs(S.regression_calibration(truth[aud], truth[aud], truth.std(), -0.8, 0.01, reps=200)["scale_factor"] - 1) < 0.05)


def test_regression_calibration_unrepresentative_audit():
    """The audit sample oversamples high model scores (as the real one does). The registered formula takes sd(H) from that
    sample and overstates the correction; the population-consistent version rebuilds sd(H) from the slope, the residual
    variance and the population sd of the model score, and recovers the truth."""
    rng = np.random.default_rng(5)
    N, theta_true = 300_000, -0.8
    truth = rng.normal(size=N)
    model = 0.8 * truth + 0.6 * rng.normal(size=N)
    y = theta_true * truth + 3 * rng.normal(size=N)
    zm = (model - model.mean()) / model.std()
    theta_m = np.polyfit(zm, y, 1)[0]
    w = 1.0 + model ** 2                                      # audit sample oversamples both tails of the model score (wider than the population)
    aud = rng.choice(N, 400, replace=False, p=w / w.sum())
    rc = S.regression_calibration(truth[aud], model[aud], model.std(), theta_m, 0.01, reps=1000)
    check("the audit sample is wider than the population (the situation being tested)", model[aud].std() > 1.15 * model.std(),
          f"{model[aud].std():.2f} vs {model.std():.2f}")
    check("the as-registered formula is biased when the audit sample is unrepresentative",
          abs(rc["theta_corrected"] - theta_true) > abs(rc["theta_corrected_pop"] - theta_true) + 0.05,
          f"registered {rc['theta_corrected']:.3f}, population-consistent {rc['theta_corrected_pop']:.3f}, truth {theta_true}")
    check("the population-consistent formula recovers the truth", abs(rc["theta_corrected_pop"] - theta_true) < 0.12,
          f"{rc['theta_corrected_pop']:.3f}")
    check("its interval covers the truth", rc["ci95_pop"][0] < theta_true < rc["ci95_pop"][1], str(rc["ci95_pop"]))


def test_car_window():
    cal = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(3)
    fac = pd.DataFrame({"mktrf": rng.normal(0, 0.01, 400), "smb": rng.normal(0, 0.005, 400), "hml": rng.normal(0, 0.005, 400),
                        "umd": rng.normal(0, 0.005, 400), "rf": 0.0001}, index=cal)
    r = fac.mktrf.values * 1.1 + fac.rf.values + rng.normal(0, 0.004, 400)
    date_i = 300
    r[date_i] += 0.05                                              # a 5% reaction on the event day
    crsp = pd.DataFrame({"permno": 1, "date": cal, "ret": r})
    ev = pd.DataFrame({"permno": [1], "d": [cal[date_i]]})
    out = S.car_window(ev, crsp, fac, "d", (-1, 1))
    check("CAR picks up a planted event-day reaction", 0.03 < out.car.iloc[0] < 0.07, str(out.car.iloc[0]))
    ev2 = pd.DataFrame({"permno": [1], "d": [cal[150]]})
    check("CAR of an ordinary day is small", abs(S.car_window(ev2, crsp, fac, "d", (-1, 1)).car.iloc[0]) < 0.02)
    check("an event too early to estimate a beta is NaN", np.isnan(S.car_window(pd.DataFrame({"permno": [1], "d": [cal[30]]}), crsp, fac, "d").car.iloc[0]))


def test_cluster_boot():
    rng = np.random.default_rng(4)
    n = 3000
    cl = rng.integers(0, 60, n)
    g = (rng.random(n) < 0.3).astype(int)
    y = rng.normal(size=n) - 0.8 * g
    r = S.cluster_boot_diff(y, g, cl, reps=1000)
    check("bootstrap recovers a planted difference", abs(r["diff"] + 0.8) < 0.15, str(r["diff"]))
    check("a planted negative difference has a small one-sided p", r["p_one_sided_neg"] < 0.01, str(r["p_one_sided_neg"]))
    r0 = S.cluster_boot_diff(rng.normal(size=n), g, cl, reps=1000)
    check("no difference is not significant", r0["p_one_sided_neg"] > 0.02, str(r0["p_one_sided_neg"]))
    check("interval brackets the estimate", r["ci95"][0] < r["diff"] < r["ci95"][1])


if __name__ == "__main__":
    for fn in (test_pvals_and_stats, test_bayes_factor, test_noninferiority, test_regression_calibration, test_car_window,
               test_cluster_boot, test_sparse_power_and_size):
        fn()
    print("\n%d failures" % len(FAILS), FAILS)
    sys.exit(1 if FAILS else 0)
