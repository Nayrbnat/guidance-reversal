"""
gg_stage11.py -- Stage 11 (docs/STAGE11_SPEC.md): four ways to extract more from the same data.

  P11a  reaction to the first guidance after the exit      car_window, cluster_boot_diff
  P11b  sparse-effect tests (higher criticism, Cauchy)     lower_tail_pvals, hc_stat, cauchy_pos_stat, sparse_calibrate
  P11c  how large an effect do the data rule out           bf01_halfnormal, noninferiority
  P11d  correcting the opacity results for the noisy score regression_calibration

Everything is written from the textbook definitions (see docs/METHODS_NOTES_POWER.md); nothing is copied from outside code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

FACTORS = ["mktrf", "smb", "hml", "umd"]


# --------------------------------------------------------------------------------------------------------- P11a
def car_window(df: pd.DataFrame, crsp: pd.DataFrame, fac: pd.DataFrame, date_col: str, window=(-1, 1), *,
               beta_window: int = 252, min_pre: int = 120, col: str = "car") -> pd.DataFrame:
    """Add `col`: the sum over trading days (day0 + window[0]) .. (day0 + window[1]) of r - rf - beta'F, in return units.
    day0 is the first trading day on or after df[date_col]. beta comes from an OLS (with intercept) of r - rf on the four
    factors over the `beta_window` trading days ending the day BEFORE the first day of the window (at least `min_pre`
    observations). A window with any missing return day is NaN. df needs a `permno` column."""
    cal = pd.DatetimeIndex(fac.index)
    F = fac[FACTORS].to_numpy(float)
    rf = fac["rf"].to_numpy(float)
    rets = {}
    c = crsp[crsp["permno"].isin(set(df["permno"].astype("int64")))]
    for p, g in c.groupby("permno"):
        rets[int(p)] = g.drop_duplicates("date").set_index("date")["ret"].reindex(cal).to_numpy(float)
    out = df.copy()
    vals = np.full(len(out), np.nan)
    for k, (p, t) in enumerate(zip(out["permno"].astype("int64").values, pd.to_datetime(out[date_col]))):
        r = rets.get(int(p))
        if r is None or pd.isna(t):
            continue
        i0 = int(cal.searchsorted(t, side="left"))
        a, b = i0 + window[0], i0 + window[1]
        if a < 1 or b >= len(cal):
            continue
        pre = np.arange(max(0, a - beta_window), a)
        pre = pre[np.isfinite(r[pre])]
        if len(pre) < min_pre:
            continue
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(pre)), F[pre]]), r[pre] - rf[pre], rcond=None)[0][1:]
        idx = np.arange(a, b + 1)
        if not np.all(np.isfinite(r[idx])):
            continue
        vals[k] = float(np.sum(r[idx] - rf[idx] - F[idx] @ beta))
    out[col] = vals
    return out


def cluster_boot_diff(y: np.ndarray, g: np.ndarray, cluster: np.ndarray, reps: int = 5000, seed: int = 0) -> dict:
    """Difference in means (group 1 minus group 0) with a bootstrap over clusters. Returns the estimate, the bootstrap SE,
    a 95% percentile interval, the one-sided p-value for 'difference < 0' from a normal approximation (Phi(est/se)) and the
    smallest detectable difference (2.49 x se)."""
    y, g, cl = np.asarray(y, float), np.asarray(g, int), np.asarray(cluster)
    est = y[g == 1].mean() - y[g == 0].mean()
    uc, inv = np.unique(cl, return_inverse=True)
    S1 = np.bincount(inv, weights=y * (g == 1), minlength=len(uc))
    N1 = np.bincount(inv, weights=(g == 1).astype(float), minlength=len(uc))
    S0 = np.bincount(inv, weights=y * (g == 0), minlength=len(uc))
    N0 = np.bincount(inv, weights=(g == 0).astype(float), minlength=len(uc))
    rng = np.random.default_rng(seed)
    b = np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)).astype(float)
        n1, n0 = (w * N1).sum(), (w * N0).sum()
        b[r] = np.nan if n1 == 0 or n0 == 0 else (w * S1).sum() / n1 - (w * S0).sum() / n0
    b = b[np.isfinite(b)]
    se = float(b.std(ddof=1))
    return {"diff": float(est), "se": se, "ci95": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
            "p_one_sided_neg": float(stats.norm.cdf(est / se)), "mde": 2.49 * se,
            "n1": int((g == 1).sum()), "n0": int((g == 0).sum()), "n_clusters": int(len(uc))}


# --------------------------------------------------------------------------------------------------------- P11b
def hc_stat(p, alpha0: float = 0.5) -> float:
    """Donoho-Jin higher criticism over the smallest alpha0 x n p-values."""
    p = np.sort(np.asarray(p, float))
    n = len(p)
    i = np.arange(1, n + 1)
    keep = i <= max(1, int(alpha0 * n))
    pp = np.clip(p[keep], 1e-12, 1 - 1e-12)
    return float(np.max(np.sqrt(n) * (i[keep] / n - pp) / np.sqrt(pp * (1 - pp))))


def cauchy_pos_stat(p) -> float:
    """Positive Cauchy combination: the mean over p < 0.5 of tan((0.5 - p) pi) (large values mean an excess of small p)."""
    p = np.asarray(p, float)
    q = p[p < 0.5]
    return float(np.sum(np.tan((0.5 - q) * np.pi)) / len(p))


def lower_tail_pvals(values, years, pool_values, pool_years) -> np.ndarray:
    """p_i = (1 + #{pool events in the same year with value <= value_i}) / (1 + #pool events in that year)."""
    values, years = np.asarray(values, float), np.asarray(years)
    pool_values, pool_years = np.asarray(pool_values, float), np.asarray(pool_years)
    p = np.full(len(values), np.nan)
    for y in np.unique(years):
        pv = np.sort(pool_values[pool_years == y])
        m = years == y
        p[m] = (1 + np.searchsorted(pv, values[m], side="right")) / (1 + len(pv))
    return p


def sparse_calibrate(values, years, pool_values, pool_years, reps: int = 5000, seed: int = 0) -> dict:
    """Observed HC, positive-Cauchy and mean-value statistics for `values` (flagged events) against the pool, with p-values
    from `reps` placebo groups drawn from the pool with the same number per year (leave-out p-values for each draw)."""
    values, years = np.asarray(values, float), np.asarray(years)
    pool_values, pool_years = np.asarray(pool_values, float), np.asarray(pool_years)
    obs_p = lower_tail_pvals(values, years, pool_values, pool_years)
    obs = {"hc": hc_stat(obs_p), "cauchy": cauchy_pos_stat(obs_p), "mean": float(values.mean())}
    yrs, need = np.unique(years, return_counts=True)
    pools = {y: np.sort(pool_values[pool_years == y]) for y in yrs}
    rng = np.random.default_rng(seed)
    hc = np.empty(reps)
    ca = np.empty(reps)
    mu = np.empty(reps)
    for r in range(reps):
        ps, vs = [], []
        for y, k in zip(yrs, need):
            pv = pools[y]
            k = min(int(k), len(pv) - 1)
            if k <= 0:
                continue
            v = pv[rng.choice(len(pv), k, replace=False)]
            cnt_pool = np.searchsorted(pv, v, side="right")
            cnt_draw = np.searchsorted(np.sort(v), v, side="right")
            ps.append((1 + cnt_pool - cnt_draw) / (1 + len(pv) - k))
            vs.append(v)
        ps, vs = np.concatenate(ps), np.concatenate(vs)
        hc[r], ca[r], mu[r] = hc_stat(ps), cauchy_pos_stat(ps), vs.mean()
    return {"n": int(len(values)), "observed": obs, "reps": reps,
            "p_hc": float((1 + (hc >= obs["hc"]).sum()) / (1 + reps)),
            "p_cauchy": float((1 + (ca >= obs["cauchy"]).sum()) / (1 + reps)),
            "p_mean": float((1 + (mu <= obs["mean"]).sum()) / (1 + reps)),
            "null_hc_q95": float(np.quantile(hc, 0.95)), "null_cauchy_q95": float(np.quantile(ca, 0.95)),
            "n_events_p_below_005": int((obs_p < 0.05).sum()), "expected_below_005": float(0.05 * len(values))}


# --------------------------------------------------------------------------------------------------------- P11c
def bf01_halfnormal(est: float, se: float, scale: float, harm_sign: int = 1) -> float:
    """Bayes factor for H0 (no effect) against H1 (effect half-normal with the given scale in the harmful direction), on the
    likelihood N(est; theta, se^2). harm_sign = +1 if a positive effect is the harmful (hypothesised) direction, -1 if negative."""
    x = harm_sign * est
    v = scale ** 2 + se ** 2
    m0 = stats.norm.pdf(x, 0.0, se)
    m1 = 2.0 * stats.norm.pdf(x, 0.0, np.sqrt(v)) * stats.norm.cdf((x * scale ** 2 / v) / np.sqrt(scale ** 2 * se ** 2 / v))
    return float(m0 / m1)


def noninferiority(est: float, se: float, sesoi: float, harm_sign: int = 1) -> dict:
    """One-sided test of H0: the effect is at least `sesoi` in the harmful direction. Reject at 5% if the one-sided 95% bound
    on the harmful side is below the SESOI. Also returns that bound (the largest harmful effect not ruled out)."""
    x = harm_sign * est
    bound = x + 1.645 * se
    return {"est_harm_dir": float(x), "bound95": float(bound), "sesoi": float(sesoi),
            "p": float(1 - stats.norm.cdf((sesoi - x) / se)), "reject_harm_at_sesoi": bool(bound < sesoi)}


# --------------------------------------------------------------------------------------------------------- P11d
def regression_calibration(h, m, sd_global_m: float, theta_m: float, se_m: float, reps: int = 5000, seed: int = 0) -> dict:
    """Correct a coefficient on the standardised model score for the noise in the score, using audit pairs (human h, model m).
    h = a + beta m (OLS on the pairs). theta_H(sd) = theta_m * sd(h) / (beta * sd_global_m): the effect of one standard
    deviation of the true (hand-scored) score. Uncertainty: bootstrap of the pairs combined with theta_m ~ N(est, se^2)."""
    h, m = np.asarray(h, float), np.asarray(m, float)
    n = len(h)

    def factors(hh, mm):
        """(as registered, population-consistent). The registered factor uses sd(H) of the audit sample; that is only right if
        the audit sample has the population's spread of the model score. The population-consistent factor rebuilds sd(H) in the
        population as sqrt(beta^2 sd_global^2 + residual variance), assuming the calibration line and its scatter transport."""
        b, a0 = np.polyfit(mm, hh, 1)
        s_u2 = np.sum((hh - (a0 + b * mm)) ** 2) / (len(hh) - 2)
        reg = hh.std(ddof=1) / (b * sd_global_m)
        pop = np.sqrt(b ** 2 * sd_global_m ** 2 + s_u2) / (b * sd_global_m)
        return b, reg, pop

    beta, f_reg, f_pop = factors(h, m)
    r2 = np.corrcoef(h, m)[0, 1] ** 2
    rng = np.random.default_rng(seed)
    out_r, out_p = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        if np.ptp(m[i]) == 0:
            out_r[r] = out_p[r] = np.nan
            continue
        _, fr, fp = factors(h[i], m[i])
        th = rng.normal(theta_m, se_m)
        out_r[r], out_p[r] = th * fr, th * fp
    out_r, out_p = out_r[np.isfinite(out_r)], out_p[np.isfinite(out_p)]
    q = lambda x: [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]
    return {"n_pairs": int(n), "beta_h_on_m": float(beta), "r2": float(r2), "sd_h": float(h.std(ddof=1)),
            "sd_global_m": float(sd_global_m), "sd_m_audit": float(m.std(ddof=1)),
            "theta_model": float(theta_m), "se_model": float(se_m),
            # as registered (sd of the hand scores in the audit sample)
            "scale_factor": float(f_reg), "theta_corrected": float(theta_m * f_reg), "median": float(np.median(out_r)),
            "ci95": q(out_r), "se_corrected": float(out_r.std(ddof=1)), "p_gt0": float((out_r >= 0).mean()), "p_lt0": float((out_r <= 0).mean()),
            # population-consistent (the ruling made after the first run; see docs/STAGE11_SPEC.md)
            "scale_factor_pop": float(f_pop), "theta_corrected_pop": float(theta_m * f_pop), "median_pop": float(np.median(out_p)),
            "ci95_pop": q(out_p), "se_corrected_pop": float(out_p.std(ddof=1)), "p_gt0_pop": float((out_p >= 0).mean()),
            "p_lt0_pop": float((out_p <= 0).mean())}
