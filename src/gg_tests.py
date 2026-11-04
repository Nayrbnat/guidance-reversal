"""
gg_tests.py -- the statistical tests for the Guidance Reversal screen.

Primary design: a CALENDAR-TIME PORTFOLIO. Every flagged event is held from `skip` trading days after its
announcement for `horizon` trading days; on each calendar day the portfolio is the equal-weighted average of
the positions that are open. Its excess return is regressed on the Fama-French factors (+ momentum), and
the intercept (alpha) is the test statistic. This handles overlapping windows and cross-sectional correlation
(events bunch in time) that make event-time t-tests on BHARs over-confident.

Because there are few flagged events, inference also uses a PLACEBO distribution: the same test applied to
random sets of unflagged departures with the same number of events per year. The p-value is where the real
alpha falls in that distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTORS_FF4 = ["mktrf", "smb", "hml", "umd"]
FACTORS_FF6 = ["mktrf", "smb", "hml", "rmw", "cma", "umd"]


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def apply_delisting(crsp: pd.DataFrame, dl: pd.DataFrame) -> pd.DataFrame:
    """Fold CRSP delisting returns into the daily return series (Shumway 1997 convention).

    `crsp` has permno, date, ret ...; `dl` has permno, dlstdt, dlstcd, dlret. Without this, firms that fail
    (bankruptcy, performance delistings) silently lose their final crash, biasing long-run returns upward.
    A missing delisting return for a performance-related code (500-584) is set to -30%; other missing
    values are treated as 0.
    """
    if dl is None or dl.empty:
        return crsp
    d = dl.copy()
    d["permno"] = d.permno.astype(int)
    d["dlstdt"] = pd.to_datetime(d.dlstdt).astype("datetime64[ns]")
    d["dlret"] = pd.to_numeric(d.dlret, errors="coerce")
    code = pd.to_numeric(d.dlstcd, errors="coerce")
    d.loc[d.dlret.isna() & code.between(500, 584), "dlret"] = -0.30
    d["dlret"] = d.dlret.fillna(0.0)
    d = d.drop_duplicates(["permno", "dlstdt"])[["permno", "dlstdt", "dlret"]]
    out = crsp.copy()
    out["date"] = pd.to_datetime(out.date).astype("datetime64[ns]")
    m = out.merge(d, left_on=["permno", "date"], right_on=["permno", "dlstdt"], how="left")
    hit = m.dlret.notna()
    r = m.ret.fillna(0.0)
    m.loc[hit, "ret"] = (1 + r[hit]) * (1 + m.dlret[hit]) - 1
    m = m.drop(columns=["dlstdt", "dlret"])
    # delisting dates with no trading row get their own row
    have = set(zip(out.permno.values, out.date.values))
    extra = d[[(p, t) not in have for p, t in zip(d.permno.values, d.dlstdt.values)]]
    last_date = out.groupby("permno").date.max()
    extra = extra[extra.permno.map(last_date).notna() & (extra.dlstdt >= extra.permno.map(last_date))]
    if len(extra):
        add = pd.DataFrame({"permno": extra.permno.values, "date": extra.dlstdt.values, "ret": extra.dlret.values})
        for c in m.columns:
            if c not in add:
                add[c] = np.nan
        m = pd.concat([m, add[m.columns]], ignore_index=True)
    return m.sort_values(["permno", "date"]).reset_index(drop=True)


def build_windows(events: pd.DataFrame, crsp: pd.DataFrame, calendar: np.ndarray, *, date_col="dep_ann",
                  skip=2, horizon=252, id_col="event_id") -> dict:
    """event id -> (calendar index array, daily simple return array) for days skip..skip+horizon-1 after day 0.

    Day 0 is the first trading day on/after the announcement. Windows are truncated if the stock stops
    trading (delisting returns are already folded into the last day by apply_delisting) or data end.
    """
    cal = np.asarray(calendar, dtype="datetime64[ns]")
    by_p = {p: (g.date.values.astype("datetime64[ns]"), g.ret.values.astype(float))
            for p, g in crsp.sort_values(["permno", "date"]).groupby("permno")}
    out = {}
    for _, e in events.iterrows():
        pr = by_p.get(int(e.permno))
        if pr is None:
            continue
        d, r = pr
        i0 = int(np.searchsorted(d, np.datetime64(e[date_col], "ns")))
        a, b = i0 + skip, min(i0 + skip + horizon, len(d))
        if b - a < 20 or a >= len(d):
            continue
        dd, rr = d[a:b], r[a:b]
        ok = ~np.isnan(rr)
        idx = np.searchsorted(cal, dd[ok])
        valid = (idx < len(cal)) & (cal[np.minimum(idx, len(cal) - 1)] == dd[ok])
        out[e[id_col]] = (idx[valid], rr[ok][valid])
    return out


def portfolio(windows: dict, ids, T: int):
    """Equal-weighted calendar-time return and position count for the events in `ids`."""
    ii = [windows[i] for i in ids if i in windows]
    if not ii:
        return np.full(T, np.nan), np.zeros(T)
    idx = np.concatenate([w[0] for w in ii])
    ret = np.concatenate([w[1] for w in ii])
    cnt = np.bincount(idx, minlength=T).astype(float)
    s = np.bincount(idx, weights=ret, minlength=T)
    with np.errstate(invalid="ignore", divide="ignore"):
        port = np.where(cnt > 0, s / cnt, np.nan)
    return port, cnt


def alpha_test(port, cnt, fac: pd.DataFrame, factors=FACTORS_FF4, maxlags=10, min_pos=1) -> dict:
    """WLS (weights = open positions) of excess return on factors; Newey-West standard errors.

    `fac` is indexed like the calendar (same length as `port`) with columns rf + factor names.
    Returns annualised alpha (daily alpha x 252) and its t-stat.
    """
    import statsmodels.api as sm
    y = port - fac["rf"].values
    ok = (cnt >= min_pos) & np.isfinite(y) & fac[factors].notna().all(axis=1).values
    if ok.sum() < 250:
        return dict(alpha_ann=np.nan, t=np.nan, p=np.nan, n_days=int(ok.sum()), mean_pos=np.nan, betas={})
    X = sm.add_constant(fac.loc[ok, factors].values)
    m = sm.WLS(y[ok], X, weights=cnt[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return dict(alpha_ann=float(m.params[0] * 252), t=float(m.tvalues[0]), p=float(m.pvalues[0]),
                n_days=int(ok.sum()), mean_pos=float(cnt[ok].mean()),
                betas=dict(zip(factors, [float(b) for b in m.params[1:]])))


def placebo_alphas(windows: dict, flagged_ids, pool: pd.DataFrame, fac: pd.DataFrame, T: int, *, strata_col="year",
                   id_col="event_id", reps=1000, seed=0, factors=FACTORS_FF4, maxlags=10) -> np.ndarray:
    """Alphas of random same-size portfolios drawn from `pool` (unflagged events with data), matching the
    flagged set's count per `strata_col` (usually announcement year)."""
    rng = np.random.default_rng(seed)
    fl = pool[pool[id_col].isin(flagged_ids)]
    need = fl.groupby(strata_col).size()
    cand = pool[~pool[id_col].isin(flagged_ids) & pool[id_col].isin(windows.keys())]
    by_s = {s: g[id_col].values for s, g in cand.groupby(strata_col)}
    out = []
    for _ in range(reps):
        pick = []
        for s, k in need.items():
            arr = by_s.get(s)
            if arr is None or len(arr) == 0:
                continue
            pick.extend(rng.choice(arr, size=min(k, len(arr)), replace=False))
        port, cnt = portfolio(windows, pick, T)
        out.append(alpha_fast(port, cnt, fac, factors))
    return np.array(out, dtype=float)


def placebo_p(observed: float, null: np.ndarray, side="lower") -> float:
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return np.nan
    if side == "lower":
        return float((1 + np.sum(null <= observed)) / (1 + len(null)))
    if side == "upper":
        return float((1 + np.sum(null >= observed)) / (1 + len(null)))
    return float((1 + np.sum(np.abs(null - np.nanmean(null)) >= abs(observed - np.nanmean(null)))) / (1 + len(null)))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict name -> p."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def load_factors(db) -> pd.DataFrame:
    """Daily Fama-French 5 factors + momentum + risk-free rate from WRDS (decimal returns)."""
    loc = db.find([("ff", "fivefactors_daily"), ("ff", "factors_daily")])
    if loc is None:
        raise RuntimeError("Fama-French daily factors not available on this WRDS account")
    cols = ["date", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"]
    f = db.cached("ff_daily_factors", lambda: db.fetch(*loc, cols))
    f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
    for c in f.columns:
        if c != "date":
            f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.set_index("date").sort_index()
    if f["mktrf"].abs().mean() > 0.02:      # delivered in percent
        f = f / 100.0
    return f


def matched_difference(df: pd.DataFrame, flagged_ids, k=5, reps=2000, seed=0, outcome="bhar_252_w") -> dict:
    """EXPLORATORY. Mean 12-month BHAR of flagged events minus that of k matched unflagged departures
    (same announcement year and role; nearest on log market cap and prior-year return). Bootstrap CI over events."""
    d = df.dropna(subset=[outcome, "log_mcap", "pre_ret_w"]).copy()
    d["flag"] = d.event_id.isin(flagged_ids)
    sd = d[["log_mcap", "pre_ret_w"]].std().replace(0, 1).values
    diffs = []
    for _, r in d[d.flag].iterrows():
        c = d[(~d.flag) & (d.year == r.year) & (d.role == r.role)]
        if len(c) < k:
            c = d[(~d.flag) & (d.year == r.year)]
        if len(c) < k:
            continue
        dist = (((c[["log_mcap", "pre_ret_w"]].values - np.array([r.log_mcap, r.pre_ret_w])) / sd) ** 2).sum(axis=1)
        nn = c.iloc[np.argsort(dist)[:k]]
        diffs.append(float(r[outcome] - nn[outcome].mean()))
    diffs = np.array(diffs)
    if len(diffs) < 5:
        return {"n": int(len(diffs))}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diffs, len(diffs)).mean() for _ in range(reps)]
    return {"n": int(len(diffs)), "mean_diff": float(diffs.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "median_diff": float(np.median(diffs))}


# --------------------------------------------------------------------------------------
# Stage 2 helpers: fast alpha, within-year label permutation
# --------------------------------------------------------------------------------------
def alpha_fast(port, cnt, fac: pd.DataFrame, factors=FACTORS_FF4, min_days=250) -> float:
    """Annualised alpha from the same weighted regression as alpha_test (point estimate only, no HAC)."""
    y = port - fac["rf"].values
    Xf = fac[factors].values
    ok = (cnt >= 1) & np.isfinite(y) & np.isfinite(Xf).all(axis=1)
    if ok.sum() < min_days:
        return np.nan
    w = np.sqrt(cnt[ok])
    X = np.column_stack([np.ones(ok.sum()), Xf[ok]]) * w[:, None]
    beta = np.linalg.lstsq(X, y[ok] * w, rcond=None)[0]
    return float(beta[0] * 252)


def split_permutation(windows: dict, ids, flags, years, fac, T, factors=FACTORS_FF4, reps=5000, seed=0):
    """Does the group with flags==True differ from the rest? Statistic: alpha(True) - alpha(False), annualised.
    Null: permute the flags among events within the same announcement year."""
    ids = np.asarray(list(ids))
    flags = np.asarray(flags, dtype=bool)
    years = np.asarray(years)
    def stat(f):
        a, ca = portfolio(windows, ids[f], T)
        b, cb = portfolio(windows, ids[~f], T)
        return alpha_fast(a, ca, fac, factors) - alpha_fast(b, cb, fac, factors), alpha_fast(a, ca, fac, factors), alpha_fast(b, cb, fac, factors)
    obs, a_true, a_false = stat(flags)
    rng = np.random.default_rng(seed)
    groups = [np.where(years == y)[0] for y in np.unique(years)]
    null = np.empty(reps)
    for r in range(reps):
        f = flags.copy()
        for g in groups:
            f[g] = rng.permutation(flags[g])
        null[r] = stat(f)[0]
    return obs, a_true, a_false, null


# --------------------------------------------------------------------------------------
# Stage 5 helpers: placebo contrast between two groups
# --------------------------------------------------------------------------------------
def placebo_draws(windows: dict, flagged: pd.DataFrame, pool: pd.DataFrame, fac: pd.DataFrame, H: int, *,
                  strata_col="year", id_col="event_id", reps=1000, seed=0, factors=FACTORS_FF4) -> np.ndarray:
    """Alphas of random same-size portfolios drawn from `pool` (non-flagged events), matching the flagged set's count
    per `strata_col`. `H` is the calendar length (the `T` argument of portfolio)."""
    rng = np.random.default_rng(seed)
    need = flagged.groupby(strata_col).size()
    cand = pool[pool[id_col].isin(windows.keys())]
    by_s = {s: g[id_col].values for s, g in cand.groupby(strata_col)}
    out = np.empty(reps)
    for r in range(reps):
        pick = []
        for s, k in need.items():
            arr = by_s.get(s)
            if arr is None or len(arr) == 0:
                continue
            pick.extend(rng.choice(arr, size=min(k, len(arr)), replace=False))
        port, cnt = portfolio(windows, pick, H)
        out[r] = alpha_fast(port, cnt, fac, factors)
    return out


def fpc_factor(n_flagged: int, n_pool: int) -> float:
    """Standard-deviation inflation for a placebo null drawn from a finite pool. A placebo portfolio varies only around the
    pool's own mean, while a flagged portfolio also carries the pool's sampling error, so the draws understate the spread
    of (flagged - pool mean) by sqrt((N + n) / (N - n)). Negligible for large pools (the earlier stages), up to 1.22 when
    the pool is only 5x the flagged count."""
    return float(np.sqrt((n_pool + n_flagged) / (n_pool - n_flagged))) if n_pool > n_flagged > 0 else float("nan")


def placebo_difference(windows, flagged_low, flagged_high, pool_low, pool_high, fac, H, *, reps=1000, seed=0,
                       strata_col="year", id_col="event_id", factors=FACTORS_FF4) -> np.ndarray:
    """Null for 'the excess alpha of flagged firms is the same in the low and the high group'. Each replication draws a
    same-size, per-stratum-matched portfolio from each group's own non-flagged pool; the statistic is the draw alpha
    minus that group's mean placebo alpha (scaled by fpc_factor), low minus high. Centring removes baseline alpha
    differences between groups and any common shift, so it is the null of EQUAL effects, not of two zero effects."""
    a_lo = placebo_draws(windows, flagged_low, pool_low, fac, H, strata_col=strata_col, id_col=id_col, reps=reps, seed=seed, factors=factors)
    a_hi = placebo_draws(windows, flagged_high, pool_high, fac, H, strata_col=strata_col, id_col=id_col, reps=reps, seed=seed + 1, factors=factors)
    f_lo = fpc_factor(len(flagged_low), int(pool_low[id_col].isin(windows.keys()).sum()))
    f_hi = fpc_factor(len(flagged_high), int(pool_high[id_col].isin(windows.keys()).sum()))
    return f_lo * (a_lo - np.nanmean(a_lo)) - f_hi * (a_hi - np.nanmean(a_hi))
