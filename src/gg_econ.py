"""
gg_econ.py -- fixed-effects regression with firm-clustered standard errors (Stage 3).

fe_ols(): absorbs firm fixed effects by within-firm demeaning, includes year dummies explicitly, and uses the
cluster-robust sandwich with the usual small-sample factor  G/(G-1) * (N-1)/(N-K), where K counts the regressors
and year dummies but NOT the firm effects (they are nested inside the clusters, as in Stata's reghdfe).
Inference uses t(G-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as st


def _demean(a: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    cnt = np.bincount(codes, minlength=n_groups).astype(float)
    if a.ndim == 1:
        return a - (np.bincount(codes, weights=a, minlength=n_groups) / cnt)[codes]
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(codes, weights=a[:, j], minlength=n_groups) / cnt)[codes]
    return out


def fe_ols(df: pd.DataFrame, y: str, xs: list, firm: str = "firm", year: str | None = "year",
           cluster: str | None = None, firm_fe: bool = True, cluster2: str | None = None) -> dict:
    """OLS of df[y] on df[xs] (+ firm FE, + dummies for `year`, which may be any time category such as year-quarter).
    Standard errors are clustered by `cluster` (default: firm); if `cluster2` is given they are two-way clustered
    (Cameron-Gelbach-Miller: V = V1 + V2 - V12, with eigenvalue clipping), and t(min(G1, G2) - 1) is used."""
    cols = [y] + list(xs) + [firm] + ([year] if year else []) + ([cluster] if cluster and cluster not in (firm, year) else [])         + ([cluster2] if cluster2 and cluster2 not in (firm, year, cluster) else [])
    d = df[list(dict.fromkeys(cols))].dropna().reset_index(drop=True)
    firm_codes, firm_u = pd.factorize(d[firm])
    clus = firm_codes if (cluster is None or cluster == firm) else pd.factorize(d[cluster])[0]
    X = d[list(xs)].to_numpy(float)
    names = list(xs)
    if year:
        yd = pd.get_dummies(d[year], prefix="yr", drop_first=True).to_numpy(float)
        X = np.hstack([X, yd])
    yv = d[y].to_numpy(float)
    if firm_fe:
        X = _demean(X, firm_codes, len(firm_u))
        yv = _demean(yv, firm_codes, len(firm_u))
    else:
        X = np.hstack([np.ones((len(d), 1)), X]); names = ["const"] + names
    n, k = X.shape
    XtX = X.T @ X
    keep = np.where(np.diag(XtX) > 1e-12)[0]       # drop columns that vanish after demeaning (no within variation)
    Xk = X[:, keep]
    inv = np.linalg.pinv(Xk.T @ Xk)
    beta = inv @ (Xk.T @ yv)
    u = yv - Xk @ beta
    S = Xk * u[:, None]

    def _v(codes):
        Gc = int(np.max(codes)) + 1
        grp = np.zeros((Gc, Xk.shape[1]))
        np.add.at(grp, codes, S)
        return Gc / (Gc - 1) * (n - 1) / (n - len(keep)) * inv @ (grp.T @ grp) @ inv, Gc

    V, G = _v(clus)
    if cluster2:
        c2 = pd.factorize(d[cluster2])[0]
        V2, G2 = _v(c2)
        V12, _ = _v(pd.factorize(pd.Series(list(zip(clus, c2))))[0])
        V = V + V2 - V12
        w, Q = np.linalg.eigh((V + V.T) / 2)
        V = Q @ np.diag(np.clip(w, 0, None)) @ Q.T
        G = min(G, G2)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    out = {"n": int(n), "n_clusters": G, "k": int(len(keep)), "r2_within": float(1 - (u @ u) / (yv @ yv)) if firm_fe else None}
    allnames = names + [f"yr{j}" for j in range(X.shape[1] - len(names))]
    for pos, j in enumerate(keep):
        if j < len(xs) + (0 if firm_fe else 1):
            t = beta[pos] / se[pos] if se[pos] > 0 else np.nan
            out[allnames[j]] = {"coef": float(beta[pos]), "se": float(se[pos]), "t": float(t),
                                "p_two": float(2 * st.t.sf(abs(t), G - 1)) if np.isfinite(t) else np.nan}
    return out


def one_sided(res: dict, name: str, sign: int) -> float:
    """One-sided p for H1: coef has the given sign (+1 / -1), from t(G-1)."""
    r = res[name]
    t = r["t"] * sign
    return float(st.t.sf(t, res["n_clusters"] - 1))


def min_detectable(res: dict, name: str) -> float:
    """Coefficient size detected with 80% power at one-sided 5% (about 2.49 x the standard error)."""
    return float((st.norm.ppf(0.95) + st.norm.ppf(0.80)) * res[name]["se"])


def logit_cluster(df: pd.DataFrame, y: str, xs: list, cluster: str, year: str | None = None) -> dict:
    """Pooled logit (+ year dummies) with firm-clustered SEs (robustness R6)."""
    import statsmodels.api as sm
    cols = [y] + xs + [cluster] + ([year] if year else [])
    d = df[cols].dropna().reset_index(drop=True)
    X = d[xs].astype(float)
    if year:
        X = pd.concat([X, pd.get_dummies(d[year], prefix="yr", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    m = sm.Logit(d[y].astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    out = {"n": int(len(d)), "n_clusters": int(d[cluster].nunique())}
    for n in xs:
        out[n] = {"coef": float(m.params[n]), "se": float(m.bse[n]), "t": float(m.tvalues[n]), "p_two": float(m.pvalues[n])}
    return out
