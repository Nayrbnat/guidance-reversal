"""
gg_stage12.py -- Stage 12 (docs/STAGE12_SPEC.md): a rolling-window out-of-sample test with a machine-learning model.

  market_features   pre-event size, return and volatility, measured before day +2 of the return window
  rolling_oos       train on years t-6..t-2 (embargo year t-1), predict year t, roll forward
  year_aucs         within-year AUC (the primary metric)
  perm_delta_auc    permutation test that shuffles the Guidance-Reversal feature block within year and re-runs the whole procedure
  boot_delta_auc    bootstrap interval for the AUC difference, resampling announcement quarters (predictions fixed)
  rolling_gap       rolling five-year windows of a flagged-minus-comparator gap
  cochran_q         heterogeneity of the gaps over non-overlapping blocks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------------------------------------ features
def market_features(df: pd.DataFrame, crsp: pd.DataFrame, date_col: str = "dep_ann_final", lag: int = 2, window: int = 252) -> pd.DataFrame:
    """Log market cap on trading day (day0 - lag), and the compounded return and annualised volatility over the `window` trading
    days ending on day (day0 - lag). day0 is the first trading day on or after df[date_col]. Everything ends before day +2 of the
    return window, so no feature uses the outcome period."""
    out = df.copy()
    c = crsp[["permno", "date", "ret", "prc", "shrout"]].dropna(subset=["date"]).sort_values(["permno", "date"])
    by = {int(p): (g.date.values.astype("datetime64[ns]"), g.ret.values.astype(float), np.abs(g.prc.values.astype(float)), g.shrout.values.astype(float))
          for p, g in c.groupby("permno")}
    lm, r12, vol = np.full(len(out), np.nan), np.full(len(out), np.nan), np.full(len(out), np.nan)
    for k, (p, t) in enumerate(zip(out["permno"].astype("int64").values, pd.to_datetime(out[date_col]).values)):
        s = by.get(int(p))
        if s is None:
            continue
        d, r, px, sh = s
        i0 = int(np.searchsorted(d, t))
        e = i0 - lag
        if e < window or e >= len(d):
            continue
        rr = r[e - window + 1:e + 1]
        ok = np.isfinite(rr)
        if ok.sum() < 0.8 * window:
            continue
        r12[k] = np.prod(1 + rr[ok]) - 1
        vol[k] = rr[ok].std(ddof=1) * np.sqrt(252)
        if np.isfinite(px[e]) and np.isfinite(sh[e]) and px[e] > 0 and sh[e] > 0:
            lm[k] = np.log(px[e] * sh[e] * 1000.0)
    out["f_lmcap"], out["f_ret12"], out["f_vol"] = lm, r12, vol
    return out


# ------------------------------------------------------------------------------------------------ models
def make_model(kind: str):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, min_samples_leaf=40,
                                              l2_regularization=1.0, early_stopping=False, random_state=0)
    if kind == "xgb":                                   # Stage 12b (docs/STAGE12B_SPEC.md)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=0, verbosity=0)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="relu", solver="adam", alpha=0.1, learning_rate_init=1e-3,
                             max_iter=200, early_stopping=False, random_state=0)
    if kind == "svm_lin":
        from sklearn.svm import LinearSVC
        return LinearSVC(C=0.1, loss="squared_hinge", dual=False, max_iter=5000)
    if kind == "svm_poly":                              # polynomial kernel, never radial (spec Section 3)
        from sklearn.svm import SVC
        return SVC(kernel="poly", degree=2, C=0.1, coef0=1.0, gamma="scale", cache_size=500)
    return LogisticRegression(C=1.0, max_iter=2000)


def score(m, X: np.ndarray) -> np.ndarray:
    """Ranking score for the AUC: probability when the model has one, otherwise the signed margin (SVMs)."""
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def _prep(Xtr: np.ndarray, Xte: np.ndarray, scale: bool):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    a, b = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xte), med, Xte)
    if scale:
        mu, sd = a.mean(0), a.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        a, b = (a - mu) / sd, (b - mu) / sd
    return a, b


def split_years(t: int, window: int = 5, embargo: int = 1, expanding: bool = False, first: int = 2004) -> range:
    """Training announcement years for test year t: t-embargo-window .. t-embargo-1 (embargo year t-1 left out so every training
    label, a 12-month window, is complete before year t starts)."""
    hi = t - embargo - 1
    lo = first if expanding else hi - window + 1
    return range(max(lo, first), hi + 1)


def rolling_oos(df: pd.DataFrame, cols: list, y: str, kind: str, test_years, *, window: int = 5, embargo: int = 1,
                expanding: bool = False, year_col: str = "year") -> pd.DataFrame:
    """One prediction per test-year event: fit on the training years, never on the test year. Returns index, year, p."""
    X = df[cols].to_numpy(float)
    yy = df[y].to_numpy(float)
    yrs = df[year_col].to_numpy()
    rows = []
    for t in test_years:
        tr = np.isin(yrs, list(split_years(t, window, embargo, expanding)))
        te = yrs == t
        if tr.sum() < 100 or te.sum() == 0 or len(np.unique(yy[tr])) < 2:
            continue
        a, b = _prep(X[tr], X[te], scale=(kind != "hgb"))
        m = make_model(kind).fit(a, yy[tr])
        rows.append(pd.DataFrame({"idx": np.where(te)[0], "year": t, "p": score(m, b), "y": yy[te]}))
    return pd.concat(rows, ignore_index=True)


def year_aucs(pred: pd.DataFrame, min_pos: int = 5) -> pd.Series:
    """AUC within each test year (years with at least min_pos positives and min_pos negatives)."""
    out = {}
    for t, g in pred.groupby("year"):
        pos = int(g.y.sum())
        if pos >= min_pos and len(g) - pos >= min_pos:
            out[t] = roc_auc_score(g.y, g.p)
    return pd.Series(out, dtype=float)


def mean_auc(pred: pd.DataFrame, min_pos: int = 5) -> float:
    a = year_aucs(pred, min_pos)
    return float(a.mean()) if len(a) else float("nan")


# ------------------------------------------------------------------------------------------------ inference
def perm_delta_auc(df: pd.DataFrame, base_cols: list, r_cols: list, y: str, kind: str, test_years, reps: int, seed: int = 0, **kw) -> dict:
    """Out-of-sample mean within-year AUC of base+R minus base alone, with a permutation p-value: the R rows are shuffled across
    events within each announcement year (breaking R from the outcome and from the baseline) and the whole rolling procedure re-run."""
    p_b = rolling_oos(df, base_cols, y, kind, test_years, **kw)
    p_br = rolling_oos(df, base_cols + r_cols, y, kind, test_years, **kw)
    auc_b, auc_br = mean_auc(p_b), mean_auc(p_br)
    obs = auc_br - auc_b
    rng = np.random.default_rng(seed)
    dfp = df.copy()
    groups = [np.where(df["year"].to_numpy() == t)[0] for t in np.unique(df["year"].to_numpy())]
    Rv = df[r_cols].to_numpy(float)
    null = np.empty(reps)
    for r in range(reps):
        R = Rv.copy()
        for g in groups:
            R[g] = Rv[rng.permutation(g)]
        dfp[r_cols] = R
        null[r] = mean_auc(rolling_oos(dfp, base_cols + r_cols, y, kind, test_years, **kw)) - auc_b
    return {"auc_base": auc_b, "auc_base_plus_r": auc_br, "delta": float(obs), "reps": reps,
            "p_one_sided": float((1 + (null >= obs).sum()) / (1 + reps)), "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "by_year": {int(t): float(a) for t, a in (year_aucs(p_br) - year_aucs(p_b)).dropna().items()},
            "pred_base": p_b, "pred_br": p_br}


def boot_delta_auc(pred_base: pd.DataFrame, pred_br: pd.DataFrame, cluster: np.ndarray, reps: int = 1000, seed: int = 0) -> list:
    """95% interval for the AUC difference, resampling clusters (announcement quarters) of events; predictions fixed.
    `cluster` is indexed by the `idx` column of the predictions."""
    d = pred_base.rename(columns={"p": "p0"}).merge(pred_br[["idx", "p"]].rename(columns={"p": "p1"}), on="idx")
    d["q"] = np.asarray(cluster)[d.idx.values]
    uq = d.q.unique()
    by = {q: g for q, g in d.groupby("q")}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        b = pd.concat([by[q] for q in rng.choice(uq, len(uq), replace=True)])
        a0, a1 = [], []
        for t, g in b.groupby("year"):
            pos = int(g.y.sum())
            if pos >= 5 and len(g) - pos >= 5:
                a0.append(roc_auc_score(g.y, g.p0))
                a1.append(roc_auc_score(g.y, g.p1))
        if a0:
            out.append(np.mean(a1) - np.mean(a0))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


# ------------------------------------------------------------------------------------------------ P12b
def gap_boot(y: np.ndarray, g: np.ndarray, cluster: np.ndarray, reps: int = 1000, seed: int = 0) -> dict:
    """Mean of y in group 1 minus group 0 with a bootstrap over clusters."""
    y, g, cl = np.asarray(y, float), np.asarray(g, int), np.asarray(cluster)
    if (g == 1).sum() < 5 or (g == 0).sum() < 5:
        return {"gap": np.nan, "se": np.nan, "n1": int((g == 1).sum()), "n0": int((g == 0).sum())}
    est = y[g == 1].mean() - y[g == 0].mean()
    uc, inv = np.unique(cl, return_inverse=True)
    S1, N1 = np.bincount(inv, weights=y * (g == 1), minlength=len(uc)), np.bincount(inv, weights=(g == 1) * 1.0, minlength=len(uc))
    S0, N0 = np.bincount(inv, weights=y * (g == 0), minlength=len(uc)), np.bincount(inv, weights=(g == 0) * 1.0, minlength=len(uc))
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, len(uc), len(uc)), minlength=len(uc)).astype(float)
        if (w * N1).sum() > 0 and (w * N0).sum() > 0:
            b.append((w * S1).sum() / (w * N1).sum() - (w * S0).sum() / (w * N0).sum())
    return {"gap": float(est), "se": float(np.std(b, ddof=1)), "n1": int((g == 1).sum()), "n0": int((g == 0).sum())}


def cochran_q(est, se) -> dict:
    """Heterogeneity of independent estimates: Q = sum w_k (est_k - weighted mean)^2, df = k - 1."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    w = 1 / se ** 2
    m = np.sum(w * est) / np.sum(w)
    q = float(np.sum(w * (est - m) ** 2))
    return {"Q": q, "df": int(len(est) - 1), "p": float(1 - stats.chi2.cdf(q, len(est) - 1)), "pooled": float(m), "k": int(len(est))}
