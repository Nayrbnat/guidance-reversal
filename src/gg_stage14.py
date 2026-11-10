"""Stage 14 (docs/STAGE14_SPEC.md): do valuation and turnaround explain the flagged firms' returns? Valuation variables with a publication lag (no look-ahead),
a characteristic-matched comparison, an event-level regression with clustered errors, and a noisy outside-successor proxy. Exploratory."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_value_vars(ev: pd.DataFrame, funda: pd.DataFrame, lag_days: int = 90) -> pd.DataFrame:
    """Add bm, log_bm, bm_missing, ep, roa, droa, neg_earn from Compustat annual data (gvkey, datadate, ceq, at, ib; ceq, at, ib in $ millions; ev.mcap in dollars).
    Each event uses the last annual report whose fiscal year end is at least `lag_days` before the announcement (dep_ann). Row order and columns are kept."""
    f = funda.copy()
    f["gvkey"] = f["gvkey"].astype(str).str.zfill(6)
    f["datadate"] = pd.to_datetime(f["datadate"]).astype("datetime64[ns]")
    f = f.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    f["roa"] = f["ib"] / f["at"].where(f["at"] > 0)
    f["roa_prev"] = f.groupby("gvkey")["roa"].shift(1)
    f["avail"] = f["datadate"] + pd.Timedelta(days=lag_days)
    e = ev.copy()
    e["_i"] = np.arange(len(e))
    e["gvkey"] = e["gvkey"].astype(str).str.zfill(6)
    e["_d"] = pd.to_datetime(e["dep_ann"]).astype("datetime64[ns]")
    m = pd.merge_asof(e.sort_values("_d"), f[["gvkey", "avail", "ceq", "ib", "roa", "roa_prev"]].sort_values("avail"), left_on="_d", right_on="avail", by="gvkey", direction="backward")
    m = m.sort_values("_i").reset_index(drop=True)
    out = ev.copy().reset_index(drop=True)
    mc = pd.to_numeric(m["mcap"], errors="coerce").where(lambda s: s > 0)
    out["bm"] = m["ceq"] * 1e6 / mc
    out["log_bm"] = np.log(out["bm"].where(m["ceq"] > 0))
    out["bm_missing"] = (m["ceq"] <= 0).astype(int)
    out["ep"] = m["ib"] * 1e6 / mc
    out["roa"] = m["roa"]
    out["droa"] = m["roa"] - m["roa_prev"]
    out["neg_earn"] = (m["ib"] < 0).astype(float).where(m["ib"].notna())
    out.index = ev.index
    return out


def matched_diff(d: pd.DataFrame, y: str, flag_ids: list, covs: list, k: int = 5, reps: int = 2000, seed: int = 0, id_col: str = "event_id") -> dict:
    """Mean of (flagged outcome minus the mean outcome of its k nearest unflagged events announced in the SAME year with the SAME role), nearest in Euclidean
    distance on the standardised covariates. Flagged events missing a covariate or the outcome are dropped (n_dropped). 95% interval: bootstrap over flagged events."""
    flag = set(flag_ids)
    cols = [id_col, "year", "role", y] + list(covs)
    x = d[cols].copy()
    ok = x.dropna(subset=[y] + list(covs))
    sd = ok[list(covs)].std().replace(0, 1.0)
    z = (ok[list(covs)] - ok[list(covs)].mean()) / sd
    isf = ok[id_col].isin(flag).to_numpy()
    n_dropped = int(len(flag & set(d[id_col])) - int(isf.sum()))
    diffs = []
    for i in np.where(isf)[0]:
        pool = np.where(~isf & (ok["year"].to_numpy() == ok["year"].iloc[i]) & (ok["role"].to_numpy() == ok["role"].iloc[i]))[0]
        if len(pool) == 0:
            continue
        dist = ((z.iloc[pool].to_numpy() - z.iloc[i].to_numpy()) ** 2).sum(axis=1)
        near = pool[np.argsort(dist, kind="stable")[: min(k, len(pool))]]
        diffs.append(float(ok[y].iloc[i] - ok[y].iloc[near].mean()))
    if not diffs:
        return {"diff": float("nan"), "ci": [float("nan"), float("nan")], "n": 0, "n_dropped": n_dropped, "k": k}
    diffs = np.array(diffs)
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(reps, len(diffs)), replace=True).mean(axis=1)
    return {"diff": float(diffs.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "n": int(len(diffs)), "n_dropped": n_dropped, "k": k}


def flag_regression(d: pd.DataFrame, y: str, flag: str, controls: list, fe: str | None = None, cluster: str | None = None) -> dict:
    """OLS of y on the flag, controls and fixed-effect dummies for `fe`, errors clustered by `cluster`; complete cases only. Returns coef, se, p_two, n, clusters."""
    import statsmodels.api as sm
    cols = [y, flag] + list(controls) + ([fe] if fe else []) + ([cluster] if cluster else [])
    z = d[list(dict.fromkeys(cols))].dropna().reset_index(drop=True)
    X = z[[flag] + list(controls)].astype(float)
    if fe:
        X = pd.concat([X, pd.get_dummies(z[fe], prefix="fe", drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    if cluster:
        fit = sm.OLS(z[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(z[cluster])[0]}, use_t=True)
    else:
        fit = sm.OLS(z[y].astype(float), X).fit(cov_type="HC1")
    return {"coef": float(fit.params[flag]), "se": float(fit.bse[flag]), "p_two": float(fit.pvalues[flag]), "n": int(len(z)),
            "clusters": int(z[cluster].nunique()) if cluster else None}


def successor_outside(ev: pd.DataFrame, ec: pd.DataFrame) -> pd.DataFrame:
    """For CEO departures: the successor (another executive listed as CEO at the firm in the departure year or the next) and an outside-hire PROXY (the
    successor has no ExecuComp record at the firm before the year they first appear as CEO; ExecuComp covers only the top five executives, so this is noisy).
    Columns added: successor, outside (NaN for non-CEO departures or when no successor is found)."""
    out = ev.copy()
    out["successor"] = pd.Series([None] * len(out), index=out.index, dtype="object")      # executive IDs may be strings or numbers
    out["outside"] = np.nan
    ec = ec.assign(gvkey=ec["gvkey"].astype(str).str.zfill(6))
    for i, r in ev.iterrows():
        if r.get("role") != "CEO":
            continue
        g = ec[ec["gvkey"] == str(r["gvkey"]).zfill(6)]
        yr = pd.Timestamp(r["dep_ann"]).year
        c = g[g["year"].between(yr, yr + 1) & (g["ceoann"] == "CEO") & (g["execid"] != r["execid"])].sort_values("year")
        if c.empty:
            continue
        s = c.iloc[0]["execid"]
        ceo_year = int(c[c["execid"] == s]["year"].min())
        first_year = int(g[g["execid"] == s]["year"].min())
        out.loc[i, "successor"] = s
        out.loc[i, "outside"] = int(first_year >= ceo_year)
    return out
