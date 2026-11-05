"""Stage 6: H5 at the level of the individual insider sale (spec: docs/STAGE6_SPEC.md, written before any outcome was computed).

After a CEO sells, is the firm's subsequent abnormal return lower (the sale "avoided a loss") where transparency T is low, and is that
concentrated in guide-sell-leave sales? Unit: a CEO sale-day. Outcome Y = -100 x CAR(+1,+h), the four-factor cumulative abnormal return
after the sale with betas from the 252 trading days BEFORE it."""
from __future__ import annotations

import numpy as np
import pandas as pd

import gg_econ as E
import gg_stage3 as S3

FACTORS = ["mktrf", "smb", "hml", "umd"]
CONTROLS = ["lmcap", "ret12", "lval", "ltenure", "age"]


# ---------------------------------------------------------------- sale-days and their treatment flags
def sale_days(sales: pd.DataFrame, ep: pd.DataFrame, permno_map: dict, *, min_value: float = 100_000.0,
              start: str = "2004-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """One row per CEO episode and day with its sales summed; keeps value >= min_value inside [start, end]. X = 1 if the CEO leaves within
    365 days AFTER the sale (ex-post treatment, as in Stage 3 P2). A CEO who dies within 365 days of the sale is dropped."""
    s = sales.copy()
    s["trandate"] = pd.to_datetime(s["trandate"])
    s = s[(s["trandate"] >= pd.Timestamp(start)) & (s["trandate"] <= pd.Timestamp(end))]
    d = s.groupby(["ep_id", "trandate"], as_index=False)["val"].sum()
    d = d[d["val"] >= min_value].reset_index(drop=True)
    e = ep.set_index("ep_id")
    left = pd.to_datetime(e["leftofc"].reindex(d["ep_id"]).values)
    gap = pd.Series((left - d["trandate"].values).days if len(d) else [], index=d.index, dtype=float)
    within = (gap > 0) & (gap <= 365)
    dead = within & pd.Series(e["deceased"].reindex(d["ep_id"]).fillna(False).astype(bool).values, index=d.index)
    d["X"] = within.astype(float)
    d["gvkey"] = e["gvkey"].reindex(d["ep_id"]).values
    d["permno"] = d["ep_id"].map(permno_map)
    d = d[~dead & d["permno"].notna()].reset_index(drop=True)
    d["permno"] = d["permno"].astype("int64")
    return d


def attach_guidance(df: pd.DataFrame, g: pd.DataFrame, max_days: int = 180) -> pd.DataFrame:
    """G = 1 if the firm issued guidance in the max_days strictly BEFORE the sale date; O = 1 if the latest such guidance midpoint was above
    consensus (opt > 0); days_since_guid. Row order and index of df are preserved."""
    x = df[["permno", "trandate"]].copy()
    x["_i"] = np.arange(len(x))
    x["permno"] = x["permno"].astype("int64")
    x["d0"] = (pd.to_datetime(x["trandate"]) - pd.Timedelta(days=1)).astype("datetime64[ns]")
    gg = g[["permno", "t", "opt"]].copy()
    gg["permno"] = gg["permno"].astype("int64")
    gg["t"] = pd.to_datetime(gg["t"]).astype("datetime64[ns]")
    m = pd.merge_asof(x.sort_values("d0"), gg.sort_values("t"), left_on="d0", right_on="t", by="permno", direction="backward").sort_values("_i")
    days = (pd.to_datetime(x["trandate"]).values - m["t"].values) / np.timedelta64(1, "D")
    ok = np.isfinite(days) & (days <= max_days)
    out = df.copy()
    out["days_since_guid"] = np.where(ok, days, np.nan)
    out["G"] = ok.astype(float)
    out["O"] = (ok & (m["opt"].values > 0)).astype(float)
    return out


# ---------------------------------------------------------------- cumulative abnormal returns with pre-sale betas
def car_table(sales: pd.DataFrame, crsp: pd.DataFrame, fac: pd.DataFrame, horizons=(21, 63, 126, 252), beta_window: int = 252,
              min_pre: int = 120, min_frac: float = 0.5) -> pd.DataFrame:
    """For each sale (permno, trandate) add car{h}: sum over the h trading days after the sale of r - rf - beta'F, with beta from an OLS (with
    intercept) of r - rf on the four factors over the beta_window trading days ending the day BEFORE the sale (at least min_pre
    observations). The window starts the next trading day (a non-trading-day sale starts on the next trading day). A window with
    fewer than min_frac x h valid days (sale near the end of the data, early delisting) is NaN; a window cut short by a delisting sums
    the days that exist, delisting return included."""
    cal = pd.DatetimeIndex(fac.index)
    F = fac[FACTORS].to_numpy(float)
    rf = fac["rf"].to_numpy(float)
    permnos = sorted(set(sales["permno"].astype("int64")))
    rets = {}
    c = crsp[crsp["permno"].isin(permnos)]
    for p, grp in c.groupby("permno"):
        rets[int(p)] = grp.drop_duplicates("date").set_index("date")["ret"].reindex(cal).to_numpy(float)
    out = sales.copy()
    cols = {h: np.full(len(out), np.nan) for h in horizons}
    dates = pd.DatetimeIndex(pd.to_datetime(sales["trandate"]))
    for k, (p, t) in enumerate(zip(sales["permno"].astype("int64").values, dates)):
        r = rets.get(int(p))
        if r is None:
            continue
        i0 = int(cal.searchsorted(t, side="left"))
        if i0 >= len(cal):
            continue
        start = i0 + 1 if cal[i0] == t else i0
        lo = max(0, i0 - beta_window)
        pre = np.arange(lo, i0)
        pre = pre[np.isfinite(r[pre])]
        if len(pre) < min_pre:
            continue
        X = np.column_stack([np.ones(len(pre)), F[pre]])
        beta = np.linalg.lstsq(X, r[pre] - rf[pre], rcond=None)[0][1:]
        for h in horizons:
            idx = np.arange(start, min(start + h, len(cal)))
            idx = idx[np.isfinite(r[idx])] if len(idx) else idx
            if len(idx) >= min_frac * h:
                cols[h][k] = float(np.sum(r[idx] - rf[idx] - F[idx] @ beta))
    for h in horizons:
        out[f"car{h}"] = cols[h]
    return out


# ---------------------------------------------------------------- regressions
def run_primary(d: pd.DataFrame, y: str, *, z: str = "Z", g: str = "G", controls=CONTROLS, firm: str = "gvkey", month: str = "ym",
                moderators=()) -> dict:
    """The Stage 6 primary regression: y on G, X, GX, Z and Z interacted with each, plus controls, firm and year-month fixed effects, standard
    errors clustered by firm and month. Key ZGX; H5 predicts a NEGATIVE coefficient, so the one-sided p is taken for sign -1."""
    d = d.copy()
    gv, zv = d[g].astype(float), d[z].astype(float)
    d["G"], d["Z"] = gv, zv                                  # the chosen guidance flag and transparency measure take the standard names
    d["GX"] = gv * d["X"]
    d["ZG"], d["ZX"], d["ZGX"] = zv * gv, zv * d["X"], zv * gv * d["X"]
    xs = ["G", "X", "GX", "Z", "ZG", "ZX", "ZGX"] + [c for c in controls if c in d.columns]
    for m in moderators:                                      # each moderator enters as M, M x G, M x X and M x G x X
        mv = d[m].astype(float)
        d[m + "G"], d[m + "X"], d[m + "GX"] = mv * gv, mv * d["X"], mv * gv * d["X"]
        xs += [m, m + "G", m + "X", m + "GX"]
    res = E.fe_ols(d, y, xs, firm=firm, year=month, cluster=firm, cluster2=month)
    s = S3.summarise(res, "ZGX", -1)
    s["n_GX"] = int(d.dropna(subset=[y])["GX"].sum())
    return s


def fps_industry(sic) -> pd.Series:
    """1 for the litigation-prone industries of Francis, Philbrick and Schipper (1994) as commonly implemented: SIC 2833-2836, 3570-3577,
    3600-3674, 7370-7374 and 5200-5961; 0 otherwise (including a missing SIC)."""
    x = pd.to_numeric(pd.Series(sic), errors="coerce")
    bands = [(2833, 2836), (3570, 3577), (3600, 3674), (7370, 7374), (5200, 5961)]
    out = pd.Series(0, index=x.index)
    for lo, hi in bands:
        out = out.where(~((x >= lo) & (x <= hi)), 1)
    return out.astype(int)


def pre_sale_vol(sales: pd.DataFrame, crsp: pd.DataFrame, fac: pd.DataFrame, beta_window: int = 252, min_obs: int = 120) -> pd.DataFrame:
    """Add vol: the standard deviation of daily returns over the beta_window trading days ending the day BEFORE the sale (NaN with fewer
    than min_obs observations). Nothing on or after the sale date enters."""
    cal = pd.DatetimeIndex(fac.index)
    permnos = sorted(set(sales["permno"].astype("int64")))
    rets = {}
    c = crsp[crsp["permno"].isin(permnos)]
    for p, grp in c.groupby("permno"):
        rets[int(p)] = grp.drop_duplicates("date").set_index("date")["ret"].reindex(cal).to_numpy(float)
    vol = np.full(len(sales), np.nan)
    for k, (p, t) in enumerate(zip(sales["permno"].astype("int64").values, pd.DatetimeIndex(pd.to_datetime(sales["trandate"])))):
        r = rets.get(int(p))
        if r is None:
            continue
        i0 = int(cal.searchsorted(t, side="left"))
        x = r[max(0, i0 - beta_window):i0]
        x = x[np.isfinite(x)]
        if len(x) >= min_obs:
            vol[k] = float(np.std(x, ddof=1))
    out = sales.copy()
    out["vol"] = vol
    return out


def power_sim(d: pd.DataFrame, y: str, deltas=(0.0, 1.0, 2.0, 4.0), reps: int = 100, seed: int = 0, **kw) -> dict:
    """Share of firm-cluster bootstrap replications in which the primary test rejects (coefficient < 0 and one-sided p < 0.05) when an effect
    of -delta x Z is planted into y for guide-sell-leave sales (G x X = 1). delta = 0 gives the false-positive rate."""
    rng = np.random.default_rng(seed)
    firms = d["gvkey"].unique()
    groups = {f: d[d["gvkey"] == f] for f in firms}
    out = {}
    for delta in deltas:
        hits = 0
        for k in range(reps):
            pick = rng.choice(len(firms), len(firms))
            parts = [groups[firms[j]].assign(gvkey=f"{i}_{firms[j]}") for i, j in enumerate(pick)]
            b = pd.concat(parts, ignore_index=True)
            b[y] = b[y] - delta * b["Z"] * b["G"] * b["X"]
            r = run_primary(b, y, **kw)
            hits += int(r["ZGX"]["coef"] < 0 and r["p_one"] < 0.05)
        out[delta] = hits / reps
    return out
