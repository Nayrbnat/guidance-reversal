"""Daily returns from Compustat daily security prices (comp.secd), for dates after this account's CRSP ends (2025-12-31).
Return = (prccd / ajexdi x trfd)_t / (same)_{t-1} - 1: price adjusted for splits (ajexdi) and carrying dividends (trfd). It is NOT CRSP: no delisting
returns, and the primary issue is chosen from the CRSP-Compustat link table. Validate against CRSP on the overlap before any use (overlap_vs_crsp)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_DAYS = 7


def secd_to_daily(secd: pd.DataFrame, primary_iid: dict | None = None) -> pd.DataFrame:
    """comp.secd rows -> gvkey, date, ret, prc, shrout (+ iid). Keeps US-dollar common shares (tpci '0') of the primary issue (primary_iid maps gvkey -> iid;
    without it, every issue is kept). A return is missing when either price is missing or non-positive, or when more than MAX_GAP_DAYS calendar days
    separate the two observations (a hole in the series is not stretched). shrout is shares outstanding in thousands (cshoc / 1000)."""
    d = secd.copy()
    d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
    d = d[(d["tpci"].astype(str) == "0") & (d["curcdd"].astype(str) == "USD")]
    if primary_iid is not None:
        d = d[d["iid"].astype(str) == d["gvkey"].map(primary_iid)]
    d["date"] = pd.to_datetime(d["datadate"])
    for c in ("prccd", "ajexdi", "trfd", "cshoc"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.sort_values(["gvkey", "iid", "date"]).drop_duplicates(["gvkey", "iid", "date"], keep="last")
    ok = d["prccd"].gt(0) & d["ajexdi"].gt(0) & d["trfd"].fillna(1.0).gt(0)
    adj = (d["prccd"] / d["ajexdi"] * d["trfd"].fillna(1.0)).where(ok)
    g = d.groupby(["gvkey", "iid"], sort=False)
    prev = adj.groupby([d["gvkey"], d["iid"]], sort=False).shift(1)
    gap = (d["date"] - g["date"].shift(1)).dt.days
    d["ret"] = (adj / prev - 1.0).where(prev.notna() & adj.notna() & gap.le(MAX_GAP_DAYS))
    d["prc"] = d["prccd"].where(d["prccd"].gt(0))
    d["shrout"] = d["cshoc"] / 1000.0
    return d[["gvkey", "date", "ret", "prc", "shrout", "iid"]].sort_values(["gvkey", "date"]).reset_index(drop=True)


def overlap_vs_crsp(secd_ret: pd.DataFrame, crsp_ret: pd.DataFrame, tol: float = 1e-3) -> dict:
    """Compare Compustat-built and CRSP returns on the gvkey-dates both have (both frames: gvkey, date, ret): rows compared, share within tol, correlation."""
    m = secd_ret[["gvkey", "date", "ret"]].merge(crsp_ret[["gvkey", "date", "ret"]], on=["gvkey", "date"], suffixes=("_s", "_c"))
    m = m.dropna(subset=["ret_s", "ret_c"])
    if m.empty:
        return {"n": 0, "share_within": float("nan"), "corr": float("nan")}
    diff = (m["ret_s"] - m["ret_c"]).abs()
    return {"n": int(len(m)), "share_within": float((diff <= tol).mean()), "corr": float(np.corrcoef(m["ret_s"], m["ret_c"])[0, 1]),
            "p99_abs_diff": float(diff.quantile(0.99)), "max_abs_diff": float(diff.max())}


def splice_returns(crsp: pd.DataFrame, secd_px: pd.DataFrame, gvkey_to_permno: dict, cutoff: str) -> pd.DataFrame:
    """One permno-keyed daily file in the CRSP layout (permno, date, ret, prc, shrout): CRSP rows up to and including `cutoff`, Compustat-built rows
    strictly after it for the gvkeys that map to a permno. A Compustat row on or before the cutoff never replaces a CRSP row."""
    cut = pd.Timestamp(cutoff)
    a = crsp.loc[crsp["date"] <= cut, ["permno", "date", "ret", "prc", "shrout"]]
    b = secd_px.loc[secd_px["date"] > cut].copy()
    b["permno"] = b["gvkey"].astype(str).str.zfill(6).map({str(k).zfill(6): v for k, v in gvkey_to_permno.items()})
    b = b.dropna(subset=["permno"])
    b["permno"] = b["permno"].astype("int64")
    out = pd.concat([a, b[["permno", "date", "ret", "prc", "shrout"]]], ignore_index=True)
    return out.drop_duplicates(["permno", "date"], keep="first").sort_values(["permno", "date"]).reset_index(drop=True)


def choose_issues(px: pd.DataFrame, crsp: pd.DataFrame, permno_to_gvkey: dict, train_year: int = 2024, min_days: int = 100, min_rate: float = 0.95,
                  tol: float = 1e-3) -> pd.DataFrame:
    """Which Compustat issue (iid) is each CRSP security? For each permno, the issue whose daily returns agree with the CRSP security's (within tol) on the
    largest share of days IN THE TRAINING YEAR; the security is dropped if its best issue agrees on less than min_rate of training days (with at least min_days
    days of overlap). A security with no usable overlap keeps its gvkey's issue only when the gvkey has exactly one issue (rate missing). Uses returns of the
    training year only, so a later year can validate the choice out of sample. Columns: permno, gvkey, iid, rate. (One merge and one groupby, no per-security loop.)"""
    cmap = pd.DataFrame({"permno": [int(k) for k in permno_to_gvkey], "gvkey": [str(v).zfill(6) for v in permno_to_gvkey.values()]})
    c = crsp[crsp["date"].dt.year == train_year][["permno", "date", "ret"]].dropna().assign(permno=lambda x: x["permno"].astype("int64"))
    s = px[px["date"].dt.year == train_year][["gvkey", "iid", "date", "ret"]].dropna().assign(gvkey=lambda x: x["gvkey"].astype(str).str.zfill(6))
    m = cmap.merge(c, on="permno").merge(s, on=["gvkey", "date"], suffixes=("_c", "_s"))
    m["ok"] = (m["ret_s"] - m["ret_c"]).abs() <= tol
    g = m.groupby(["permno", "gvkey", "iid"]).agg(rate=("ok", "mean"), n=("ok", "size")).reset_index()
    g = g[g["n"] >= min_days]
    best = g.sort_values(["permno", "rate"], ascending=[True, False]).drop_duplicates("permno")
    chosen = best[best["rate"] >= min_rate][["permno", "gvkey", "iid", "rate"]]
    decided = set(best["permno"])                                    # had usable overlap: either chosen or dropped on purpose
    pxg = px.assign(gvkey=px["gvkey"].astype(str).str.zfill(6)).groupby("gvkey")["iid"].agg(["nunique", "first"])
    fb = []
    for _, r in cmap[~cmap["permno"].isin(decided)].iterrows():
        if r["gvkey"] in pxg.index and pxg.loc[r["gvkey"], "nunique"] == 1:
            fb.append({"permno": int(r["permno"]), "gvkey": r["gvkey"], "iid": pxg.loc[r["gvkey"], "first"], "rate": np.nan})
    out = pd.concat([chosen, pd.DataFrame(fb, columns=["permno", "gvkey", "iid", "rate"])], ignore_index=True)
    return out.astype({"permno": "int64"}).reset_index(drop=True)


def by_permno(px: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    """The daily file keyed by permno (permno, date, ret, prc, shrout) from the chosen (permno, gvkey, iid) pairs; only the chosen issue's rows."""
    p = px.copy()
    p["gvkey"] = p["gvkey"].astype(str).str.zfill(6)
    m = p.merge(chosen[["permno", "gvkey", "iid"]], on=["gvkey", "iid"])
    for col in ("prc", "shrout"):
        if col not in m:
            m[col] = np.nan
    m["permno"] = m["permno"].astype("int64")
    return m[["permno", "date", "ret", "prc", "shrout"]].drop_duplicates(["permno", "date"]).sort_values(["permno", "date"]).reset_index(drop=True)
