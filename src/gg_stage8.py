"""Stage 8: does opacity change what insiders avoid? The mechanism, on all officers and directors (spec: docs/STAGE8_SPEC.md, written before
any Stage 8 outcome was computed). Data building only; the outcome is gg_stage6.car_table and the regression is gg_stage6.run_primary-style
(see scripts/run_stage8.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

OFFICER_CODES = {"O", "GC", "P", "CO", "CT", "DO", "OD"}
DIRECTOR_CODES = {"D", "CB"}


def role_group(code):
    """CEO, CFO, OFFICER (other officers, including director-officers), DIRECTOR (directors and the chairman); None for roles that are
    not company insiders here (beneficial owners B, holders, trusts and unknown codes)."""
    c = str(code)
    if c == "CEO":
        return "CEO"
    if c == "CFO":
        return "CFO"
    if c in OFFICER_CODES:
        return "OFFICER"
    if c in DIRECTOR_CODES:
        return "DIRECTOR"
    return None


def link_permno(df: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Add permno from the eight-digit CUSIP (cusip6 + cusip2) using the CRSP name table row valid on the sale date (a CUSIP can be reused by
    a different security). No match gives NaN. Row order and index are preserved."""
    x = pd.DataFrame({"_i": np.arange(len(df)), "ncusip": (df["cusip6"].astype(str) + df["cusip2"].astype(str)).values,
                      "d": pd.to_datetime(df["trandate"]).values})
    n = names[["permno", "ncusip", "namedt", "nameenddt"]].copy()
    n["ncusip"] = n["ncusip"].astype(str).str[:8]
    m = x.merge(n, on="ncusip", how="left")
    ok = (m["d"] >= m["namedt"]) & (m["d"] <= m["nameenddt"])
    m = m[ok].sort_values("namedt").drop_duplicates("_i", keep="last")
    out = df.copy()
    out["permno"] = pd.Series(m["permno"].values, index=m["_i"].values).reindex(np.arange(len(df))).values
    return out


def link_gvkey(df: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Add gvkey from the CRSP-Compustat link valid on the sale date, primary links only (linkprim P or C, linktype LC or LU); an open end
    date counts as still valid. No match gives NaN."""
    lk = link[link["linkprim"].isin(["P", "C"]) & link["linktype"].isin(["LC", "LU"]) & link["lpermno"].notna()].copy()
    lk["lpermno"] = lk["lpermno"].astype("int64")
    lk["linkenddt"] = lk["linkenddt"].fillna(pd.Timestamp("2262-01-01"))
    x = pd.DataFrame({"_i": np.arange(len(df)), "permno": pd.to_numeric(df["permno"], errors="coerce").values, "d": pd.to_datetime(df["trandate"]).values})
    x = x[x["permno"].notna()].copy()
    x["permno"] = x["permno"].astype("int64")
    m = x.merge(lk, left_on="permno", right_on="lpermno", how="left")
    ok = (m["d"] >= m["linkdt"]) & (m["d"] <= m["linkenddt"])
    m = m[ok].sort_values("linkdt").drop_duplicates("_i", keep="last")
    out = df.copy()
    out["gvkey"] = pd.Series(m["gvkey"].values, index=m["_i"].values).reindex(np.arange(len(df))).values
    return out


def insider_sale_days(ins: pd.DataFrame, names: pd.DataFrame, link: pd.DataFrame, *, min_value: float = 100_000.0,
                      start: str = "2004-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """One row per person and issuer and day with open-market sales (transaction code S) by officers and directors, values summed
    (|shares| x price), kept if value >= min_value and the date is in [start, end], linked to permno and gvkey (rows that do not link are
    dropped). Columns: personid, trandate, val, rolecode1, role, cusip6, cusip2, permno, gvkey."""
    s = ins[ins["trancode"] == "S"].copy()
    s["trandate"] = pd.to_datetime(s["trandate"], errors="coerce")
    s["role"] = s["rolecode1"].map(role_group)
    s = s[s["role"].notna() & (s["trandate"] >= pd.Timestamp(start)) & (s["trandate"] <= pd.Timestamp(end))]
    s["val"] = pd.to_numeric(s["shares"], errors="coerce").abs() * pd.to_numeric(s["tprice"], errors="coerce")
    d = s.groupby(["personid", "cusip6", "cusip2", "trandate"], as_index=False).agg(val=("val", "sum"), rolecode1=("rolecode1", "first"), role=("role", "first"))
    d = d[d["val"] >= min_value].reset_index(drop=True)
    d = link_permno(d, names)
    d = d[d["permno"].notna()].reset_index(drop=True)
    d["permno"] = d["permno"].astype("int64")
    d = link_gvkey(d, link)
    return d[d["gvkey"].notna()].reset_index(drop=True)


# ---------------------------------------------------------------- the mechanism regression and its power
def run_mechanism(d: pd.DataFrame, y: str, *, z: str = "Z", g: str | None = None, controls=(), firm: str = "sic2", period: str = "yq",
                  cluster: str = "gvkey", two_way: bool = True) -> dict:
    """y on Z (or on G, Z and Z x G when g is given) and controls, with `firm` (industry) and `period` fixed effects and standard errors
    clustered by `cluster` and (when two_way, the default) by `period`. Key Z (or ZG); H5 predicts a NEGATIVE coefficient (the avoided loss is larger where
    transparency is low), so the one-sided p is taken for sign -1."""
    import gg_econ as E
    import gg_stage3 as S3
    d = d.copy()
    d["Z"] = d[z].astype(float)
    xs = ["Z"]
    key = "Z"
    if g is not None:
        d["G"] = d[g].astype(float)
        d["ZG"] = d["Z"] * d["G"]
        xs = ["G", "Z", "ZG"]
        key = "ZG"
    xs = xs + [c for c in controls if c in d.columns]
    res = E.fe_ols(d, y, xs, firm=firm, year=period, cluster=cluster, cluster2=period if two_way else None)
    return S3.summarise(res, key, -1)


def power_sim(d: pd.DataFrame, y: str, deltas=(0.0, 1.0, 2.0), reps: int = 30, seed: int = 0, **kw) -> dict:
    """Share of firm-cluster bootstrap replications in which the mechanism test rejects (coefficient < 0 and one-sided p < 0.05) when an
    effect of -delta x Z is planted into y for every row. delta = 0 gives the false-positive rate."""
    rng = np.random.default_rng(seed)
    firms = d["gvkey"].unique()
    groups = {f: d[d["gvkey"] == f] for f in firms}
    out = {}
    for delta in deltas:
        hits = 0
        for _ in range(reps):
            pick = rng.choice(len(firms), len(firms))
            b = pd.concat([groups[firms[j]].assign(gvkey=f"{i}_{firms[j]}") for i, j in enumerate(pick)], ignore_index=True)
            b[y] = b[y] - delta * b["Z"]
            r = run_mechanism(b, y, **kw)
            hits += int(r["Z"]["coef"] < 0 and r["p_one"] < 0.05)
        out[delta] = hits / reps
    return out
