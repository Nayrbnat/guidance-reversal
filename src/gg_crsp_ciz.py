"""CRSP v2 (CIZ) to the project's v1 layout. The v1 daily file (crsp.dsf) and name table stop at 2024-12-31 on this account; crsp.dsf_v2 and
crsp.stocknames_v2 run through 2025-12-31. These helpers only rename, type and de-duplicate; nothing here changes a return."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_v2_to_v1(v2: pd.DataFrame) -> pd.DataFrame:
    """dsf_v2 (permno, dlycaldt, dlyret, dlyprc, shrout) -> permno, date, ret, prc, shrout. dlyret in v2 already includes delisting returns."""
    d = pd.DataFrame({"permno": pd.to_numeric(v2["permno"], errors="coerce"), "date": pd.to_datetime(v2["dlycaldt"]),
                      "ret": pd.to_numeric(v2["dlyret"], errors="coerce"), "prc": pd.to_numeric(v2["dlyprc"], errors="coerce"),
                      "shrout": pd.to_numeric(v2["shrout"], errors="coerce")})
    d = d[d["permno"].notna()].copy()
    d["permno"] = d["permno"].astype("int64")
    return d.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"]).reset_index(drop=True)


def names_v2_to_v1(v2: pd.DataFrame) -> pd.DataFrame:
    """stocknames_v2 -> the v1 name layout (permno, ncusip, ticker, comnam, siccd, namedt, nameenddt); rows without an 8-character cusip are dropped."""
    d = pd.DataFrame({"permno": pd.to_numeric(v2["permno"], errors="coerce"), "ncusip": v2["cusip"].astype("string").str[:8],
                      "ticker": v2.get("ticker"), "comnam": v2.get("issuernm"), "siccd": v2.get("siccd"),
                      "namedt": pd.to_datetime(v2["namedt"]), "nameenddt": pd.to_datetime(v2["nameenddt"])})
    d = d[d["ncusip"].notna() & d["permno"].notna()].copy()
    d["permno"] = d["permno"].astype("int64")
    d["ncusip"] = d["ncusip"].astype(str)
    return d.drop_duplicates(["permno", "ncusip", "namedt"]).reset_index(drop=True)


def overlap_report(old: pd.DataFrame, new: pd.DataFrame, tol: float = 1e-6) -> dict:
    """Compare returns of the old (v1) and new (v2) files on the permno-dates both have: rows compared and rows that differ by more than tol."""
    m = old[["permno", "date", "ret"]].merge(new[["permno", "date", "ret"]], on=["permno", "date"], suffixes=("_old", "_new"))
    both = m.dropna(subset=["ret_old", "ret_new"])
    return {"n": int(len(m)), "compared": int(len(both)), "mismatch": int((both["ret_old"] - both["ret_new"]).abs().gt(tol).sum())}


def merge_names(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Old (v1, cut at the cache date) and new (v2) name rows on (permno, ncusip, namedt): the NEW row wins, so a name still active at the old cut-off
    gets its real end date. Rows only in the old table are kept."""
    cols = ["permno", "ncusip", "namedt", "nameenddt"]
    both = pd.concat([new[cols], old[cols]], ignore_index=True)
    return both.drop_duplicates(["permno", "ncusip", "namedt"], keep="first").reset_index(drop=True)
