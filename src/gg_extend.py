"""gg_extend.py -- extend the daily returns and the market series through 2025-12-31 (Stage 1W Amendment 1 and the Stage 10, 12 and 12b reruns).

The legacy CRSP daily file and the market index in the main cache stop on 2024-12-31 on this account. The CRSP v2 daily file
(`gg_cache/ciz_dsf_2023_2025.pkl`, through 2025-12-31, delisting returns included, already in the project layout) covers 2025. For the market
return after 2024 there is no CRSP index in the cache, so the Fama-French market (mktrf + rf) is used; over 2022 to 2024 it correlates 0.998 with
the cached CRSP index and runs about 1.4% a year higher, so market-adjusted returns on 2025 days are slightly lower than with the CRSP index.
Nothing before 2025 is changed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gg_tests as GT
import guide_and_go as G

CUT = pd.Timestamp("2024-12-31")


def extend_returns(db, crsp: pd.DataFrame, mkt: pd.DataFrame, permnos) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (crsp, mkt) with the 2025 rows appended. crsp keeps its columns (permno, date, ret, prc, shrout, ...); mkt has columns date, mkt."""
    v2 = pd.read_pickle(G.CACHE / "ciz_dsf_2023_2025.pkl")
    v2["date"] = pd.to_datetime(v2["date"])
    keep = pd.Series(permnos).dropna().astype(int).unique()
    new = v2[(v2.date > CUT) & v2.permno.isin(keep)]
    new = new[[c for c in crsp.columns if c in new.columns]]
    crsp2 = pd.concat([crsp[pd.to_datetime(crsp.date) <= CUT], new], ignore_index=True)
    fac = GT.load_factors(db).apply(pd.to_numeric, errors="coerce")
    days = pd.DatetimeIndex(sorted(new.date.unique() if "date" in new else []))
    days = days[days.isin(fac.index)]
    m_new = pd.DataFrame({"date": days, "mkt": (fac.loc[days, "mktrf"] + fac.loc[days, "rf"]).values})
    mkt2 = pd.concat([mkt[pd.to_datetime(mkt.date) <= CUT][["date", "mkt"]], m_new], ignore_index=True)
    print(f"  [extend] +{len(new):,} stock-days and +{len(m_new)} market days after 2024-12-31; returns now end {crsp2.date.max():%Y-%m-%d}", flush=True)
    return crsp2, mkt2
