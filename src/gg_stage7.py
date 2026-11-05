"""Stage 7: re-testing H5 at the sale level with industry-by-month fixed effects, abrupt exits and routine-sale classification
(spec: docs/STAGE7_SPEC.md, written before any Stage 7 regression was run). Helpers only; the regression is gg_stage6.run_primary with
firm="sic2"."""
from __future__ import annotations

import numpy as np
import pandas as pd


def exit_flags(d: pd.DataFrame, ev: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """For each sale-day (gvkey, execid, trandate): X_ann = 1 if that executive has a departure event announced strictly after the sale
    and within 365 days; abrupt_class = 1 (abrupt), 0 (planned or other) or NaN (announced but unclassified, or no exit). The
    annotators' label (abrupt_ann_label) overrides the rule (abrupt_rule); with several announcements in the window the earliest counts.
    Row order and index of d are preserved."""
    e = ev.merge(labels[["event_id", "abrupt_ann_label"]], on="event_id", how="left")
    e["abrupt"] = np.where(e["abrupt_ann_label"].notna(), e["abrupt_ann_label"], e["abrupt_rule"])
    e = e[["gvkey", "execid", "dep_ann", "abrupt"]].dropna(subset=["dep_ann"])
    m = pd.DataFrame({"_i": np.arange(len(d)), "gvkey": d["gvkey"].values, "execid": d["execid"].values,
                      "trandate": pd.to_datetime(d["trandate"]).values}).merge(e, on=["gvkey", "execid"], how="left")
    gap = (m["dep_ann"] - m["trandate"]).dt.days
    first = m[(gap > 0) & (gap <= 365)].sort_values("dep_ann").drop_duplicates("_i")
    x = np.zeros(len(d))
    cls = np.full(len(d), np.nan)
    x[first["_i"].values] = 1.0
    cls[first["_i"].values] = first["abrupt"].values
    out = d.copy()
    out["X_ann"] = x
    out["abrupt_class"] = cls
    return out


def routine_class(d: pd.DataFrame, sales_all: pd.DataFrame) -> pd.DataFrame:
    """Add cls to each sale-day (ep_id, trandate), after the routine versus opportunistic rule of Cohen, Malloy and Pomorski (2012) as
    specified in docs/STAGE7_SPEC.md: 'routine' if the CEO sold in the same calendar month in each of the three previous calendar years;
    'opportunistic' if the CEO sold in each of the three previous years but not in that month in all three; 'unclassified' otherwise.
    Only years before the sale year are used, so later sales never matter."""
    s = pd.DataFrame({"ep": sales_all["ep_id"].values, "y": pd.to_datetime(sales_all["trandate"]).dt.year.values,
                      "m": pd.to_datetime(sales_all["trandate"]).dt.month.values}).drop_duplicates()
    ym = set(zip(s["ep"], s["y"], s["m"]))
    yy = set(zip(s["ep"], s["y"]))
    cls = []
    for ep, t in zip(d["ep_id"].values, pd.to_datetime(d["trandate"])):
        y, mo = t.year, t.month
        if not all((ep, y - k) in yy for k in (1, 2, 3)):
            cls.append("unclassified")
        elif all((ep, y - k, mo) in ym for k in (1, 2, 3)):
            cls.append("routine")
        else:
            cls.append("opportunistic")
    out = d.copy()
    out["cls"] = cls
    return out


def sic2(sic) -> pd.Series:
    """Two-digit industry (first two digits of the four-digit SIC, zero-padded), NaN for a missing code."""
    x = pd.to_numeric(pd.Series(sic).reset_index(drop=True), errors="coerce")
    out = x.map(lambda v: f"{int(v):04d}"[:2] if pd.notna(v) else np.nan)
    out.index = pd.Series(sic).index
    return out
