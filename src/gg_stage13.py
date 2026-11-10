"""Stage 13: a wider sale condition for the Guidance Reversal flag (spec: docs/STAGE13_SPEC.md).

The original flag needs the executive's sale to fall AFTER the latest guidance. Stage 13 replaces that one condition with: a sale from `months_before`
months BEFORE the latest guidance up to the announcement date. Nothing after the announcement is used, so the flag is still built only from information
known on that date. Guidance within 180 days and an abrupt exit are unchanged. This module only adds NEW columns (sell_val_win9, gr_candidate9, gr_rule9);
it never changes the original flag columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sale_value_in_window(sales: pd.DataFrame, last_guid, dep_ann, months_before: int = 9) -> float:
    """Sum of `val` of sales dated in [last_guid - months_before months, dep_ann] (both ends inclusive); 0 when there is no guidance date or no sale."""
    if pd.isna(last_guid) or sales is None or len(sales) == 0:
        return 0.0
    lo = pd.Timestamp(last_guid) - pd.DateOffset(months=months_before)
    d = pd.to_datetime(sales["trandate"])
    return float(sales.loc[(d >= lo) & (d <= pd.Timestamp(dep_ann)), "val"].sum())


def add_flags(ev: pd.DataFrame) -> pd.DataFrame:
    """Add gr_candidate9 (guided in the last 180 days AND a sale in the wider window) and gr_rule9 (that AND an abrupt exit). Original columns untouched."""
    out = ev.copy()
    out["gr_candidate9"] = ((out["sell_val_win9"] > 0) & (out["guided_recent"] == 1)).astype(int)
    out["gr_rule9"] = ((out["gr_candidate9"] == 1) & (out["abrupt"] == 1)).astype(int)
    return out


def window_features(ev: pd.DataFrame, ins: pd.DataFrame, crsp, months_before: int = 9) -> pd.Series:
    """sell_val_win9 for each event: the departing executive's open-market sale value in the wider window, with the SAME owner matching, price filling and
    dispositions as guide_and_go.insider_features (strict name match, score 3). `ins` is the output of guide_and_go.load_insiders."""
    import guide_and_go as G
    import gg_labels as GL
    s = ins[ins.acqdisp == "D"].copy()
    s["tprice"] = s.tprice.where(s.tprice > 0)
    s["tprice"] = s.tprice.fillna(G.price_at(crsp, s, "trandate"))
    s["val"] = s.shares.abs() * s.tprice
    by_p = dict(tuple(s.groupby("permno")))
    out = pd.Series(0.0, index=ev.index)
    for i, r in ev.iterrows():
        ss = by_p.get(r.permno)
        dsg = r.get("days_since_guid")
        if ss is None or pd.isna(dsg):
            continue
        p = GL.parse_name(r.exec_name)
        scores = {o: GL.match_parsed(p, o) for o in ss.owner.dropna().unique()}
        mine = ss[ss.owner.map(scores).fillna(0) >= 3]
        lg = pd.Timestamp(r.dep_ann) - pd.Timedelta(days=int(dsg))
        out.loc[i] = sale_value_in_window(mine, lg, r.dep_ann, months_before)
    return out
