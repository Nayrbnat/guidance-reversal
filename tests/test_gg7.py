"""
Tests for Stage 7 (sale-level H5 with industry fixed effects, abrupt exits and routine-sale classification): gg_stage7.
Run:  uv run python tests/test_gg7.py    (from the repo root; no pytest needed)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


def test_exit_flags():
    import gg_stage7 as S7
    d = pd.DataFrame({
        "gvkey": ["000001", "000001", "000002", "000003", "000004", "000005", "000006", "000007"],
        "execid": ["e1", "e1", "e2", "e3", "e4", "e5", "e6", "e9"],
        "trandate": pd.to_datetime(["2015-01-05", "2016-07-01", "2015-01-05", "2015-01-05", "2015-01-05", "2015-01-05", "2015-01-05", "2015-01-05"])})
    ev = pd.DataFrame({
        "event_id": ["v1", "v2", "v3", "v4", "v5", "v6", "v6b"],
        "gvkey": ["000001", "000002", "000003", "000004", "000005", "000006", "000006"],
        "execid": ["e1", "e2", "e3", "e4", "e5", "e6", "e6"],
        "dep_ann": pd.to_datetime(["2015-05-01", "2016-06-01", "2015-03-01", "2015-06-01", "2015-01-05", "2015-09-01", "2015-04-01"]),
        "abrupt_rule": [0.0, 1.0, np.nan, 0.0, 1.0, 1.0, 0.0]})
    labels = pd.DataFrame({"event_id": ["v1"], "abrupt_ann_label": [1.0]})
    out = S7.exit_flags(d, ev, labels)
    check("an exit announced within 365 days after the sale is flagged, an earlier or later one is not",
          out.X_ann.tolist() == [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], str(out.X_ann.tolist()))
    check("an annotator label overrides the rule", out.abrupt_class.iloc[0] == 1.0)
    check("an exit with no label and no rule is unclassified (NaN), not zero", np.isnan(out.abrupt_class.iloc[3]) and out.X_ann.iloc[3] == 1.0)
    check("a rule-classified planned exit has class 0", out.abrupt_class.iloc[4] == 0.0)
    check("an announcement on the sale day itself is not after the sale", out.X_ann.iloc[5] == 0.0)
    check("with two announcements in the window the earliest decides", out.abrupt_class.iloc[6] == 0.0)
    check("a sale-day with no exit has X = 0 and no abruptness class", out.X_ann.iloc[7] == 0.0 and np.isnan(out.abrupt_class.iloc[7]))
    check("rows and order are preserved", len(out) == len(d) and out.gvkey.tolist() == d.gvkey.tolist())


def test_routine_class():
    import gg_stage7 as S7
    sales = pd.DataFrame({
        "ep_id": [1, 1, 1, 1, 1,   2, 2, 2, 2,   3, 3,   4, 4, 4,   5, 5, 5, 5, 5],
        "trandate": pd.to_datetime([
            "2012-03-10", "2013-03-20", "2014-03-05", "2015-03-02", "2016-03-04",     # ep 1: every March, and again in 2016
            "2012-06-10", "2013-01-20", "2014-09-05", "2015-03-02",                   # ep 2: sells each year, never March
            "2014-03-10", "2015-03-02",                                               # ep 3: only one prior year
            "2012-03-10", "2014-03-05", "2015-03-02",                                 # ep 4: no sale in 2013
            "2012-03-01", "2012-07-01", "2013-03-09", "2014-03-14", "2015-03-02"])})  # ep 5: March each year plus other months
    q = pd.DataFrame({"ep_id": [1, 2, 3, 4, 5], "trandate": pd.to_datetime(["2015-03-02"] * 5)})
    out = S7.routine_class(q, sales).set_index("ep_id")["cls"]
    check("sold in the same month in each of the three previous years is routine", out.loc[1] == "routine" and out.loc[5] == "routine")
    check("sold in each of the three previous years but not in that month is opportunistic", out.loc[2] == "opportunistic")
    check("fewer than three prior years of sales is unclassified", out.loc[3] == "unclassified" and out.loc[4] == "unclassified")
    q2 = pd.DataFrame({"ep_id": [1], "trandate": pd.to_datetime(["2015-03-02"])})
    s2 = sales[sales.trandate < "2016-01-01"]
    check("sales after the sale date never change the classification",
          S7.routine_class(q2, s2)["cls"].iloc[0] == S7.routine_class(q2, sales)["cls"].iloc[0] == "routine")
    q3 = pd.DataFrame({"ep_id": [1, 2], "trandate": pd.to_datetime(["2015-03-02", "2015-03-02"])}, index=[10, 20])
    check("index and row order are preserved", S7.routine_class(q3, sales).index.tolist() == [10, 20])


def test_sic2():
    import gg_stage7 as S7
    got = S7.sic2(pd.Series([2834, "3572", None, "0100", 7372.0]))
    check("two-digit industry from a numeric or string SIC, missing stays missing",
          got.iloc[0] == "28" and got.iloc[1] == "35" and pd.isna(got.iloc[2]) and got.iloc[3] == "01" and got.iloc[4] == "73", str(got.tolist()))


def test_industry_fe_regression():
    import gg_stage6 as S6
    rng = np.random.default_rng(7)
    rows = []
    for f in range(80):
        z = rng.normal()                                    # transparency is constant within a firm, as in the real data
        for m in range(40):
            rows.append((f"{f:06d}", f"{(f % 8):02d}", f"{2010 + m // 12}-{m % 12 + 1:02d}", z))
    d = pd.DataFrame(rows, columns=["gvkey", "sic2", "ym", "Z"])
    n = len(d)
    d["G"] = (rng.random(n) < 0.5).astype(float)
    d["X"] = (rng.random(n) < 0.2).astype(float)
    for c in ("lmcap", "ret12", "lval", "ltenure", "age", "V"):
        d[c] = rng.normal(0, 1, n)
    d["Y63"] = rng.normal(0, 4, n) - 4.0 * d.Z * d.G * d.X
    r = S6.run_primary(d, "Y63", firm="sic2", controls=S6.CONTROLS + ["V"])
    check("industry fixed effects: the H5 coefficient is recovered with the right sign", r["ZGX"]["coef"] < -2 and r["p_one"] < 0.05, str((r["ZGX"]["coef"], r["p_one"])))
    check("the volatility control is in the regression", "V" in r)
    d0 = d.assign(Y63=rng.normal(0, 4, n))
    check("no planted effect: no rejection under industry fixed effects", S6.run_primary(d0, "Y63", firm="sic2", controls=S6.CONTROLS + ["V"])["p_one"] > 0.05)


def main():
    test_exit_flags()
    test_routine_class()
    test_sic2()
    test_industry_fe_regression()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
