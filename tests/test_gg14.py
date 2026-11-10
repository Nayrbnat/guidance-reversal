"""
Tests for Stage 14 (valuation and turnaround controls for the flagged-firm returns): gg_stage14.
Run:  uv run python tests/test_gg14.py    (from the repo root; no pytest needed)
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


def _funda():
    return pd.DataFrame({"gvkey": ["000001"] * 3 + ["000002"], "datadate": pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31", "2023-12-31"]),
                         "ceq": [100.0, 200.0, 50.0, -10.0], "at": [1000.0, 1000.0, 2000.0, 500.0], "ib": [50.0, 100.0, -20.0, 5.0]})


def test_value_vars():
    import gg_stage14 as S14
    ev = pd.DataFrame({"gvkey": ["000001", "000001", "000002", "000003"], "dep_ann": pd.to_datetime(["2024-06-01", "2025-02-15", "2024-06-01", "2024-06-01"]),
                       "mcap": [4e8, 1e9, 1e8, 1e8]})
    out = S14.add_value_vars(ev, _funda(), lag_days=90)
    check("book-to-market uses the last report at least 90 days before the announcement (FY2023 for 2024-06-01)", abs(out.bm.iloc[0] - 200e6 / 4e8) < 1e-12, str(out.bm.tolist()))
    check("a report only 46 days old (FY2024 at 2025-02-15) is NOT used: the 90-day publication lag holds", abs(out.bm.iloc[1] - 200e6 / 1e9) < 1e-12, str(out.bm.tolist()))
    check("earnings yield = income / market cap and ROA = income / assets", abs(out.ep.iloc[0] - 100e6 / 4e8) < 1e-12 and abs(out.roa.iloc[0] - 0.1) < 1e-12)
    check("change in ROA is this year's ROA minus the previous fiscal year's (0.10 - 0.05)", abs(out.droa.iloc[0] - 0.05) < 1e-12, str(out.droa.tolist()))
    check("non-positive book equity gives a missing log BM and a flag", np.isnan(out.log_bm.iloc[2]) and out.bm_missing.iloc[2] == 1 and out.bm_missing.iloc[0] == 0)
    check("a firm with no Compustat report gets missing valuation variables, not zeros", np.isnan(out.bm.iloc[3]) and np.isnan(out.ep.iloc[3]))
    check("negative earnings flag", out.neg_earn.iloc[0] == 0 and S14.add_value_vars(ev.assign(dep_ann=pd.to_datetime(["2026-01-01"] * 4)), _funda(), 90).neg_earn.iloc[0] == 1)
    check("row order and the input columns are preserved", list(out.gvkey) == list(ev.gvkey) and len(out) == len(ev))


def _events(n=600, effect=0.0, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"event_id": [f"e{i}" for i in range(n)], "year": rng.integers(2005, 2012, n), "role": rng.choice(["CEO", "CFO"], n),
                      "log_mcap": rng.normal(0, 1, n), "pre_ret": rng.normal(0, 1, n), "log_bm": rng.normal(0, 1, n), "ep": rng.normal(0, 1, n)})
    d["flag"] = (rng.random(n) < 0.12).astype(int)
    d["bhar"] = 0.3 * d.log_bm + 0.2 * d.pre_ret + rng.normal(0, 1, n) + effect * d.flag
    return d


def test_matching():
    import gg_stage14 as S14
    d = _events(effect=0.8)
    r = S14.matched_diff(d, "bhar", d.loc[d.flag == 1, "event_id"].tolist(), ["log_mcap", "pre_ret", "log_bm", "ep"], k=5, reps=300, seed=0)
    check("a planted flagged effect of +0.8 is recovered by the matched difference", 0.4 < r["diff"] < 1.2 and r["ci"][0] < r["diff"] < r["ci"][1], str(r))
    check("with no effect the 95% interval contains zero", S14.matched_diff(_events(effect=0.0, seed=4), "bhar", _events(effect=0.0, seed=4).query("flag == 1").event_id.tolist(), ["log_mcap", "pre_ret", "log_bm", "ep"], k=5, reps=300, seed=0)["ci"][0] < 0 < S14.matched_diff(_events(effect=0.0, seed=4), "bhar", _events(effect=0.0, seed=4).query("flag == 1").event_id.tolist(), ["log_mcap", "pre_ret", "log_bm", "ep"], k=5, reps=300, seed=0)["ci"][1])
    # matches must come from the same year and role: make every unflagged event of another year wildly different and check they are never used
    d2 = _events(effect=0.0, seed=7)
    far = (d2.flag == 0) & (d2.year != 2006)
    d2.loc[far, "bhar"] = 1e6
    f06 = d2[(d2.flag == 1) & (d2.year == 2006)].event_id.tolist()
    r2 = S14.matched_diff(d2, "bhar", f06, ["log_mcap", "pre_ret", "log_bm", "ep"], k=3, reps=50, seed=0)
    check("matches are drawn only from the same announcement year and role", abs(r2["diff"]) < 100, str(r2["diff"]))
    d3 = d.copy()
    d3.loc[d3.flag == 1, "log_bm"] = np.nan
    r3 = S14.matched_diff(d3, "bhar", d3.loc[d3.flag == 1, "event_id"].tolist(), ["log_mcap", "log_bm"], k=5, reps=20, seed=0)
    check("flagged events missing a matching variable are dropped and counted", r3["n"] == 0 and r3["n_dropped"] > 0, str(r3))
    check("the result reports the number of flagged events used and the control matches per event", "n" in r and r["k"] == 5)


def test_regression():
    import gg_stage14 as S14
    d = _events(n=3000, effect=0.5, seed=3)
    d["q"] = d.year.astype(str) + "Q" + (np.arange(len(d)) % 4 + 1).astype(str)
    r = S14.flag_regression(d, "bhar", "flag", ["log_mcap", "pre_ret", "log_bm", "ep"], fe="year", cluster="q")
    check("the flag coefficient recovers the planted +0.5 with a standard error and n", abs(r["coef"] - 0.5) < 0.3 and r["se"] > 0 and r["n"] == 3000, str(r))
    r0 = S14.flag_regression(d, "bhar", "flag", [], fe="year", cluster="q")
    check("a regression with no controls runs and reports its own coefficient", "coef" in r0 and r0["n"] == 3000)
    dd = d.assign(bhar=d.bhar.where(np.arange(len(d)) % 2 == 0))
    check("rows with a missing outcome are dropped, not zero-filled", S14.flag_regression(dd, "bhar", "flag", ["log_mcap"], fe="year", cluster="q")["n"] == 1500)


def test_successor():
    import gg_stage14 as S14
    ec = pd.DataFrame({"gvkey": ["001"] * 6, "year": [2008, 2009, 2010, 2010, 2011, 2011], "execid": [1, 1, 1, 2, 2, 3], "ceoann": ["CEO", "CEO", "CEO", None, "CEO", None]})
    ev = pd.DataFrame({"gvkey": ["001"], "execid": [1], "dep_ann": pd.to_datetime(["2010-09-01"]), "role": ["CEO"]})
    out = S14.successor_outside(ev, ec)
    check("the successor is the other executive listed as CEO in the departure year or the next", out.successor.iloc[0] == 2, str(out.to_dict("records")))
    check("a successor with an earlier record at the firm (2010) is an insider, not an outside hire", out.outside.iloc[0] == 0)
    ec2 = ec.assign(year=[2008, 2009, 2010, 2011, 2011, 2011])
    out2 = S14.successor_outside(ev, ec2)
    check("a successor first listed at the firm only in the CEO year is an outside-hire proxy", out2.outside.iloc[0] == 1, str(out2.to_dict("records")))
    check("a CFO departure gets no successor classification", S14.successor_outside(ev.assign(role="CFO"), ec).outside.isna().all())
    ecs = ec.assign(execid=ec.execid.astype(str).map(lambda x: "X" + x))
    evs = ev.assign(execid="X1")
    outs = S14.successor_outside(evs, ecs)
    check("string executive IDs (as in the real ExecuComp file) work: successor kept as a string, outside as a number", outs.successor.iloc[0] == "X2" and outs.outside.iloc[0] == 0, str(outs.to_dict("records")))


def main():
    test_value_vars()
    test_matching()
    test_regression()
    test_successor()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
