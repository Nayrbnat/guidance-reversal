"""
Tests for gg_compustat_px: turning Compustat daily security prices (comp.secd) into the daily return file layout the project uses
(permno-free: keyed by gvkey), because this account's CRSP ends 2025-12-31 while Compustat runs to 2026-11.
Run:  uv run python tests/test_gg_secd.py    (from the repo root; no pytest needed)
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


def _rows(prc, ajex, trfd, dates=None, gv="001690", iid="01", cshoc=None):
    n = len(prc)
    dates = dates or pd.bdate_range("2026-01-05", periods=n).strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({"gvkey": gv, "iid": iid, "datadate": dates, "prccd": prc, "ajexdi": ajex, "trfd": trfd, "cshoc": cshoc or [1e6] * n,
                         "tpci": "0", "curcdd": "USD"})


def test_plain_return():
    import gg_compustat_px as P
    out = P.secd_to_daily(_rows([10.0, 11.0, 9.9], [1.0] * 3, [1.0] * 3))
    check("price 10 -> 11 -> 9.9 gives returns NaN, +10%, -10%", np.isnan(out.ret.iloc[0]) and abs(out.ret.iloc[1] - 0.10) < 1e-12 and abs(out.ret.iloc[2] + 0.10) < 1e-12, str(out.ret.tolist()))


def test_split_and_dividend():
    import gg_compustat_px as P
    out = P.secd_to_daily(_rows([20.0, 10.0, 10.5], [2.0, 1.0, 1.0], [1.0, 1.0, 1.0]))
    check("a 2-for-1 split (price halves, ajexdi halves: larger before the split so prccd/ajexdi stays continuous) is NOT a -50% return", abs(out.ret.iloc[1]) < 1e-12, str(out.ret.tolist()))
    out = P.secd_to_daily(_rows([10.0, 9.5], [1.0, 1.0], [1.0, 1.05]))
    check("a dividend carried in the total-return factor adds to the return (9.5 x 1.05 / 10 - 1)", abs(out.ret.iloc[1] - (9.5 * 1.05 / 10.0 - 1)) < 1e-12, str(out.ret.tolist()))


def test_gaps_and_missing():
    import gg_compustat_px as P
    out = P.secd_to_daily(_rows([10.0, np.nan, 12.0], [1.0] * 3, [1.0] * 3))
    check("a missing price gives a missing return and the next return is not stretched across the hole", out.ret.isna().all(), str(out.ret.tolist()))
    dates = ["2026-01-05", "2026-01-06", "2026-02-20"]
    out = P.secd_to_daily(_rows([10.0, 10.1, 15.0], [1.0] * 3, [1.0] * 3, dates=dates))
    check("a gap of more than 7 calendar days between observations gives a missing return", np.isnan(out.ret.iloc[2]) and abs(out.ret.iloc[1] - 0.01) < 1e-12, str(out.ret.tolist()))
    check("rows with a zero or negative price are treated as missing", P.secd_to_daily(_rows([10.0, 0.0, 10.0], [1.0] * 3, [1.0] * 3)).ret.isna().all())


def test_layout_and_filters():
    import gg_compustat_px as P
    a = _rows([10.0, 11.0], [1.0] * 2, [1.0] * 2)
    b = _rows([5.0, 5.5], [1.0] * 2, [1.0] * 2, iid="02")                          # a non-primary issue of the same firm
    c = _rows([7.0, 7.7], [1.0] * 2, [1.0] * 2, gv="009999").assign(tpci="F")      # not a common share
    d = _rows([3.0, 3.3], [1.0] * 2, [1.0] * 2, gv="008888").assign(curcdd="CAD")  # not in US dollars
    out = P.secd_to_daily(pd.concat([a, b, c, d]), primary_iid={"001690": "01", "009999": "01", "008888": "01"})
    check("only the primary issue, common shares in US dollars are kept", set(out.gvkey) == {"001690"} and set(out.iid) == {"01"}, str(out.groupby("gvkey").size().to_dict()))
    check("the output layout is gvkey, date, ret, prc, shrout (shares in thousands, as the CRSP files)", list(out.columns[:5]) == ["gvkey", "date", "ret", "prc", "shrout"] and abs(out.shrout.iloc[0] - 1000.0) < 1e-9, str(list(out.columns)))
    check("dates are datetimes and rows are sorted by gvkey and date", str(out.date.dtype).startswith("datetime") and out.date.is_monotonic_increasing)


def test_overlap_report():
    import gg_compustat_px as P
    s = pd.DataFrame({"gvkey": ["1"] * 4, "date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]), "ret": [0.010, 0.020, -0.010, 0.050]})
    c = pd.DataFrame({"gvkey": ["1"] * 4, "date": s.date, "ret": [0.0101, 0.020, -0.010, 0.020]})
    r = P.overlap_vs_crsp(s, c, tol=1e-3)
    check("overlap report: rows compared, share within tolerance, and correlation", r["n"] == 4 and abs(r["share_within"] - 0.75) < 1e-12 and 0.5 < r["corr"] < 1.0, str(r))
    check("overlap report is empty-safe", P.overlap_vs_crsp(s.iloc[:0], c)["n"] == 0)


def test_splice():
    import gg_compustat_px as P
    crsp = pd.DataFrame({"permno": [1, 1, 1, 2], "date": pd.to_datetime(["2025-12-29", "2025-12-30", "2025-12-31", "2025-12-31"]),
                         "ret": [0.01, 0.02, 0.03, 0.04], "prc": [10.0, 10.2, 10.5, 5.0], "shrout": [1000.0] * 4})
    px = pd.DataFrame({"gvkey": ["000001"] * 4 + ["000009"] * 2, "date": pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-02", "2026-01-05"]),
                       "ret": [0.99, 0.05, -0.02, 0.01, 0.5, 0.5], "prc": [10.5, 11.0, 10.8, 10.9, 3.0, 3.1], "shrout": [1000.0] * 6, "iid": "01"})
    out = P.splice_returns(crsp, px, {"000001": 1}, cutoff="2025-12-31")
    one = out[out.permno == 1].set_index("date")
    check("CRSP rows are kept up to and including the cutoff, Compustat rows only after it", one.loc["2025-12-31", "ret"] == 0.03 and one.loc["2026-01-02", "ret"] == 0.05, str(one.ret.to_dict()))
    check("the Compustat row ON the cutoff date does not replace the CRSP row (no duplicates)", not out.duplicated(["permno", "date"]).any())
    check("a Compustat firm with no permno mapping is dropped", set(out.permno) == {1, 2})
    check("the output keeps the CRSP layout (permno, date, ret, prc, shrout) and is sorted", list(out.columns) == ["permno", "date", "ret", "prc", "shrout"] and out.sort_values(["permno", "date"]).index.equals(out.index))


def _days(n, start="2024-01-02"):
    return pd.bdate_range(start, periods=n)


def test_choose_issues():
    import gg_compustat_px as P
    d = _days(150)
    rng = np.random.default_rng(3)
    r_true = rng.normal(0, 0.02, len(d))
    crsp = pd.DataFrame({"permno": 10, "date": d, "ret": r_true})
    px = pd.concat([pd.DataFrame({"gvkey": "000001", "iid": "01", "date": d, "ret": rng.normal(0, 0.02, len(d))}),           # another share class: unrelated returns
                    pd.DataFrame({"gvkey": "000001", "iid": "02", "date": d, "ret": r_true})])                             # the matching issue
    out = P.choose_issues(px, crsp, {10: "000001"}, train_year=2024)
    check("the issue whose returns match the CRSP security on the training year is chosen", out.iloc[0].iid == "02" and len(out) == 1, str(out.to_dict("records")))
    check("the output carries permno, gvkey, iid and the match rate", {"permno", "gvkey", "iid", "rate"} <= set(out.columns) and out.iloc[0].rate > 0.99)
    bad = px[px.iid == "01"].assign(gvkey="000002")
    out = P.choose_issues(bad, crsp.assign(permno=11), {11: "000002"}, train_year=2024)
    check("a security whose best issue matches on under 95% of training days is dropped", len(out) == 0, str(out.to_dict("records")))
    short = P.choose_issues(px.assign(date=px.date), crsp.iloc[:50], {10: "000001"}, train_year=2024, min_days=100)
    check("a security with fewer than min_days of overlap in the training year is not chosen by matching", len(short) == 0, str(short.to_dict("records")))
    one = pd.DataFrame({"gvkey": "000003", "iid": "01", "date": d, "ret": rng.normal(0, 0.02, len(d))})
    fb = P.choose_issues(one, crsp.iloc[:0], {12: "000003"}, train_year=2024)
    check("a firm with exactly one issue and no CRSP overlap keeps its single issue (rate missing)", len(fb) == 1 and fb.iloc[0].iid == "01" and fb.iloc[0].permno == 12 and np.isnan(fb.iloc[0].rate))
    two = pd.concat([one, one.assign(iid="02")])
    check("a firm with two issues and no CRSP overlap is dropped (cannot choose)", len(P.choose_issues(two, crsp.iloc[:0], {12: "000003"}, train_year=2024)) == 0)
    outs = P.by_permno(px, P.choose_issues(px, crsp, {10: "000001"}, train_year=2024))
    check("the permno-keyed daily file has permno, date, ret, prc, shrout and only the chosen issue", list(outs.columns[:5]) == ["permno", "date", "ret", "prc", "shrout"] and set(outs.permno) == {10} and len(outs) == len(d), str(list(outs.columns)))


def main():
    test_plain_return()
    test_split_and_dividend()
    test_gaps_and_missing()
    test_layout_and_filters()
    test_overlap_report()
    test_splice()
    test_choose_issues()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
