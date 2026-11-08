"""
Tests for gg_crsp_ciz: converting CRSP's v2 (CIZ) daily file and name table to the column layout the project already uses.
Run:  uv run python tests/test_gg_ciz.py    (from the repo root; no pytest needed)
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


def test_daily():
    import gg_crsp_ciz as C
    v2 = pd.DataFrame({"permno": ["10001", "10001"], "dlycaldt": ["2025-01-02", "2025-01-03"], "dlyret": ["0.01", None],
                       "dlyprc": ["-12.5", "13.0"], "shrout": ["1000", "1000"]})
    out = C.daily_v2_to_v1(v2)
    check("columns renamed to the v1 layout", list(out.columns) == ["permno", "date", "ret", "prc", "shrout"], str(list(out.columns)))
    check("types: integer permno, datetime date, numeric ret and prc", out.permno.dtype.kind == "i" and str(out.date.dtype).startswith("datetime") and out.ret.dtype.kind == "f")
    check("a missing return stays missing, not zero", np.isnan(out.ret.iloc[1]) and out.ret.iloc[0] == 0.01)
    check("price sign is kept (the loader takes abs later, as for v1)", out.prc.iloc[0] == -12.5)
    check("duplicate permno-date rows are collapsed to one", len(C.daily_v2_to_v1(pd.concat([v2, v2]))) == 2)


def test_names():
    import gg_crsp_ciz as C
    v2 = pd.DataFrame({"permno": [1, 2], "cusip": ["AAAA0001", None], "namedt": ["2020-01-01", "2020-01-01"], "nameenddt": ["2025-12-31", "2025-12-31"],
                       "ticker": ["A", "B"], "siccd": [1, 2], "issuernm": ["x", "y"]})
    out = C.names_v2_to_v1(v2)
    check("names get ncusip from the 8-character cusip", out.ncusip.iloc[0] == "AAAA0001")
    check("names without a cusip are dropped (they cannot be linked)", len(out) == 1)
    check("name dates are datetimes", str(out.namedt.dtype).startswith("datetime") and str(out.nameenddt.dtype).startswith("datetime"))


def test_merge_names():
    import gg_crsp_ciz as C
    old = pd.DataFrame({"permno": [1, 2], "ncusip": ["AAAA0001", "BBBB0002"], "namedt": pd.to_datetime(["2001-01-01", "2001-01-01"]),
                        "nameenddt": pd.to_datetime(["2024-12-31", "2010-01-01"])})
    new = pd.DataFrame({"permno": [1], "ncusip": ["AAAA0001"], "namedt": pd.to_datetime(["2001-01-01"]), "nameenddt": pd.to_datetime(["2025-12-31"])})
    out = C.merge_names(old, new).set_index("permno")
    check("a name row present in both takes the NEW end date (the old one was cut at the cache date)", out.nameenddt.loc[1] == pd.Timestamp("2025-12-31"), str(out.nameenddt.to_dict()))
    check("rows only in the old table are kept", 2 in out.index and len(out) == 2)


def test_overlap():
    import gg_crsp_ciz as C
    a = pd.DataFrame({"permno": [1, 1, 2], "date": pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-02"]), "ret": [0.01, 0.02, 0.03]}).iloc[[0, 2]]
    b = pd.DataFrame({"permno": [1, 2], "date": pd.to_datetime(["2024-01-02"] * 2), "ret": [0.01, 0.05]})
    r = C.overlap_report(a, b)
    check("overlap report counts matched rows and mismatches beyond tolerance", r["n"] == 2 and r["mismatch"] == 1, str(r))
    check("overlap report is empty-safe", C.overlap_report(a.iloc[:0], b)["n"] == 0)


def main():
    test_daily()
    test_names()
    test_merge_names()
    test_overlap()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
