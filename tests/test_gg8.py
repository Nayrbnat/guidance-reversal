"""
Tests for Stage 8 (opacity and the information content of insider sales, all officers and directors): gg_stage8.
Run:  uv run python tests/test_gg8.py    (from the repo root; no pytest needed)
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


def _names():
    return pd.DataFrame({"permno": [1, 2, 3], "ncusip": ["AAAA0001", "AAAA0001", "BBBB0002"],
                         "namedt": pd.to_datetime(["2004-01-01", "2011-01-01", "2004-01-01"]),
                         "nameenddt": pd.to_datetime(["2010-12-31", "2020-12-31", "2024-12-31"])})


def _link():
    return pd.DataFrame({"gvkey": ["000010", "000011", "000012", "000013"], "lpermno": [1.0, 2.0, 3.0, 3.0],
                         "linkdt": pd.to_datetime(["2000-01-01", "2011-01-01", "2004-01-01", "2004-01-01"]),
                         "linkenddt": pd.to_datetime(["2010-12-31", pd.NaT, "2024-12-31", "2024-12-31"]),
                         "linktype": ["LC", "LU", "LC", "LC"], "linkprim": ["P", "P", "P", "J"]})


def test_link_permno():
    import gg_stage8 as S8
    df = pd.DataFrame({"cusip6": ["AAAA00", "AAAA00", "AAAA00", "BBBB00", "ZZZZ00"], "cusip2": ["01", "01", "01", "02", "01"],
                       "trandate": pd.to_datetime(["2008-05-01", "2015-05-01", "2025-05-01", "2020-01-01", "2008-05-01"])})
    out = S8.link_permno(df, _names())
    check("a reused CUSIP maps to the permno valid on the sale date", out.permno.iloc[0] == 1 and out.permno.iloc[1] == 2, str(out.permno.tolist()))
    check("a sale outside every name validity window has no permno", pd.isna(out.permno.iloc[2]))
    check("an unknown CUSIP has no permno", pd.isna(out.permno.iloc[4]) and out.permno.iloc[3] == 3)
    check("row order and length are preserved", len(out) == len(df) and out.cusip6.tolist() == df.cusip6.tolist())


def test_link_gvkey():
    import gg_stage8 as S8
    df = pd.DataFrame({"permno": [1.0, 2.0, 3.0, 1.0, np.nan], "trandate": pd.to_datetime(["2008-05-01", "2015-05-01", "2020-01-01", "2012-01-01", "2008-05-01"])})
    out = S8.link_gvkey(df, _link())
    check("the gvkey valid on the sale date is used, an open-ended link counts", out.gvkey.iloc[0] == "000010" and out.gvkey.iloc[1] == "000011", str(out.gvkey.tolist()))
    check("only primary links are used (a non-primary one is ignored)", out.gvkey.iloc[2] == "000012")
    check("a sale after the link ended or with no permno has no gvkey", pd.isna(out.gvkey.iloc[3]) and pd.isna(out.gvkey.iloc[4]))


def test_sale_days():
    import gg_stage8 as S8
    ins = pd.DataFrame({
        "personid": [1, 1, 2, 3, 4, 5, 6, 7],
        "owner": ["a", "a", "b", "c", "d", "e", "f", "g"],
        "cusip6": ["AAAA00"] * 7 + ["ZZZZ00"], "cusip2": ["01"] * 8,                 # the last row is an unknown CUSIP
        "rolecode1": ["CEO", "CEO", "B", "D", "O", "CFO", "CEO", "O"],
        "trandate": pd.to_datetime(["2008-05-01", "2008-05-01", "2008-05-01", "2008-05-01", "2008-05-01", "2008-05-01", "2003-05-01", "2020-06-01"]),
        "acqdisp": ["D"] * 8, "trancode": ["S", "S", "S", "S", "P", "S", "S", "S"],
        "shares": [-600.0, -500.0, 5000.0, 3000.0, 4000.0, 100.0, 9000.0, 4000.0], "tprice": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]})
    out = S8.insider_sale_days(ins, _names(), _link(), min_value=100_000, start="2004-01-01", end="2024-12-31")
    one = out[out.personid == 1]
    check("two sales by one person on one day become one insider-day with the summed value", len(one) == 1 and one.val.iloc[0] == 110_000.0, str(one.val.tolist()))
    check("beneficial owners are excluded", 2 not in set(out.personid))
    check("purchases are excluded", 4 not in set(out.personid))
    check("a sale under the value filter is dropped", 5 not in set(out.personid))
    check("a sale before the window is dropped", 6 not in set(out.personid))
    check("directors are kept and tagged", out.set_index("personid").role.loc[3] == "DIRECTOR")
    check("an insider-day carries permno and gvkey", {"permno", "gvkey", "role", "val"} <= set(out.columns) and out.set_index("personid").gvkey.loc[1] == "000010")
    check("an insider-day that does not link to a gvkey is dropped", 7 not in set(out.personid), str(out.personid.tolist()))


def test_role_group():
    import gg_stage8 as S8
    got = [S8.role_group(r) for r in ["CEO", "CFO", "O", "GC", "D", "CB", "DO", "P", "B", "XYZ"]]
    check("role groups: CEO, CFO, officers, directors; unknown and beneficial-owner roles are not insiders",
          got == ["CEO", "CFO", "OFFICER", "OFFICER", "DIRECTOR", "DIRECTOR", "OFFICER", "OFFICER", None, None], str(got))


def _sim(n_firms=60, n_q=24, per=6, effect=0.0, effect_g=0.0, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        z = rng.normal()                                       # transparency is a firm trait, as in the real data
        for q in range(n_q):
            for _ in range(per):
                rows.append((f"{f:06d}", f"{f % 6:02d}", f"{2010 + q // 4}Q{q % 4 + 1}", z))
    d = pd.DataFrame(rows, columns=["gvkey", "sic2", "yq", "Z"])
    n = len(d)
    d["G"] = (rng.random(n) < 0.5).astype(float)
    for c in ("lmcap", "ret12", "lval", "V"):
        d[c] = rng.normal(0, 1, n)
    for c in ("role_CEO", "role_CFO", "role_DIRECTOR"):
        d[c] = (rng.random(n) < 0.2).astype(float)
    d["Y63"] = rng.normal(0, 5, n) + effect * d.Z + effect_g * d.Z * d.G
    return d


CTRL = ["lmcap", "ret12", "lval", "V", "role_CEO", "role_CFO", "role_DIRECTOR"]


def test_mechanism_regression():
    import gg_stage8 as S8
    r = S8.run_mechanism(_sim(effect=-1.0), "Y63", controls=CTRL)
    check("the mechanism regression returns the Z coefficient with its standard error and a one-sided p", r["key"] == "Z" and "p_one" in r and r["Z"]["se"] > 0)
    check("a planted negative Z effect is recovered with the right sign and a small one-sided p", r["Z"]["coef"] < -0.5 and r["p_one"] < 0.05, str((r["Z"]["coef"], r["p_one"])))
    rej = [S8.run_mechanism(_sim(effect=0.0, seed=100 + k), "Y63", controls=CTRL)["p_one"] < 0.05 for k in range(20)]
    check("no planted effect: the test rejects at about the nominal rate across 20 simulations, not far above it", sum(rej) / 20 <= 0.2, str(sum(rej)))
    r2 = S8.run_mechanism(_sim(effect=0.0, effect_g=-1.0, seed=3), "Y63", controls=CTRL, g="G")
    check("with a guidance interaction the key is Z x G and a planted effect is recovered", r2["key"] == "ZG" and r2["ZG"]["coef"] < -0.5 and r2["p_one"] < 0.05, str(r2.get("ZG")))
    check("rows with a missing outcome are dropped, not zero-filled", S8.run_mechanism(_sim().assign(Y63=lambda x: x.Y63.where(x.index % 2 == 0)), "Y63", controls=CTRL)["n"] < len(_sim()))


def test_one_way_clustering():
    import gg_stage8 as S8
    d = _sim(effect=-1.0, seed=7)
    two = S8.run_mechanism(d, "Y63", controls=CTRL)
    one = S8.run_mechanism(d, "Y63", controls=CTRL, two_way=False)
    check("one-way (firm) clustering runs and returns the same coefficient as two-way", abs(one["Z"]["coef"] - two["Z"]["coef"]) < 1e-9, str((one["Z"]["coef"], two["Z"]["coef"])))
    check("one-way clustering gives a different standard error (the quarter term is dropped)", abs(one["Z"]["se"] - two["Z"]["se"]) > 1e-9, str((one["Z"]["se"], two["Z"]["se"])))


def test_mechanism_power():
    import gg_stage8 as S8
    d = _sim(n_firms=40, n_q=16, per=4, seed=5)
    pw = S8.power_sim(d, "Y63", deltas=(0.0, 3.0), reps=25, seed=1, controls=CTRL)
    check("power rises with the planted effect", pw[3.0] > pw[0.0] + 0.3, str(pw))
    check("the false-positive rate at no effect is near the nominal level", pw[0.0] <= 0.2, str(pw))


def main():
    test_link_permno()
    test_link_gvkey()
    test_role_group()
    test_sale_days()
    test_mechanism_regression()
    test_one_way_clustering()
    test_mechanism_power()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
