"""
Tests for Stage 6 (H5 at the level of the individual insider sale): gg_stage6.
Run:  uv run python tests/test_gg6.py    (from the repo root; no pytest needed)
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


def _ep():
    return pd.DataFrame({"ep_id": [1, 2, 3, 4], "gvkey": ["000001", "000002", "000003", "000004"],
                         "leftofc": pd.to_datetime(["2015-06-01", "2016-09-01", pd.NaT, "2015-03-10"]),
                         "deceased": [False, False, False, True]})


def test_sale_days():
    import gg_stage6 as S6
    sales = pd.DataFrame({
        "ep_id": [1, 1, 1, 2, 2, 3, 4, 1],
        "trandate": pd.to_datetime(["2015-01-05", "2015-01-05", "2015-02-02", "2015-01-05", "2003-12-31", "2015-01-05", "2015-01-05", "2025-06-01"]),
        "val": [60_000.0, 70_000.0, 20_000.0, 500_000.0, 900_000.0, 300_000.0, 400_000.0, 800_000.0]})
    pm = {1: 101, 2: 102, 3: 103, 4: 104}
    d = S6.sale_days(sales, _ep(), pm, min_value=100_000, start="2004-01-01", end="2024-12-31")
    one = d[(d.ep_id == 1) & (d.trandate == "2015-01-05")]
    check("two sales on one day by one CEO become one sale-day with the summed value", len(one) == 1 and one.val.iloc[0] == 130_000.0)
    check("a sale-day under the value filter is dropped", not ((d.ep_id == 1) & (d.trandate == "2015-02-02")).any())
    check("sales outside the date window are dropped", not (d.trandate < "2004-01-01").any() and not (d.trandate > "2024-12-31").any())
    check("a CEO who dies within 365 days of the sale is dropped", 4 not in set(d.ep_id))
    x = d.set_index("ep_id").X
    check("X is 1 when the CEO leaves within 365 days after the sale", x.loc[1] == 1)
    check("X is 0 when the CEO leaves more than 365 days after the sale", x.loc[2] == 0)
    check("X is 0 when the CEO never leaves", x.loc[3] == 0)
    check("permno and gvkey are attached", set(d.columns) >= {"permno", "gvkey", "X", "val"} and d.set_index("ep_id").permno.loc[3] == 103)


def test_guidance():
    import gg_stage6 as S6
    df = pd.DataFrame({"permno": [1, 1, 1, 1, 2], "trandate": pd.to_datetime(["2015-03-01", "2015-03-01", "2015-03-01", "2015-03-01", "2015-03-01"])})
    df = df.iloc[[0]].reset_index(drop=True)
    g = pd.DataFrame({"permno": [1, 1, 1, 2], "t": pd.to_datetime(["2015-01-20", "2015-02-10", "2015-03-01", "2014-06-01"]), "opt": [0.01, -0.02, 0.05, 0.03]})
    out = S6.attach_guidance(df, g, max_days=180)
    check("the latest guidance strictly before the sale date is used (same-day guidance is not)", out.G.iloc[0] == 1 and out.O.iloc[0] == 0 and out.days_since_guid.iloc[0] == 19, str(out.T if hasattr(out, 'T') else out))
    df2 = pd.DataFrame({"permno": [1, 2], "trandate": pd.to_datetime(["2015-03-02", "2015-03-01"])})
    out2 = S6.attach_guidance(df2, g, max_days=180).set_index("permno")
    check("same-day guidance is excluded, the next-day sale sees it", out2.loc[1, "days_since_guid"] == 1 and out2.loc[1, "O"] == 1)
    check("guidance older than the window gives G = 0 and O = 0", out2.loc[2, "G"] == 0 and out2.loc[2, "O"] == 0)
    check("order and length of the sales frame are preserved", list(out2.index) == [1, 2])


def _market(n=900, seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2012-01-02", periods=n)
    fac = pd.DataFrame(rng.normal(0, 0.008, (n, 4)), index=cal, columns=["mktrf", "smb", "hml", "umd"])
    fac["rf"] = 0.00005
    return cal, fac


def _stock(cal, fac, beta, alpha_post, sale_idx, shift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    r = fac["rf"].values + fac[["mktrf", "smb", "hml", "umd"]].values @ beta + rng.normal(0, 0.0005, len(cal))
    r = r.copy()
    r[sale_idx + 1: sale_idx + 1 + 63] += alpha_post
    r[sale_idx + 1:] += shift
    return pd.DataFrame({"permno": 7, "date": cal, "ret": r})


def test_car():
    import gg_stage6 as S6
    cal, fac = _market()
    beta = np.array([1.1, 0.4, -0.2, 0.1])
    si = 500
    sales = pd.DataFrame({"permno": [7], "trandate": [cal[si]]})
    crsp = _stock(cal, fac, beta, alpha_post=0.002, sale_idx=si)
    out = S6.car_table(sales, crsp, fac, horizons=(21, 63))
    check("CAR over 63 days recovers the planted daily abnormal return", abs(out["car63"].iloc[0] - 63 * 0.002) < 0.02, str(out["car63"].iloc[0]))
    check("a shorter horizon uses only its own days", abs(out["car21"].iloc[0] - 21 * 0.002) < 0.02, str(out["car21"].iloc[0]))
    # no look-ahead: shifting every post-sale return by a constant moves the CAR by exactly shift x days, so the betas did not move
    shifted = _stock(cal, fac, beta, alpha_post=0.002, sale_idx=si, shift=0.001)
    out2 = S6.car_table(sales, shifted, fac, horizons=(63,))
    check("post-sale returns never enter the benchmark betas (shift moves CAR by exactly shift x h)", abs((out2["car63"].iloc[0] - out["car63"].iloc[0]) - 63 * 0.001) < 1e-9, str(out2["car63"].iloc[0] - out["car63"].iloc[0]))
    s_early = pd.DataFrame({"permno": [7], "trandate": [cal[50]]})
    check("a sale with fewer than 120 prior days has no CAR", np.isnan(S6.car_table(s_early, crsp, fac, horizons=(63,))["car63"].iloc[0]))
    s_late = pd.DataFrame({"permno": [7], "trandate": [cal[len(cal) - 10]]})
    check("a sale with too few later days has no CAR", np.isnan(S6.car_table(s_late, crsp, fac, horizons=(63,))["car63"].iloc[0]))
    sat = pd.DataFrame({"permno": [7], "trandate": [pd.Timestamp("2013-06-15")]})   # a Saturday: window starts on the next trading day
    check("a sale on a non-trading day starts its window at the next trading day", np.isfinite(S6.car_table(sat, crsp, fac, horizons=(63,))["car63"].iloc[0]))


def _sim(n_firms=70, n_months=60, delta=0.0, seed=3, sd=5.0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        for m in range(n_months):
            rows.append((f"{f:06d}", f"{2010 + m // 12}-{m % 12 + 1:02d}"))
    d = pd.DataFrame(rows, columns=["gvkey", "ym"])
    n = len(d)
    d["G"] = (rng.random(n) < 0.4).astype(float)
    d["X"] = (rng.random(n) < 0.12).astype(float)
    d["Z"] = rng.normal(0, 1, n)
    for c in ("lmcap", "ret12", "lval", "ltenure", "age"):
        d[c] = rng.normal(0, 1, n)
    d["Y63"] = rng.normal(0, sd, n) - delta * d.Z * d.G * d.X
    return d


def test_primary_regression():
    import gg_stage6 as S6
    d = _sim(delta=6.0)
    r = S6.run_primary(d, "Y63")
    check("primary regression returns the H5 coefficient with its standard error", {"ZGX", "GX", "G", "X", "Z"} <= set(r) and "p_one" in r and r["key"] == "ZGX")
    check("a planted H5 effect is recovered with the right sign and a small one-sided p", r["ZGX"]["coef"] < -3 and r["p_one"] < 0.05, str((r["ZGX"]["coef"], r["p_one"])))
    r0 = S6.run_primary(_sim(delta=0.0, seed=9), "Y63")
    check("no planted effect: the H5 test does not reject", r0["p_one"] > 0.05, str(r0["p_one"]))
    check("rows with a missing outcome are dropped, not zero-filled", S6.run_primary(_sim(delta=0.0).assign(Y63=lambda x: x.Y63.where(x.index % 2 == 0)), "Y63")["n"] < len(_sim()))


def test_power_sim():
    import gg_stage6 as S6
    d = _sim(n_firms=40, n_months=48, seed=4)
    pw = S6.power_sim(d, "Y63", deltas=(0.0, 12.0), reps=30, seed=1)
    check("power rises with the planted effect", pw[12.0] > pw[0.0] + 0.3, str(pw))
    check("the false-positive rate at no effect is near the nominal level", pw[0.0] <= 0.2, str(pw))


def test_fps_industry():
    import gg_stage6 as S6
    sic = pd.Series([2834, 3572, 3650, 7372, 5411, 2000, 5962, 7380, 2832, 3674, 7374, 7375])
    got = S6.fps_industry(sic).tolist()
    check("litigation-prone industries (biotech, computers, electronics, software, retail) are flagged", got == [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0], str(got))
    check("a missing SIC is not flagged", S6.fps_industry(pd.Series([np.nan, 2834])).tolist() == [0, 1])


def test_pre_sale_vol():
    import gg_stage6 as S6
    cal, fac = _market()
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, len(cal))
    si = 500
    crsp = pd.DataFrame({"permno": 7, "date": cal, "ret": r})
    sales = pd.DataFrame({"permno": [7], "trandate": [cal[si]]})
    v = S6.pre_sale_vol(sales, crsp, fac)["vol"].iloc[0]
    check("volatility is the standard deviation of daily returns over the 252 days before the sale", abs(v - np.std(r[si - 252:si], ddof=1)) < 1e-12, str(v))
    r2 = r.copy()
    r2[si:] *= 5.0
    v2 = S6.pre_sale_vol(sales, pd.DataFrame({"permno": 7, "date": cal, "ret": r2}), fac)["vol"].iloc[0]
    check("returns on and after the sale date never enter the volatility", abs(v2 - v) < 1e-12)
    early = pd.DataFrame({"permno": [7], "trandate": [cal[50]]})
    check("fewer than 120 prior days gives no volatility", np.isnan(S6.pre_sale_vol(early, crsp, fac)["vol"].iloc[0]))


def test_moderator_regression():
    import gg_stage6 as S6
    d = _sim(delta=0.0, seed=11)
    rng = np.random.default_rng(12)
    d["M"] = rng.normal(0, 1, len(d))
    d["Y63"] = d["Y63"] + (-5.0) * d["M"] * d["G"] * d["X"]            # a planted M x G x X coefficient of -5
    r = S6.run_primary(d, "Y63", moderators=["M"])
    check("moderator terms are in the regression and returned", {"M", "MG", "MX", "MGX"} <= set(r) and r["key"] == "ZGX")
    check("a planted M x G x X effect is recovered", r["MGX"]["coef"] < -3 and r["MGX"]["p_two"] < 0.05, str(r["MGX"]))
    check("the H5 coefficient on Z is not moved by a moderator effect it does not share", r["p_one"] > 0.05, str(r["p_one"]))
    r0 = S6.run_primary(d, "Y63")
    check("without moderators the regression is unchanged", "MGX" not in r0 and r0["n"] == r["n"])


def main():
    test_sale_days()
    test_guidance()
    test_car()
    test_primary_regression()
    test_fps_industry()
    test_pre_sale_vol()
    test_moderator_regression()
    test_power_sim()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
