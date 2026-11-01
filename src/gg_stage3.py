"""
gg_stage3.py -- Stage-3 tests (PREREGISTRATION.md, "Stage 3" + Amendment 1).

    python -u gg_panel_data.py feasibility                 # builds the cached panels (counts only)
    python -u gg_stage3.py --dry-run                       # mechanics only: outcomes are RANDOMISED, real results never printed
    python -u gg_stage3.py --reps 5000                     # the registered run

P1  hazard of CEO exit (LPM, firm + year-quarter FE, two-way clustered):   OPT x SALEPOST  > 0
P2  realised miss (firm + year-quarter FE, two-way clustered):             OPT x SALE x EXIT < 0
P3  12m Carhart alpha, leavers minus stayers (within-year permutation):    < 0
Holm over {P1, P2, P3}; every robustness check reported, none in the family.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg_econ as E
import gg_panel_data as PD
import gg_tests as GT
import gg_variants as GV
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS
START, END = PD.START, PD.END
D_END = PD.D_END
MIN = {"P1_exits": 100, "P2_cell": 100, "P3_each": 40}


def wins(s: pd.Series, p=0.01) -> pd.Series:
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


# ---------------------------------------------------------------- controls (all measured the day BEFORE the event date)
def market_table(crsp: pd.DataFrame) -> pd.DataFrame:
    t = crsp[["permno", "date", "ret", "prc", "shrout"]].sort_values(["permno", "date"]).copy()
    t["date"] = t.date.astype("datetime64[ns]")
    t["permno"] = t.permno.astype("int64")
    t["cum"] = np.log1p(t.ret.fillna(0).clip(lower=-0.99)).groupby(t.permno).cumsum()
    t["lmcap"] = np.log(t.prc * t.shrout * 1000)
    return t[["permno", "date", "cum", "lmcap"]].sort_values("date")


def add_market_controls(df: pd.DataFrame, tab: pd.DataFrame, date_col: str) -> pd.DataFrame:
    x = df[["permno", date_col]].copy()
    x["_i"] = np.arange(len(x))
    x["permno"] = x.permno.astype("int64")
    x["d0"] = (x[date_col] - pd.Timedelta(days=1)).astype("datetime64[ns]")
    x["d1"] = (x.d0 - pd.Timedelta(days=365)).astype("datetime64[ns]")
    a = pd.merge_asof(x.sort_values("d0"), tab, left_on="d0", right_on="date", by="permno", direction="backward").sort_values("_i")
    b = pd.merge_asof(x.sort_values("d1"), tab[["permno", "date", "cum"]], left_on="d1", right_on="date", by="permno", direction="backward").sort_values("_i")
    ok = (a.date.values >= (a.d0 - pd.Timedelta(days=10)).values)               # stale price (halted/delisted) -> missing
    out = df.copy()
    out["lmcap"] = np.where(ok, a.lmcap.values, np.nan)
    out["ret12"] = np.where(ok & (b.date.values >= (b.d1 - pd.Timedelta(days=10)).values), np.expm1(a.cum.values - b.cum.values), np.nan)
    out["ret12"] = wins(out.ret12.dropna()).reindex(out.index)
    return out


def add_exec_controls(df: pd.DataFrame, ep: pd.DataFrame, age_tab: pd.DataFrame, date_col: str) -> pd.DataFrame:
    e = ep.set_index("ep_id")
    out = df.copy()
    ten = (out[date_col] - e.start.reindex(out.ep_id).values).dt.days / 365.25
    out["ltenure"] = np.log1p(ten.clip(lower=0))
    x = out[["ep_id", date_col]].copy()
    x["_i"] = np.arange(len(x))
    x["gvkey"] = e.gvkey.reindex(x.ep_id).values
    x["execid"] = e.execid.reindex(x.ep_id).values
    x["year"] = x[date_col].dt.year.astype(float)
    m = pd.merge_asof(x.sort_values("year"), age_tab.sort_values("year"), on="year", by=["gvkey", "execid"], direction="nearest").sort_values("_i")
    out["age"] = (m.age + (m.year - m.age_year)).values
    return out


def load_age(db) -> pd.DataFrame:
    a = db.cached("execucomp_age_v1", lambda: db.fetch("comp_execucomp", "anncomp", ["gvkey", "execid", "year", "age"]))
    a = a.dropna(subset=["age", "year"]).copy()
    a["gvkey"] = a.gvkey.astype(str).str.zfill(6)
    a["execid"] = a.execid.astype(str)
    a["year"] = a.year.astype(float)
    a["age_year"] = a.year
    a["age"] = pd.to_numeric(a.age, errors="coerce")
    return a[["gvkey", "execid", "year", "age", "age_year"]].drop_duplicates(["gvkey", "execid", "year"])


# ---------------------------------------------------------------- Panel B frame
def release_frame(g: pd.DataFrame, ep: pd.DataFrame, si, si1m) -> pd.DataFrame:
    ok = (g.ep_id.notna() & g.period_end.gt(g.t) & g.days_to_p.le(400) & g.period_end.le(D_END)
          & g.t.between(f"{START}-01-01", D_END) & g.mid.notna() & g.consensus.notna() & g.opt.notna() & (g.pdicity != "SAN"))
    b = g[ok].copy()
    b["ep_id"] = b.ep_id.astype(int)
    e = ep.set_index("ep_id")
    left = pd.DatetimeIndex(pd.to_datetime(e.leftofc.reindex(b.ep_id).values))
    b["exit_date"] = left
    b["reason"] = e.reason.reindex(b.ep_id).values
    dec = np.asarray(e.deceased.reindex(b.ep_id).values, bool)
    inwin = np.asarray((left > b.t.values) & (left <= b.period_end.values))
    b["deceased_exit"] = inwin & dec
    b["exit_in"] = inwin & ~dec
    big = np.datetime64("2100-01-01", "ns")
    ex = np.where(pd.isna(left), big, left.values.astype("datetime64[ns]"))
    ids, t = b.ep_id.values, b.t.values
    b["sale"] = si.any_in(ids, t, np.minimum(b.period_end.values, ex))
    b["sale_1m"] = si1m.any_in(ids, t, np.minimum(b.period_end.values, ex))
    b["sale_r13"] = si.any_in(ids, t, np.minimum(b.period_end.values, ex - np.timedelta64(90, "D")))
    b["sale_pre"] = si.any_in(ids, (b.t - pd.Timedelta(days=180)).values, t)
    b["near_exit"] = np.asarray((left > b.t.values) & (left <= (b.t + pd.Timedelta(days=60)).values))
    b["opt_pos"] = (b.opt > 0).astype(float)
    b["has_actual"] = b.actual.notna() & b.miss.notna()
    b["tq"] = b.t.dt.to_period("Q").astype(str)
    return b[~b.deceased_exit].reset_index(drop=True)


def summarise(res: dict, key: str, sign: int) -> dict:
    r = {"n": res["n"], "n_clusters": res["n_clusters"], "k": res["k"]}
    for nm, v in res.items():
        if isinstance(v, dict):
            r[nm] = {kk: (None if not np.isfinite(vv) else vv) for kk, vv in v.items()}
    if key in res:
        r["key"] = key
        r["p_one"] = E.one_sided(res, key, sign)
        r["mde"] = E.min_detectable(res, key)
    return r


P2_CTRL = ["lmcap", "ret12", "ldays", "qtr", "ltenure", "age"]
P1_CTRL = ["sale12", "lmcap", "ret12", "ltenure", "age"]


def run_p2(B: pd.DataFrame, *, y="miss", opt="opt_pos", sale="sale", exit_="exit_in", cluster2="tq", rng=None, dry=False,
           optz=False) -> dict:
    d = B.copy()
    d["y"] = wins(d[y].dropna() * 100).reindex(d.index)
    if dry:   # randomise the outcome within year-quarter; treatment structure untouched
        d["y"] = d.groupby("tq").y.transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
    if optz:
        o = wins(d.opt.dropna())
        d["O"] = ((o - o.mean()) / o.std()).reindex(d.index)
    else:
        d["O"] = d[opt].astype(float)
    d["S"], d["X"] = d[sale].astype(float), d[exit_].astype(float)
    d["OS"], d["OX"], d["SX"], d["OSX"] = d.O * d.S, d.O * d.X, d.S * d.X, d.O * d.S * d.X
    d["ldays"], d["qtr"] = np.log(d.days_to_p.clip(lower=1)), (d.pdicity == "QTR").astype(float)
    xs = ["O", "S", "X", "OS", "OX", "SX", "OSX"] + P2_CTRL
    n0 = len(d)
    res = E.fe_ols(d, "y", xs, firm="gvkey", year="tq", cluster="gvkey", cluster2=cluster2)
    s = summarise(res, "OSX", -1)
    s.update({"n_before_dropna": n0, "n_S": int(d.S.sum()), "n_X": int(d.X.sum()), "n_SX": int((d.S * d.X).sum()),
              "n_OSX": int(d.OSX.fillna(0).astype(bool).sum()), "firms_SX": int(d.loc[d.SX == 1, "gvkey"].nunique())})
    return s


def run_p1(Q: pd.DataFrame, *, opt="opt_pos", sale="salepost", cluster2="tq", rng=None, dry=False, optz=False, logit=False) -> dict:
    d = Q.copy()
    d["y"] = d.exit * 100.0
    if dry:
        d["y"] = d.groupby("tq").y.transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
    if optz:
        o = wins(d.opt.dropna())
        d["O"] = ((o - o.mean()) / o.std()).reindex(d.index)
    else:
        d["O"] = d[opt].astype(float)
    d["S"] = d[sale].astype(float)
    d["OS"] = d.O * d.S
    d["sale12"] = d.sale12.astype(float)
    xs = ["O", "S", "OS"] + P1_CTRL
    if logit:
        d["yb"] = (d.y > 0).astype(float)
        res = E.logit_cluster(d, "yb", xs, "gvkey", year="tq")
        s = {"n": res["n"], "n_clusters": res["n_clusters"], "OS": res["OS"]}
        s["p_one"] = float(1 - __import__("scipy.stats", fromlist=["norm"]).norm.cdf(res["OS"]["t"]))
        return s
    res = E.fe_ols(d, "y", xs, firm="gvkey", year="tq", cluster="gvkey", cluster2=cluster2)
    s = summarise(res, "OS", +1)
    s.update({"n_exits": int(d.exit.sum()), "n_S": int(d.S.sum()), "n_S_exit": int(((d.S == 1) & (d.exit == 1)).sum())})
    return s


def audit(Q, B, sales, g) -> dict:
    """Look-ahead audit: timing rules re-verified in code on the built samples."""
    out = {}
    out["A_gdate_before_quarter"] = bool((Q.gdate < Q.qs).all())
    out["A_guidance_within_365d"] = bool((Q.days_since_guid <= 365).all() and (Q.days_since_guid >= 1).all())
    out["B_price_date_before_release"] = bool((g.t_m1 < g.t).all())
    out["B_period_after_release"] = bool((B.period_end > B.t).all())
    rng = np.random.default_rng(0)
    s_by = {k: v.trandate.values for k, v in sales.groupby("ep_id")}
    bad = 0
    ts = Q[Q.salepost].sample(min(2000, int(Q.salepost.sum())), random_state=1)
    for r in ts.itertuples():
        d = s_by.get(r.ep_id, np.array([], "datetime64[ns]"))
        bad += not ((d > np.datetime64(r.gdate)) & (d <= np.datetime64(r.qs))).any()
    out["A_salepost_verified_in_window_rows_checked"] = int(len(ts)); out["A_salepost_violations"] = int(bad)
    bad = 0
    tb = B[B.sale].sample(min(2000, int(B.sale.sum())), random_state=2)
    for r in tb.itertuples():
        d = s_by.get(r.ep_id, np.array([], "datetime64[ns]"))
        stop = np.datetime64(r.period_end) if pd.isna(r.exit_date) else min(np.datetime64(r.period_end), np.datetime64(r.exit_date))
        bad += not ((d > np.datetime64(r.t)) & (d <= stop)).any()
    out["B_sale_verified_in_window_rows_checked"] = int(len(tb)); out["B_sale_violations"] = int(bad)
    out["passed"] = bool(out["A_gdate_before_quarter"] and out["A_guidance_within_365d"] and out["B_price_date_before_release"]
                         and out["B_period_after_release"] and out["A_salepost_violations"] == 0 and out["B_sale_violations"] == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-p3", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run
    rng = np.random.default_rng(123)
    need = [DATA / f for f in ("panel_episodes.pkl", "panel_sales.pkl", "panel_guid.pkl", "panel_A_quarters.pkl")]
    if not all(p.exists() for p in need):
        sys.exit("Run `python -u gg_panel_data.py feasibility` first.")
    ep = pd.read_pickle(DATA / "panel_episodes.pkl"); sales = pd.read_pickle(DATA / "panel_sales.pkl")
    g = pd.read_pickle(DATA / "panel_guid.pkl"); Q = pd.read_pickle(DATA / "panel_A_quarters.pkl")
    db = G.WRDS(refresh=False)
    gperms = sorted(g.permno.unique())
    crsp, mkt, _ = G.load_crsp(db, gperms, START, END)
    tab = market_table(crsp)
    age_tab = load_age(db)
    si, si1m = PD.SaleIndex(sales), PD.SaleIndex(sales, 1e6)
    res = {"dry_run": dry, "min_sizes": MIN}
    print("DRY RUN: outcomes are randomised; only mechanics are shown." if dry else "REGISTERED RUN")

    # ------------------------------------------------------------ Panel B
    b_all = release_frame(g, ep, si, si1m)
    att = b_all.groupby("exit_in").has_actual.agg(["mean", "size"])
    res["attrition_A18"] = {"share_with_actual_exit0": float(att.loc[False, "mean"]), "share_with_actual_exit1": float(att.loc[True, "mean"]),
                            "n_exit0": int(att.loc[False, "size"]), "n_exit1": int(att.loc[True, "size"])}
    B = b_all[b_all.has_actual].reset_index(drop=True)
    B = add_exec_controls(add_market_controls(B, tab, "t"), ep, age_tab, "t")
    B["gvkey"] = B.gvkey.astype(str)
    print(f"Panel B: releases {len(B):,} (attrition: share with an actual {res['attrition_A18']['share_with_actual_exit0']:.1%} if no exit vs {res['attrition_A18']['share_with_actual_exit1']:.1%} if exit)")

    # ------------------------------------------------------------ Panel A
    Q = Q.copy()
    Q["permno"] = Q.permno.astype("int64"); Q["tq"] = Q.qs.dt.to_period("Q").astype(str)
    Q["opt_pos"] = (Q.opt > 0).astype(float)
    Q["reason"] = ep.set_index("ep_id").reason.reindex(Q.ep_id).values
    Q["salepost_r13"] = si.any_in(Q.ep_id.values, Q.gdate.values, (Q.qs - pd.Timedelta(days=90)).values)
    Q = add_exec_controls(add_market_controls(Q, tab, "qs"), ep, age_tab, "qs")
    Q["gvkey"] = Q.gvkey.astype(str)
    print(f"Panel A: CEO-quarters {len(Q):,}, exits {int(Q.exit.sum()):,}")
    res["audit"] = audit(Q.assign(salepost=Q.salepost.astype(bool)), B.assign(sale=B.sale.astype(bool)), sales, g)
    print("look-ahead audit:", res["audit"])
    assert res["audit"]["passed"], "look-ahead audit failed"

    # ------------------------------------------------------------ primary P1, P2
    P = {}
    P["P1"] = run_p1(Q, rng=rng, dry=dry)
    P["P2"] = run_p2(B, rng=rng, dry=dry)
    P["P1"]["eligible"] = P["P1"]["n_exits"] >= MIN["P1_exits"]
    P["P2"]["eligible"] = P["P2"]["n_SX"] >= MIN["P2_cell"]
    P["P1_firm_only_se"] = run_p1(Q, cluster2=None, rng=rng, dry=dry)
    P["P2_firm_only_se"] = run_p2(B, cluster2=None, rng=rng, dry=dry)
    res["primary"] = P
    for k in ("P1", "P2"):
        r = P[k]; key = r["key"]
        print(f"{k}: coef {key} = {r[key]['coef']:+.4f} (se {r[key]['se']:.4f}, t {r[key]['t']:+.2f}) one-sided p = {r['p_one']:.4f} | MDE {r['mde']:.4f} | n {r['n']:,}, clusters {r['n_clusters']}")

    # ------------------------------------------------------------ robustness (not in family)
    R = {}
    R["P1_R1_continuous"] = run_p1(Q, optz=True, rng=rng, dry=dry)
    R["P1_R2_sale_1m"] = run_p1(Q, sale="salepost_1m", rng=rng, dry=dry)
    R["P1_R6_logit_no_firmFE"] = run_p1(Q, logit=True, rng=rng, dry=dry)
    R["P1_R9_excl_retired"] = run_p1(Q[~((Q.exit == 1) & (Q.reason.astype(str).str.upper() == "RETIRED"))], rng=rng, dry=dry)
    R["P1_R12_2004_2014"] = run_p1(Q[Q.qs.dt.year <= 2014], rng=rng, dry=dry)
    R["P1_R12_2015_2025"] = run_p1(Q[Q.qs.dt.year >= 2015], rng=rng, dry=dry)
    R["P1_R13_skip_quarter"] = run_p1(Q, sale="salepost_r13", rng=rng, dry=dry)
    R["P2_R1_continuous"] = run_p2(B, optz=True, rng=rng, dry=dry)
    R["P2_R2_sale_1m"] = run_p2(B, sale="sale_1m", rng=rng, dry=dry)
    R["P2_R4_vs_consensus"] = run_p2(B, y="surp", rng=rng, dry=dry)
    first = B.sort_values("t").drop_duplicates(["permno", "period_end", "pdicity"])
    R["P2_R5_first_release"] = run_p2(first, rng=rng, dry=dry)
    R["P2_R9_excl_retired"] = run_p2(B[~(B.exit_in & (B.reason.astype(str).str.upper() == "RETIRED"))], rng=rng, dry=dry)
    R["P2_R11_pre_guidance_sale"] = run_p2(B, sale="sale_pre", rng=rng, dry=dry)
    R["P2_R12_2004_2014"] = run_p2(B[B.t.dt.year <= 2014], rng=rng, dry=dry)
    R["P2_R12_2015_2025"] = run_p2(B[B.t.dt.year >= 2015], rng=rng, dry=dry)
    R["P2_R13_sale_ends_90d_before_exit"] = run_p2(B, sale="sale_r13", rng=rng, dry=dry)
    R["P2_R14_drop_near_exit"] = run_p2(B[~B.near_exit], rng=rng, dry=dry)
    res["robustness"] = R
    if not dry:
        print("\nRobustness (coefficient of interest; sign, one-sided p):")
        for k, r in R.items():
            key = r.get("key") or "OS"
            c = r[key]
            print(f"  {k:<36s} {c['coef']:+9.4f}  t {c['t']:+6.2f}  p1 {r['p_one']:.3f}  n {r['n']:,}")

    # ------------------------------------------------------------ P3
    if not a.skip_p3:
        res["P3"] = run_p3(db, Q, crsp, mkt, a.reps, dry, rng)
    fam = {}
    if P["P1"]["eligible"]: fam["P1"] = P["P1"]["p_one"]
    if P["P2"]["eligible"]: fam["P2"] = P["P2"]["p_one"]
    if res.get("P3", {}).get("eligible"): fam["P3"] = res["P3"]["perm_p"]
    res["holm"] = GT.holm(fam) if fam else {}
    res["family"] = fam
    if not dry:
        print("\nFamily p-values:", {k: round(v, 4) for k, v in fam.items()}, "| Holm:", {k: round(v, 4) for k, v in res["holm"].items()})

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        return o

    name = "stage3_results_DRYRUN.json" if dry else "stage3_results.json"
    (RESULTS / name).write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
    print("wrote", RESULTS / name)


def run_p3(db, Q, crsp, mkt, reps, dry, rng) -> dict:
    fac_all = GT.load_factors(db)
    cal = np.sort(mkt.date.values.astype("datetime64[ns]"))
    cal = cal[np.isin(cal, fac_all.index.values)]
    fac = fac_all.reindex(pd.DatetimeIndex(cal)); T_ = len(cal)
    # stayers (Amendment 1 rule)
    st = PD.select_stayers(Q.assign(salepost=Q.salepost.astype(bool)))
    st = st.assign(event_id=[f"S{e}_{d:%Y%m%d}" for e, d in zip(st.ep_id, st.qs)], d=st.qs, year=st.qs.dt.year, leaver=False)
    # leavers: Stage-2 L1 x U2 events (same construction as gg_variants), announcement date, <= 2024
    ev = GV.load_eligible_events(START, END, "validation/candidate_labels.csv")
    L = GV.link_masks(ev)
    lv = ev[ev.elig & L["L1"] & ev.role_last.isin(["CEO", "CFO"]) & (ev.dep_ann_final <= pd.Timestamp("2024-12-31"))].copy()
    lv = lv.assign(d=lv.dep_ann_final, year=lv.dep_ann_final.dt.year, leaver=True)
    ev_all = pd.concat([st[["event_id", "permno", "d", "year", "leaver", "days_since_guid"]],
                        lv[["event_id", "permno", "d", "year", "leaver", "days_since_guid"]]], ignore_index=True)
    ev_all["permno"] = ev_all.permno.astype(int)
    win = GT.build_windows(ev_all, crsp, cal, date_col="d", skip=2, horizon=252)
    ev_all = ev_all[ev_all.event_id.isin(win.keys())].reset_index(drop=True)
    flags = ev_all.leaver.values
    n1, n0 = int(flags.sum()), int((~flags).sum())
    r = {"n_leavers": n1, "n_stayers": n0, "n_stayer_execs": int(st.ep_id.nunique()),
         "days_since_guid_median_leavers": float(ev_all.loc[flags, "days_since_guid"].median()),
         "days_since_guid_median_stayers": float(ev_all.loc[~flags, "days_since_guid"].median()),
         "eligible": bool(n1 >= MIN["P3_each"] and n0 >= MIN["P3_each"])}
    if dry:   # randomise who is a "leaver", within year, keeping sizes
        f = flags.copy()
        for y in np.unique(ev_all.year):
            idx = np.where(ev_all.year.values == y)[0]; f[idx] = rng.permutation(flags[idx])
        flags = f
    obs, a1, a0, null = GT.split_permutation(win, ev_all.event_id.values, flags, ev_all.year.values, fac, T_, reps=reps, seed=51)
    r.update({"alpha_leavers": a1, "alpha_stayers": a0, "diff": obs, "perm_p": GT.placebo_p(obs, null, "lower"), "perm_sd": float(np.nanstd(null))})
    if not dry:
        print(f"P3: leavers {n1} alpha {a1:+.1%} | stayers {n0} alpha {a0:+.1%} | diff {obs:+.1%} | perm p {r['perm_p']:.3f} (sd {r['perm_sd']:.1%})")
    return r


if __name__ == "__main__":
    main()
