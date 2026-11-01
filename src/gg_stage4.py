"""
gg_stage4.py -- Stage-4 EXPLORATORY iteration with a hold-out (PREREGISTRATION.md, "Stage 4").

    python -u gg_stage4.py explore      # estimates on 2004-2014, selects candidates (|t| >= 2.5) -> outputs/results/stage4_candidates.json
    python -u gg_stage4.py confirm      # tests ONLY the locked candidates once on 2015-2025 (Holm over the number carried)
    python -u gg_stage4.py profile      # event-time profile of misses around exits (descriptive, both halves)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg_econ as E
import gg_panel_data as PD
import gg_stage3 as S3
import gg_tests as GT
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS
SPLIT = 2014
AUG_P2 = ["lmcap", "ret12", "ldays", "qtr", "ltenure", "ltenure2", "age55", "age60", "age65", "prev_miss", "prev_miss_na",
          "style", "style_na", "lvol", "own", "own_na"]
AUG_P1 = ["sale12", "lmcap", "ret12", "ltenure", "ltenure2", "age55", "age60", "age65", "prev_miss", "prev_miss_na",
          "style", "style_na", "lvol", "own", "own_na"]
INTEREST = [("P1_M1", "OS"), ("P1_M2", "OS_nr"), ("P1_M3_retired", "OS"), ("P1_M3_other", "OS"),
            ("P2_M1", "OSX"), ("P2_M2", "OSX_nr"), ("P2_M3", "OSXr"), ("P2_M3", "OSXo")]
SIGN = {"P1": +1, "P2": -1}      # hypothesised direction (used only to label results)


# ---------------------------------------------------------------- extra features
def vol_table(crsp):
    t = crsp[["permno", "date", "ret"]].sort_values(["permno", "date"]).copy()
    t["date"] = t.date.astype("datetime64[ns]"); t["permno"] = t.permno.astype("int64")
    r = t.ret.astype(float)
    t["cn"] = r.notna().astype(float).groupby(t.permno).cumsum()
    t["c1"] = r.fillna(0).groupby(t.permno).cumsum()
    t["c2"] = (r.fillna(0) ** 2).groupby(t.permno).cumsum()
    return t[["permno", "date", "cn", "c1", "c2"]].sort_values("date")


def add_vol(df, tab2, date_col):
    x = df[["permno", date_col]].copy()
    x["_i"] = np.arange(len(x)); x["permno"] = x.permno.astype("int64")
    x["d0"] = (x[date_col] - pd.Timedelta(days=1)).astype("datetime64[ns]")
    x["d1"] = (x.d0 - pd.Timedelta(days=365)).astype("datetime64[ns]")
    a = pd.merge_asof(x.sort_values("d0"), tab2, left_on="d0", right_on="date", by="permno").sort_values("_i")
    b = pd.merge_asof(x.sort_values("d1"), tab2, left_on="d1", right_on="date", by="permno").sort_values("_i")
    n = a.cn.values - b.cn.values
    s = (a.c2.values - b.c2.values) - (a.c1.values - b.c1.values) ** 2 / np.maximum(n, 1)
    vol = np.sqrt(np.maximum(s, 0) / np.maximum(n - 1, 1))
    ok = (n >= 150) & (a.date.values >= (a.d0 - pd.Timedelta(days=10)).values) & (b.date.values >= (b.d1 - pd.Timedelta(days=10)).values)
    out = df.copy()
    out["lvol"] = np.where(ok, np.log(np.clip(vol, 1e-4, None)), np.nan)
    return out


def add_history(df, g, date_col):
    """Last REALISED miss (actual known: period end + 60 days before the date) and the firm's share of past guidance above consensus."""
    out = df.copy()
    out["_i"] = np.arange(len(out)); out["permno"] = out.permno.astype("int64")
    left = out[["_i", "permno", date_col]].sort_values(date_col)
    k = g[g.miss.notna()][["permno", "period_end", "miss"]].copy()
    k["miss"] = S3.wins(k.miss * 100)
    k["known"] = (k.period_end + pd.Timedelta(days=60)).astype("datetime64[ns]"); k["permno"] = k.permno.astype("int64")
    m = pd.merge_asof(left, k.sort_values("known")[["permno", "known", "miss"]], left_on=date_col, right_on="known",
                      by="permno", allow_exact_matches=False).sort_values("_i")
    out["prev_miss"] = m.miss.values
    gg = g[g.opt.notna()][["permno", "t", "opt"]].sort_values(["permno", "t"]).copy()
    gg["permno"] = gg.permno.astype("int64"); gg["t"] = gg.t.astype("datetime64[ns]")
    gg["o"] = (gg.opt > 0).astype(float)
    grp = gg.groupby("permno")
    cn = grp.cumcount().values; cs = (grp.o.cumsum() - gg.o).values
    gg["style"] = np.where(cn >= 3, cs / np.maximum(cn, 1), np.nan)
    m2 = pd.merge_asof(left, gg.sort_values("t")[["permno", "t", "style"]], left_on=date_col, right_on="t", by="permno",
                       allow_exact_matches=False).sort_values("_i")
    out["style"] = m2["style"].values
    return out.drop(columns="_i")


def load_own(db):
    a = db.cached("execucomp_age_own_v1", lambda: db.fetch("comp_execucomp", "anncomp",
                  ["gvkey", "execid", "year", "age", "shrown_excl_opts_pct"]))
    a = a.copy()
    a["gvkey"] = a.gvkey.astype(str).str.zfill(6); a["execid"] = a.execid.astype(str)
    a["year"] = pd.to_numeric(a.year, errors="coerce").astype(float)
    a["own"] = pd.to_numeric(a.shrown_excl_opts_pct, errors="coerce")
    return a.dropna(subset=["year"]).drop_duplicates(["gvkey", "execid", "year"])[["gvkey", "execid", "year", "own"]]


def add_own(df, ep, own_tab, date_col):
    e = ep.set_index("ep_id")
    x = df[["ep_id", date_col]].copy(); x["_i"] = np.arange(len(x))
    x["gvkey"] = e.gvkey.reindex(x.ep_id).values; x["execid"] = e.execid.reindex(x.ep_id).values
    x["year"] = x[date_col].dt.year.astype(float)
    m = pd.merge_asof(x.sort_values("year"), own_tab.sort_values("year"), on="year", by=["gvkey", "execid"],
                      direction="backward", allow_exact_matches=False).sort_values("_i")   # prior fiscal years only
    out = df.copy(); out["own"] = m.own.values
    return out


def finish_controls(d):
    d = d.copy()
    d["ltenure2"] = d.ltenure ** 2
    d["age55"] = ((d.age >= 55) & (d.age < 60)).astype(float)
    d["age60"] = ((d.age >= 60) & (d.age < 65)).astype(float)
    d["age65"] = (d.age >= 65).astype(float)
    d.loc[d.age.isna(), ["age55", "age60", "age65"]] = np.nan
    d["prev_miss_na"] = d.prev_miss.isna().astype(float); d["prev_miss"] = d.prev_miss.fillna(0)
    d["style_na"] = d["style"].isna().astype(float); d["style"] = d["style"].fillna(d["style"].mean())
    d["own_na"] = d.own.isna().astype(float); d["own"] = S3.wins(d.own.fillna(0))
    return d


def exit_types(ep):
    """Exit type from the Capital IQ text classification (events.csv), by (gvkey, execid, leftofc): planned / forced / other."""
    ev = pd.read_csv(DATA / "events.csv", parse_dates=["leftofc"], dtype={"gvkey": str, "execid": str})
    ev["gvkey"] = ev.gvkey.str.zfill(6)
    m = ep[["ep_id", "gvkey", "execid", "leftofc"]].merge(
        ev[["gvkey", "execid", "leftofc", "dep_type_rule"]].drop_duplicates(["gvkey", "execid", "leftofc"]),
        on=["gvkey", "execid", "leftofc"], how="left")
    t = m.dep_type_rule.fillna("unclassified")
    m["xtype"] = np.where(t == "retirement_planned", "planned",
                          np.where(t.isin(["resigned_abrupt", "terminated_forced"]), "forced", "other"))
    return m.drop_duplicates("ep_id").set_index("ep_id").xtype


def classify_sales(sales):
    s = sales.copy()
    s["ym"] = s.trandate.dt.year * 12 + s.trandate.dt.month - 1
    keys = set(zip(s.ep_id.values, s.ym.values))
    s["routine"] = [all((e, k - 12 * j) in keys for j in (1, 2, 3)) for e, k in zip(s.ep_id.values, s.ym.values)]
    return s


# ---------------------------------------------------------------- data
def prepare(db):
    ep = pd.read_pickle(DATA / "panel_episodes.pkl"); sales = pd.read_pickle(DATA / "panel_sales.pkl")
    g = pd.read_pickle(DATA / "panel_guid.pkl"); Q = pd.read_pickle(DATA / "panel_A_quarters.pkl")
    crsp, mkt, _ = G.load_crsp(db, sorted(g.permno.unique()), S3.START, S3.END)
    tab, tab2 = S3.market_table(crsp), vol_table(crsp)
    age_tab, own_tab = S3.load_age(db), load_own(db)
    cs = classify_sales(sales)
    print(f"sales: {len(cs):,}; routine share {cs.routine.mean():.1%}")
    si, si1m = PD.SaleIndex(sales), PD.SaleIndex(sales, 1e6)
    si_nr, si_r = PD.SaleIndex(cs[~cs.routine]), PD.SaleIndex(cs[cs.routine])
    b_all = S3.release_frame(g, ep, si, si1m)
    bn = S3.release_frame(g, ep, si_nr, si_r)
    assert len(bn) == len(b_all) and (bn.t.values == b_all.t.values).all()
    b_all["sale_nr"], b_all["sale_r"] = bn.sale.values, bn.sale_1m.values
    B = b_all[b_all.has_actual].reset_index(drop=True)
    B = S3.add_exec_controls(S3.add_market_controls(B, tab, "t"), ep, age_tab, "t")
    B = finish_controls(add_own(add_history(add_vol(B, tab2, "t"), g, "t"), ep, own_tab, "t"))
    B["gvkey"] = B.gvkey.astype(str)
    Q = Q.copy(); Q["permno"] = Q.permno.astype("int64"); Q["tq"] = Q.qs.dt.to_period("Q").astype(str)
    Q["opt_pos"] = (Q.opt > 0).astype(float)
    Q["reason"] = ep.set_index("ep_id").reason.reindex(Q.ep_id).values
    Q["salepost_nr"] = si_nr.any_in(Q.ep_id.values, Q.gdate.values, Q.qs.values)
    Q["salepost_r"] = si_r.any_in(Q.ep_id.values, Q.gdate.values, Q.qs.values)
    Q = S3.add_exec_controls(S3.add_market_controls(Q, tab, "qs"), ep, age_tab, "qs")
    Q = finish_controls(add_own(add_history(add_vol(Q, tab2, "qs"), g, "qs"), ep, own_tab, "qs"))
    Q["gvkey"] = Q.gvkey.astype(str)
    ret = Q.reason.astype(str).str.upper().eq("RETIRED")
    Q["exit_ret"] = ((Q.exit == 1) & ret).astype(int); Q["exit_oth"] = ((Q.exit == 1) & ~ret).astype(int)
    rb = B.reason.astype(str).str.upper().eq("RETIRED")
    B["X_ret"] = (B.exit_in & rb).astype(float); B["X_oth"] = (B.exit_in & ~rb).astype(float)
    xt = exit_types(ep)
    Q["xtype"] = xt.reindex(Q.ep_id).values
    B["xtype"] = xt.reindex(B.ep_id).values
    for nm in ("planned", "forced", "other"):
        Q[f"exit_{nm}"] = ((Q.exit == 1) & (Q.xtype == nm)).astype(int)
        B[f"X_{nm}"] = (B.exit_in & (B.xtype == nm)).astype(float)
    return ep, sales, g, Q, B, si


# ---------------------------------------------------------------- models
def fit(d, y, xs, terms):
    r = E.fe_ols(d, y, xs, firm="gvkey", year="tq", cluster="gvkey", cluster2="tq")
    out = {"n": r["n"], "n_clusters": r["n_clusters"]}
    for t in terms:
        if t in r:
            out[t] = r[t]
    return out


def p2_models(B, only=None):
    d = B.copy()
    d["y"] = S3.wins(d.miss.dropna() * 100).reindex(d.index)
    d["O"] = d.opt_pos.astype(float); d["S"] = d.sale.astype(float); d["X"] = d.exit_in.astype(float)
    d["OS"], d["OX"], d["SX"] = d.O * d.S, d.O * d.X, d.S * d.X; d["OSX"] = d.O * d.S * d.X
    d["ldays"], d["qtr"] = np.log(d.days_to_p.clip(lower=1)), (d.pdicity == "QTR").astype(float)
    base = ["O", "S", "X", "OS", "OX", "SX", "OSX"]
    res = {}
    if only is None or "P2_M0" in only:
        res["P2_M0"] = fit(d, "y", base + S3.P2_CTRL, ["OSX", "X", "SX", "S", "O"])
    if only is None or "P2_M1" in only:
        res["P2_M1"] = fit(d, "y", base + AUG_P2, ["OSX", "X", "SX", "S", "O"])
    if only is None or "P2_M2" in only:
        d["Snr"], d["Sr"] = d.sale_nr.astype(float), d.sale_r.astype(float)
        d["OS_nr"], d["SX_nr"], d["OSX_nr"] = d.O * d.Snr, d.Snr * d.X, d.O * d.Snr * d.X
        res["P2_M2"] = fit(d, "y", ["O", "Snr", "X", "OS_nr", "OX", "SX_nr", "OSX_nr", "Sr"] + AUG_P2, ["OSX_nr", "SX_nr", "Snr", "Sr"])
    if only is None or "P2_M3" in only:
        for c in ("OXr", "OXo", "SXr", "SXo", "OSXr", "OSXo"):
            pass
        d["OXr"], d["OXo"] = d.O * d.X_ret, d.O * d.X_oth
        d["SXr"], d["SXo"] = d.S * d.X_ret, d.S * d.X_oth
        d["OSXr"], d["OSXo"] = d.O * d.S * d.X_ret, d.O * d.S * d.X_oth
        res["P2_M3"] = fit(d, "y", ["O", "S", "X_ret", "X_oth", "OS", "OXr", "OXo", "SXr", "SXo", "OSXr", "OSXo"] + AUG_P2,
                           ["OSXr", "OSXo", "SXr", "SXo", "X_ret", "X_oth"])
    return res


def p1_models(Q, only=None):
    d = Q.copy()
    d["O"] = d.opt_pos.astype(float); d["S"] = d.salepost.astype(float); d["OS"] = d.O * d.S
    d["sale12"] = d.sale12.astype(float)
    res = {}
    if only is None or "P1_M0" in only:
        d["y"] = d.exit * 100.0
        d["age_lin"] = d.age
        res["P1_M0"] = fit(d, "y", ["O", "S", "OS"] + S3.P1_CTRL, ["OS", "S", "O"])
    if only is None or "P1_M1" in only:
        d["y"] = d.exit * 100.0
        res["P1_M1"] = fit(d, "y", ["O", "S", "OS"] + AUG_P1, ["OS", "S", "O"])
    if only is None or "P1_M2" in only:
        d["y"] = d.exit * 100.0
        d["Snr"], d["Sr"] = d.salepost_nr.astype(float), d.salepost_r.astype(float); d["OS_nr"] = d.O * d.Snr
        res["P1_M2"] = fit(d, "y", ["O", "Snr", "OS_nr", "Sr"] + AUG_P1, ["OS_nr", "Snr", "Sr"])
    if only is None or "P1_M3_retired" in only:
        d["y"] = d.exit_ret * 100.0
        res["P1_M3_retired"] = fit(d, "y", ["O", "S", "OS"] + AUG_P1, ["OS", "S", "O"])
    if only is None or "P1_M3_other" in only:
        d["y"] = d.exit_oth * 100.0
        res["P1_M3_other"] = fit(d, "y", ["O", "S", "OS"] + AUG_P1, ["OS", "S", "O"])
    return res


INTEREST_3B = [("P1_M3b_planned", "OS"), ("P1_M3b_forced", "OS"), ("P2_M3b", "OSXp"), ("P2_M3b", "OSXf")]


def m3b_models(Q, B):
    res = {}
    d = Q.copy()
    d["O"] = d.opt_pos.astype(float)
    d["S"] = d.salepost.astype(float)
    d["OS"] = d.O * d.S
    d["sale12"] = d.sale12.astype(float)
    for nm in ("planned", "forced", "other"):
        d["y"] = d[f"exit_{nm}"] * 100.0
        res[f"P1_M3b_{nm}"] = fit(d, "y", ["O", "S", "OS"] + AUG_P1, ["OS", "S", "O"])
    d = B.copy()
    d["y"] = S3.wins(d.miss.dropna() * 100).reindex(d.index)
    d["O"] = d.opt_pos.astype(float)
    d["S"] = d.sale.astype(float)
    d["OS"] = d.O * d.S
    d["ldays"], d["qtr"] = np.log(d.days_to_p.clip(lower=1)), (d.pdicity == "QTR").astype(float)
    terms = ["O", "S", "OS"]
    for a, nm in (("p", "planned"), ("f", "forced"), ("o", "other")):
        x = d[f"X_{nm}"]
        d[f"X{a}"], d[f"OX{a}"], d[f"SX{a}"], d[f"OSX{a}"] = x, d.O * x, d.S * x, d.O * d.S * x
        terms += [f"X{a}", f"OX{a}", f"SX{a}", f"OSX{a}"]
    res["P2_M3b"] = fit(d, "y", terms + AUG_P2, ["OSXp", "OSXf", "OSXo", "SXp", "SXf", "SXo", "Xp", "Xf", "Xo"])
    return res


def show(res, interest_only=False):
    for mname, r in res.items():
        print(f"\n{mname}  (n {r['n']:,}, clusters {r['n_clusters']})")
        for k, v in r.items():
            if isinstance(v, dict):
                star = "  <-- interest" if (mname, k) in INTEREST else ""
                if interest_only and not star:
                    continue
                print(f"    {k:<8s} coef {v['coef']:+8.4f}  se {v['se']:.4f}  t {v['t']:+6.2f}{star}")


def half(df, date_col, which):
    y = df[date_col].dt.year
    return df[y <= SPLIT] if which == "explore" else df[y > SPLIT]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "explore"
    db = G.WRDS(refresh=False)
    ep, sales, g, Q, B, si = prepare(db)
    if mode == "explore":
        Qe, Be = half(Q, "qs", "explore"), half(B, "t", "explore")
        print(f"EXPLORATION 2004-{SPLIT}: CEO-quarters {len(Qe):,} (exits {int(Qe.exit.sum())}); releases {len(Be):,}")
        res = {**p1_models(Qe), **p2_models(Be)}
        show(res)
        cands = []
        for m, k in INTEREST:
            v = res[m][k]
            if abs(v["t"]) >= 2.5:
                cands.append({"model": m, "term": k, "explore_coef": v["coef"], "explore_t": v["t"], "sign": int(np.sign(v["coef"]))})
        (RESULTS / "stage4_explore.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        (RESULTS / "stage4_candidates.json").write_text(json.dumps(cands, indent=1), encoding="utf-8")
        print(f"\nCANDIDATES carried to confirmation (|t| >= 2.5): {cands if cands else 'none'}")
    elif mode == "confirm":
        p = RESULTS / "stage4_candidates.json"
        if not p.exists():
            sys.exit("Run `explore` first; candidates must be locked before confirmation.")
        cands = json.loads(p.read_text())
        if not cands:
            print("No candidates passed the exploration rule; nothing to confirm."); return
        need = sorted({c["model"] for c in cands})
        Qc, Bc = half(Q, "qs", "confirm"), half(B, "t", "confirm")
        res = {**p1_models(Qc, only=[m for m in need if m.startswith("P1")]), **p2_models(Bc, only=[m for m in need if m.startswith("P2")])}
        pv = {}
        out = []
        untestable = []
        for c in cands:
            v = res[c["model"]].get(c["term"])
            if v is None:
                untestable.append(c)
                print(f"{c['model']}:{c['term']}  UNTESTABLE in the confirmation half (term not estimable: empty cell)")
                continue
            p = float(st.t.sf(v["t"] * c["sign"], res[c["model"]]["n_clusters"] - 1))
            pv[f"{c['model']}:{c['term']}"] = p
            out.append({**c, "confirm_coef": v["coef"], "confirm_se": v["se"], "confirm_t": v["t"], "p_one_same_sign": p})
        adj = GT.holm(pv) if pv else {}
        for o in out:
            o["holm_p"] = adj[f"{o['model']}:{o['term']}"]
            print(f"{o['model']}:{o['term']}  explore {o['explore_coef']:+.4f} (t {o['explore_t']:+.2f}) -> confirm {o['confirm_coef']:+.4f} (t {o['confirm_t']:+.2f}), one-sided p {o['p_one_same_sign']:.4f}, Holm {o['holm_p']:.4f}")
        out += [{**c, "untestable": True} for c in untestable]
        (RESULTS / "stage4_confirm.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    elif mode == "explore3b":
        Qe, Be = half(Q, "qs", "explore"), half(B, "t", "explore")
        print(f"EXPLORATION 2004-{SPLIT} (M3b): exits by text type: {Qe[Qe.exit == 1].xtype.value_counts().to_dict()}")
        res = m3b_models(Qe, Be)
        for mname, r in res.items():
            print(f"\n{mname}  (n {r['n']:,}, clusters {r['n_clusters']})")
            for k, v in r.items():
                if isinstance(v, dict):
                    print(f"    {k:<8s} coef {v['coef']:+8.4f}  se {v['se']:.4f}  t {v['t']:+6.2f}" + ("  <-- interest" if (mname, k) in INTEREST_3B else ""))
        cands = [{"model": m, "term": k, "explore_coef": res[m][k]["coef"], "explore_t": res[m][k]["t"], "sign": int(np.sign(res[m][k]["coef"]))}
                 for m, k in INTEREST_3B if k in res[m] and abs(res[m][k]["t"]) >= 2.5]
        (RESULTS / "stage4_explore_m3b.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        (RESULTS / "stage4_candidates_m3b.json").write_text(json.dumps(cands, indent=1), encoding="utf-8")
        print(f"\nCANDIDATES carried to confirmation (|t| >= 2.5): {cands if cands else 'none'}")
    elif mode == "confirm3b":
        p = RESULTS / "stage4_candidates_m3b.json"
        if not p.exists():
            sys.exit("Run `explore3b` first.")
        cands = json.loads(p.read_text())
        if not cands:
            print("No M3b candidates passed the exploration rule; nothing to confirm.")
            return
        res = m3b_models(half(Q, "qs", "confirm"), half(B, "t", "confirm"))
        pv, out = {}, []
        for c in cands:
            v = res[c["model"]].get(c["term"])
            if v is None:
                print(f"{c['model']}:{c['term']} UNTESTABLE")
                continue
            pr = float(st.t.sf(v["t"] * c["sign"], res[c["model"]]["n_clusters"] - 1))
            pv[f"{c['model']}:{c['term']}"] = pr
            out.append({**c, "confirm_coef": v["coef"], "confirm_se": v["se"], "confirm_t": v["t"], "p_one_same_sign": pr})
        adj = GT.holm(pv) if pv else {}
        for o in out:
            o["holm_p"] = adj[f"{o['model']}:{o['term']}"]
            print(f"{o['model']}:{o['term']}  explore {o['explore_coef']:+.4f} (t {o['explore_t']:+.2f}) -> confirm {o['confirm_coef']:+.4f} (t {o['confirm_t']:+.2f}), one-sided p {o['p_one_same_sign']:.4f}, Holm {o['holm_p']:.4f}")
        (RESULTS / "stage4_confirm_m3b.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    elif mode == "m3b_replicate":
        # POST-HOC replication check of exploration observations (NOT a registered test, no selection rule): prints everything in M3b on 2015-2025
        res = m3b_models(half(Q, "qs", "confirm"), half(B, "t", "confirm"))
        print("POST-HOC REPLICATION on 2015-2025 (descriptive; not a registered test)")
        for mname, r in res.items():
            print(f"\n{mname}  (n {r['n']:,}, clusters {r['n_clusters']})")
            for k, v in r.items():
                if isinstance(v, dict):
                    print(f"    {k:<8s} coef {v['coef']:+8.4f}  se {v['se']:.4f}  t {v['t']:+6.2f}")
        (RESULTS / "stage4_m3b_replicate.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    elif mode == "profile":
        profile(B, ep, si)
    elif mode == "cells":      # treatment-cell sizes only (counts, no outcomes)
        for which in ("explore", "confirm"):
            d = half(B, "t", which)
            for nm, c in (("S&X_ret", d.sale & (d.X_ret == 1)), ("O&S&X_ret", (d.opt_pos == 1) & d.sale & (d.X_ret == 1)),
                          ("S&X_oth", d.sale & (d.X_oth == 1)), ("O&S&X_oth", (d.opt_pos == 1) & d.sale & (d.X_oth == 1))):
                print(f"{which:8s} {nm:<10s} releases {int(c.sum()):>5d} | firms {d.loc[c, 'gvkey'].nunique():>4d} | CEO episodes {d.loc[c, 'ep_id'].nunique():>4d}")


def profile(B, ep, si):
    e = ep[ep.leftofc.notna() & ~ep.deceased][["gvkey", "ep_id", "leftofc"]].copy()
    e["sold"] = si.any_in(e.ep_id.values, (e.leftofc - pd.Timedelta(days=365)).values, e.leftofc.values)
    x = B[["gvkey", "period_end"]].reset_index().rename(columns={"index": "_i"})
    m = x.merge(e, on="gvkey")
    m["d"] = (m.period_end - m.leftofc).dt.days
    m = m.assign(ad=m.d.abs()).sort_values("ad").drop_duplicates("_i")
    m["k"] = np.round(m.d / 91.25).astype(int)
    m = m[m.k.abs() <= 4]
    ks = np.full(len(B), 99); ss = np.full(len(B), -1)
    ks[m._i.values] = m.k.values; ss[m._i.values] = m.sold.astype(int).values
    d = B.copy()
    d["y"] = S3.wins(d.miss.dropna() * 100).reindex(d.index)
    d["ldays"], d["qtr"] = np.log(d.days_to_p.clip(lower=1)), (d.pdicity == "QTR").astype(float)
    cols = []
    for s in (0, 1):
        for k in range(-4, 5):
            c = f"k{k:+d}_{'sold' if s else 'nosale'}"
            d[c] = ((ks == k) & (ss == s)).astype(float); cols.append(c)
    print(f"releases within 4 quarters of a CEO exit: {int((ks != 99).sum()):,} (preceded by a sale: {int(((ss == 1) & (ks != 99)).sum()):,})")
    rows = {}
    for which in ("explore", "confirm"):
        dd = half(d, "t", which)
        r = E.fe_ols(dd, "y", cols + S3.P2_CTRL, firm="gvkey", year="tq", cluster="gvkey", cluster2="tq")
        rows[which] = {c: r[c] for c in cols if c in r}
    print("\nMiss (pp of price) by quarters from the period end to the nearest CEO exit (k<0: period ended BEFORE the exit; k>0: AFTER)")
    print(f"{'k':>3s} | {'no sale before exit':^34s} | {'sale in prior 365d':^34s}")
    for which in ("explore", "confirm"):
        print(f"--- {which} half ({'<=' if which == 'explore' else '>'} {SPLIT})")
        for k in range(-4, 5):
            line = []
            for tag in ("nosale", "sold"):
                v = rows[which].get(f"k{k:+d}_{tag}")
                line.append(f"{v['coef']:+7.3f} (t {v['t']:+5.1f})" if v else "      n/a        ")
            print(f"{k:+3d} | {line[0]:^34s} | {line[1]:^34s}")
    (RESULTS / "stage4_profile.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
