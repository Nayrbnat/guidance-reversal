"""
gg_panel_data.py -- data layer for Stage 3 (executive panel, guidance-release panel, stayers).

    python -u gg_panel_data.py feasibility      # builds + caches the panels, prints COUNTS ONLY (no outcomes)

Builds (all cached to outputs/data/panel_*.pkl):
  episodes   one row per CEO episode (gvkey, execid): start, end, leftofc, reason, name, age
  sales      matched open-market sales per episode (strict name match, score 3)
  guid       guidance releases with permno, gvkey, CEO-of-record episode, optimism, miss, price, mcap
  quarters   CEO-episode x quarter panel (Panel A) with lagged optimism / sale features and exit flag
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg_labels as GL
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS
START, END = 2004, 2025
D_END = pd.Timestamp(f"{END}-12-31")


# ---------------------------------------------------------------- episodes
def build_episodes(ec: pd.DataFrame) -> pd.DataFrame:
    d = ec[(ec.year >= START - 1)].copy()
    d["leftofc"] = pd.to_datetime(d.leftofc, errors="coerce")
    d["becameceo"] = pd.to_datetime(d.becameceo, errors="coerce")
    d["is_ceo"] = d.ceoann.eq("CEO") | d.pceo.eq("CEO")
    d = d[d.is_ceo].sort_values("year")
    g = d.groupby(["gvkey", "execid"])
    ep = g.agg(first_year=("year", "min"), last_year=("year", "max"), name=("exec_fullname", "last"),
               leftofc=("leftofc", "max"), becameceo=("becameceo", "min"), reason=("reason", "last")).reset_index()
    ep["gvkey"] = ep.gvkey.astype(str).str.zfill(6)
    j1 = pd.to_datetime(ep.first_year.astype(int).astype(str) + "-01-01")
    ep["start"] = j1.where(ep.becameceo.isna() | (ep.becameceo < j1), ep.becameceo)
    last = pd.to_datetime(ep.last_year.clip(upper=END).astype(int).astype(str) + "-12-31")
    ep["end"] = ep.leftofc.where(ep.leftofc.notna(), last)
    ep["censored"] = ep.leftofc.isna()
    ep["deceased"] = ep.reason.astype(str).str.upper().eq("DECEASED")
    ep["ep_id"] = np.arange(len(ep))
    return ep[ep.end >= pd.Timestamp(f"{START}-01-01")].reset_index(drop=True)


def attach_permno(df: pd.DataFrame, link: pd.DataFrame, date_col: str, by_gvkey=True) -> pd.DataFrame:
    """Add permno (CCM link valid at date_col) to a frame with gvkey; or gvkey to a frame with permno."""
    key = "gvkey" if by_gvkey else "permno"
    out = df.reset_index(drop=True).copy()
    out["_i"] = np.arange(len(out))
    m = out[["_i", key, date_col]].merge(link, on=key)
    m = m[(m[date_col] >= m.linkdt) & (m[date_col] <= m.linkenddt)].drop_duplicates("_i")
    other = "permno" if by_gvkey else "gvkey"
    out[other] = out._i.map(m.set_index("_i")[other])
    return out.drop(columns="_i")


def get_link(db) -> pd.DataFrame:
    lk = db.cached("ccm_link", lambda: db.fetch(
        "crsp", "ccmxpf_lnkhist", ["gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"],
        [("linktype", "in", ["LU", "LC"]), ("linkprim", "in", ["P", "C"])]))
    lk = lk.rename(columns={"lpermno": "permno"}).dropna(subset=["permno"])
    lk["gvkey"] = lk.gvkey.astype(str).str.zfill(6)
    lk["permno"] = lk.permno.astype(int)
    lk["linkdt"] = pd.to_datetime(lk.linkdt)
    lk["linkenddt"] = pd.to_datetime(lk.linkenddt, errors="coerce").fillna(pd.Timestamp("2099-12-31"))
    return lk[["gvkey", "permno", "linkdt", "linkenddt"]]


def ceo_of_record(g: pd.DataFrame, ep: pd.DataFrame, date_col: str) -> pd.Series:
    """ep_id of the CEO episode containing date_col for the row's gvkey (later start wins), else NaN."""
    x = g[["gvkey", date_col]].reset_index(drop=True)
    x["_i"] = np.arange(len(x))
    m = x.merge(ep[["gvkey", "ep_id", "start", "end"]], on="gvkey")
    m = m[(m[date_col] >= m.start) & (m[date_col] <= m.end)].sort_values("start").drop_duplicates("_i", keep="last")
    res = np.full(len(g), np.nan)
    res[m._i.values] = m.ep_id.values
    return pd.Series(res, index=g.index)


# ---------------------------------------------------------------- sales
def match_sales(ep: pd.DataFrame, ins: pd.DataFrame, link: pd.DataFrame, crsp) -> pd.DataFrame:
    s = ins[ins.acqdisp == "D"].copy()
    s["tprice"] = s.tprice.where(s.tprice > 0)
    s["tprice"] = s.tprice.fillna(G.price_at(crsp, s, "trandate"))
    s["val"] = s.shares.abs() * s.tprice
    by_p = {p: x for p, x in s.groupby("permno")}
    perms = link.groupby("gvkey").permno.apply(lambda v: sorted(set(v))).to_dict()
    rows = []
    for r in ep.itertuples():
        frames = [by_p[p] for p in perms.get(r.gvkey, []) if p in by_p]
        if not frames:
            continue
        ss = pd.concat(frames)
        p = GL.parse_name(r.name)
        sc = {o: GL.match_parsed(p, o) for o in ss.owner.dropna().unique()}
        mine = ss[ss.owner.map(sc).fillna(0) >= 3]
        if len(mine):
            rows.append(pd.DataFrame({"ep_id": r.ep_id, "trandate": mine.trandate.values, "val": mine.val.values}))
    out = pd.concat(rows) if rows else pd.DataFrame(columns=["ep_id", "trandate", "val"])
    out["trandate"] = pd.to_datetime(out.trandate).astype("datetime64[ns]")
    return out.sort_values(["ep_id", "trandate"]).reset_index(drop=True)


class SaleIndex:
    """Fast 'any sale (value >= thr) by episode in (a, b]' queries."""

    def __init__(self, sales: pd.DataFrame, thr=0.0):
        s = sales[sales.val.fillna(0) > thr]
        self.d = {k: v.trandate.values.astype("datetime64[ns]") for k, v in s.groupby("ep_id")}

    def any_in(self, ep_ids, a, b) -> np.ndarray:
        a = np.asarray(a, dtype="datetime64[ns]"); b = np.asarray(b, dtype="datetime64[ns]")
        out = np.zeros(len(a), bool)
        for i, (e, lo, hi) in enumerate(zip(ep_ids, a, b)):
            d = self.d.get(e)
            if d is None or np.isnat(lo) or np.isnat(hi):
                continue
            out[i] = np.searchsorted(d, hi, side="right") - np.searchsorted(d, lo, side="right") > 0
        return out


# ---------------------------------------------------------------- guidance panel
def build_guid(gd: pd.DataFrame, link: pd.DataFrame, ep: pd.DataFrame, crsp) -> pd.DataFrame:
    g = gd.rename(columns={"anndats": "t"}).copy()
    g = attach_permno(g, link, "t", by_gvkey=False)
    g = g.dropna(subset=["gvkey"]).reset_index(drop=True)
    g["permno"] = g.permno.astype(int)
    g["t"] = g.t.astype("datetime64[ns]")
    g["ep_id"] = ceo_of_record(g, ep, "t")
    g["t_m1"] = g.t - pd.Timedelta(days=1)          # price BEFORE the release (it contains no reaction)
    g["px"] = G.price_at(crsp, g, "t_m1").values
    g["opt"] = (g["mid"] - g["consensus"]) / g.px
    g["miss"] = (g["actual"] - g["mid"]) / g.px
    g["surp"] = (g["actual"] - g["consensus"]) / g.px
    for c in ["opt", "miss", "surp"]:
        g.loc[g[c].abs() > G.MAX_GAP, c] = np.nan
    g["days_to_p"] = (g.period_end - g.t).dt.days
    return g


def main_feasibility():
    db = G.WRDS(refresh=False)
    ec = pd.read_pickle(G.CACHE / "execucomp_2004_2025.pkl")
    ec["gvkey"] = ec.gvkey.astype(str).str.zfill(6)
    ep = build_episodes(ec)
    print(f"CEO episodes (2003+): {len(ep):,} | with leftofc {int(ep.leftofc.notna().sum()):,} | deceased {int(ep.deceased.sum())} | firms {ep.gvkey.nunique():,}")
    link = get_link(db)
    perms = sorted(link[link.gvkey.isin(ep.gvkey.unique())].permno.unique())
    print(f"linked permnos for CEO firms: {len(perms):,}")

    gd = G.load_guidance(db, perms, START, END)
    print(f"guidance rows (EPS, linked): {len(gd):,}; with actual {int(gd.actual.notna().sum()):,}; with consensus {int(gd.consensus.notna().sum()):,}")
    gperms = sorted(gd.permno.unique())
    crsp, mkt, names = G.load_crsp(db, gperms, START, END)
    ins = G.load_insiders(db, gperms, names, START, END)
    print(f"insider sale/purchase rows pulled: {len(ins):,}")

    sales = match_sales(ep[ep.gvkey.isin(link[link.permno.isin(gperms)].gvkey)].copy(), ins, link, crsp)
    sales.to_pickle(DATA / "panel_sales.pkl"); ep.to_pickle(DATA / "panel_episodes.pkl")
    print(f"matched sales: {len(sales):,} rows over {sales.ep_id.nunique():,} episodes")

    g = build_guid(gd, link, ep, crsp)
    g.to_pickle(DATA / "panel_guid.pkl")
    ok = g.ep_id.notna()
    print(f"guidance releases with a CEO of record: {int(ok.sum()):,} of {len(g):,} ({ok.mean():.0%})")

    si = SaleIndex(sales)
    si1m = SaleIndex(sales, 1e6)
    ep_i = ep.set_index("ep_id")

    # ---- Panel B feasibility (sample + treatment cells only; outcomes not looked at)
    b = g[ok & g.period_end.gt(g.t) & g.days_to_p.le(400) & g.period_end.le(D_END) & g.t.between(f"{START}-01-01", D_END)
          & g.mid.notna() & g.consensus.notna() & g.actual.notna() & g.opt.notna() & g.miss.notna() & (g.pdicity != "SAN")].copy()
    b["ep_id"] = b.ep_id.astype(int)
    left = ep_i.leftofc.reindex(b.ep_id).values
    dec = ep_i.deceased.reindex(b.ep_id).values
    exitd = pd.to_datetime(left)
    b["exit_in"] = (exitd > b.t.values) & (exitd <= b.period_end.values) & ~dec
    stop = np.minimum(b.period_end.values, np.where(pd.isna(exitd), np.datetime64("2100-01-01", "ns"), exitd.values.astype("datetime64[ns]")))
    b["sale"] = si.any_in(b.ep_id.values, b.t.values, stop)
    b["sale_1m"] = si1m.any_in(b.ep_id.values, b.t.values, stop)
    b["sale_pre"] = si.any_in(b.ep_id.values, (b.t - pd.Timedelta(days=180)).values, b.t.values)
    b["opt_pos"] = b.opt > 0
    b.to_pickle(DATA / "panel_B_sample.pkl")
    print("\n=== Panel B (guidance releases) ===")
    print(f"releases {len(b):,} | firms {b.gvkey.nunique():,} | CEO episodes {b.ep_id.nunique():,} | annual {int((b.pdicity=='ANN').sum()):,} quarterly {int((b.pdicity=='QTR').sum()):,}")
    print(f"SALE=1 {int(b.sale.sum()):,} ({b.sale.mean():.1%}) | SALE>=1m {int(b.sale_1m.sum()):,} | EXIT=1 {int(b.exit_in.sum()):,} ({b.exit_in.mean():.1%}) | SALE&EXIT {int((b.sale&b.exit_in).sum()):,}")
    print(f"opt>0: {b.opt_pos.mean():.0%} | SALE&EXIT&opt>0 {int((b.sale&b.exit_in&b.opt_pos).sum()):,} | SALE&EXIT firms {b[b.sale&b.exit_in].gvkey.nunique()}")
    fx = b.groupby("gvkey").exit_in.agg(["min", "max"])
    print(f"firms with EXIT variation (both 0 and 1 among releases): {int(((fx['min']==0)&(fx['max']==1)).sum())} | firms with any exit {int((fx['max']==1).sum())}")
    print(f"releases in firms w/ exit variation: {int(b.gvkey.isin(fx[(fx['min']==0)&(fx['max']==1)].index).sum()):,}")
    print(f"pre-guidance sale (placebo R11) {int(b.sale_pre.sum()):,}")

    # ---- Panel A feasibility
    q = build_quarters(ep, link, g, si, si1m)
    q.to_pickle(DATA / "panel_A_quarters.pkl")
    print("\n=== Panel A (CEO x quarter, guiders only) ===")
    print(f"CEO-quarters {len(q):,} | CEOs {q.ep_id.nunique():,} | firms {q.gvkey.nunique():,} | exits {int(q.exit.sum()):,} ({q.exit.mean():.2%}/quarter)")
    print(f"SALEPOST=1 {int(q.salepost.sum()):,} ({q.salepost.mean():.1%}) | SALEPOST & exit {int((q.salepost&q.exit.astype(bool)).sum()):,} | opt>0 {(q.opt>0).mean():.0%} | SALEPOST&opt>0 {int((q.salepost&(q.opt>0)).sum()):,}")
    fx = q.groupby("gvkey").exit.agg(["min", "max"]); nv = int(((fx['min'] == 0) & (fx['max'] == 1)).sum())
    print(f"firms with exit variation: {nv} | exits in those firms: {int(q[q.gvkey.isin(fx[(fx['min']==0)&(fx['max']==1)].index)].exit.sum())}")

    # ---- Panel C feasibility (stayers)
    q["pattern"] = (q.days_since_guid <= 180) & q.salepost
    st = q[q.pattern & ~q.exit.astype(bool)].copy()
    st = select_stayers(q)
    print("\n=== Panel C (L1 pattern) ===")
    print(f"stayer pseudo-events (thinned, observed >=365d after): {len(st):,} | execs {st.ep_id.nunique():,} | firms {st.gvkey.nunique():,} | by year: {st.qs.dt.year.value_counts().sort_index().to_dict()}")
    st.to_pickle(DATA / "panel_C_stayers.pkl")
    import json
    s2 = json.load(open(RESULTS / "stage2_results.json"))
    l1 = [x for x in s2["F1"] if x["name"] == "F1 L1 x U2"][0]
    print(f"leavers (Stage 2 L1 x U2) with windows: {l1['n_events']}")


def select_stayers(q: pd.DataFrame) -> pd.DataFrame:
    """Panel C stayers (Amendment 1): pattern L1 at the quarter start, no exit in the next 365 days, NO requirement to stay in
    ExecuComp (that would condition on firm survival); at most one pseudo-event per executive per 4 quarters; qs <= 2024-12-31."""
    q = q.copy()
    q["pattern"] = (q.days_since_guid <= 180) & q.salepost
    st = q[q.pattern & (q.qs <= pd.Timestamp("2024-12-31"))].copy()
    st = st[~st.leftofc.between(st.qs, st.qs + pd.Timedelta(days=365))].sort_values(["ep_id", "qs"])
    keep, last = [], {}
    for i, r in zip(st.index, st.itertuples()):
        if r.ep_id not in last or (r.qs - last[r.ep_id]).days >= 365:
            keep.append(i); last[r.ep_id] = r.qs
    return st.loc[keep]


def build_quarters(ep, link, g, si, si1m) -> pd.DataFrame:
    rows = []
    qstarts = pd.date_range(f"{START}-01-01", f"{END}-10-01", freq="QS")
    for r in ep.itertuples():
        qs = qstarts[(qstarts >= r.start) & (qstarts < r.end)]
        if len(qs) == 0:
            continue
        rows.append(pd.DataFrame({"ep_id": r.ep_id, "gvkey": r.gvkey, "qs": qs, "start": r.start, "end": r.end,
                                  "leftofc": r.leftofc, "deceased": r.deceased}))
    q = pd.concat(rows, ignore_index=True)
    q["qe"] = q.qs + pd.offsets.QuarterBegin(startingMonth=1)
    q["exit"] = (q.leftofc >= q.qs) & (q.leftofc < q.qe)
    q = q[~(q.exit & q.deceased)].copy()
    q["exit"] = q["exit"].astype(int)
    q = attach_permno(q, link, "qs", by_gvkey=True).dropna(subset=["permno"])
    q["permno"] = q.permno.astype(int)
    gg = g[g.opt.notna()][["permno", "t", "opt"]].sort_values("t").rename(columns={"t": "gdate"})
    gg["gdate"] = gg.gdate.astype("datetime64[ns]")
    q = q.sort_values("qs")
    q["qs"] = q.qs.astype("datetime64[ns]")
    q = pd.merge_asof(q, gg, left_on="qs", right_on="gdate", by="permno", direction="backward", allow_exact_matches=False)
    q["days_since_guid"] = (q.qs - q.gdate).dt.days
    q = q[q.days_since_guid <= 365].copy()
    q["salepost"] = si.any_in(q.ep_id.values, q.gdate.values, q.qs.values)
    q["salepost_1m"] = si1m.any_in(q.ep_id.values, q.gdate.values, q.qs.values)
    q["sale12"] = si.any_in(q.ep_id.values, (q.qs - pd.Timedelta(days=365)).values, q.qs.values)
    return q.reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "feasibility":
        main_feasibility()
    else:
        print(__doc__)
