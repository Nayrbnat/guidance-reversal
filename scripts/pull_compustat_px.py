"""Stream Compustat daily security prices (comp.secd; the WRDS REST API honours the date filter but NOT a firm filter, so every page of 2024-01-01 onward
comes back: about 14M rows) and keep only the firms with officer or director sales from 2024-06 to 2026-06, US-dollar common shares. Memory stays small and the
pull is resumable (checkpoint every 100 pages). Then build daily returns (src/gg_compustat_px.py) and validate them against the CRSP v2 returns on the 2024-2025
overlap. NEW cache files only; no outcomes.

python scripts/pull_compustat_px.py"""
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import guide_and_go as G
import gg_compustat_px as P
import gg_stage8 as S8

last = lambda p: sorted(glob.glob(str(G.CACHE / p)))[-1]
db = G.WRDS(refresh=True)

# firms with officer or director sales 2024-06 .. 2026-06 (old cached insiders plus the streamed 2026 rows)
old = pd.read_pickle(last("tr_insiders_2004_2025_*.pkl"))
new = pd.read_pickle(G.CACHE / "tr_insiders_2025_06_plus.pkl")
new["trandate"] = pd.to_datetime(new["trandate"], errors="coerce")
for df in (old, new):
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["tprice"] = pd.to_numeric(df["tprice"], errors="coerce")
add = new[(new.trandate >= "2026-01-01") & new.cleanse.isin(["R", "H"])]
ins = pd.concat([old, add[[c for c in old.columns if c in add.columns]]], ignore_index=True)
names = pd.read_pickle(G.CACHE / "ciz_names_merged.pkl")
names.loc[names["nameenddt"] == pd.Timestamp("2025-12-31"), "nameenddt"] = pd.Timestamp("2026-12-31")
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
for c in ("linkdt", "linkenddt"):
    link[c] = pd.to_datetime(link[c], errors="coerce")
d = S8.insider_sale_days(ins, names, link, min_value=100_000.0, start="2024-06-01", end="2026-06-30")
gv = set(d["gvkey"].astype(str).str.zfill(6))
print("firms with officer or director sales 2024-06..2026-06:", len(gv), flush=True)
lk = link[link["linkprim"].isin(["P", "C"]) & link["linktype"].isin(["LC", "LU"])].copy()
lk["gvkey"] = lk["gvkey"].astype(str).str.zfill(6)
lk = lk.sort_values("linkdt").drop_duplicates("gvkey", keep="last")
primary = dict(zip(lk["gvkey"], lk["liid"].astype(str))) if "liid" in lk else {}
print("primary issue known for", sum(g in primary for g in gv), "of", len(gv), "firms", flush=True)

# stream comp.secd from 2024-01-01, keep only needed firms
COLS = ["gvkey", "iid", "datadate", "prccd", "ajexdi", "trfd", "cshoc", "tpci", "curcdd"]
ckpt, rows_pkl = G.CACHE / "secd_stream_ckpt.json", G.CACHE / "secd_stream_rows.pkl"
kept, nxt, pages, seen = [], None, 0, 0
if ckpt.exists() and rows_pkl.exists():
    st = json.loads(ckpt.read_text())
    nxt, pages, seen = st["next"], st["pages"], st["seen"]
    kept = pd.read_pickle(rows_pkl).to_dict("records")
    print(f"resuming at page {pages} ({seen:,} rows seen, {len(kept):,} kept)", flush=True)
done = ckpt.exists() and json.loads(ckpt.read_text()).get("done")
if not done:
    url = db._url("comp", "secd")
    params = {"datadate__gte": "2024-01-01", "limit": db.limit}
    cur, first, t0 = nxt or url, nxt is None, time.time()
    while cur:
        js = db._get(cur, params if first else None)
        first = False
        if js is None:
            sys.exit("empty response; rerun to resume")
        recs = js if isinstance(js, list) else js.get("results", [])
        seen += len(recs)
        for r in recs:
            if str(r.get("gvkey", "")).zfill(6) in gv:
                kept.append({c: r.get(c) for c in COLS})
        pages += 1
        cur = None if isinstance(js, list) else js.get("next")
        if pages % 100 == 0:
            pd.DataFrame(kept).to_pickle(rows_pkl)
            ckpt.write_text(json.dumps({"next": cur, "pages": pages, "seen": seen}))
            print(f"page {pages}: {seen:,} rows seen, {len(kept):,} kept, {(time.time() - t0) / 60:.0f} min", flush=True)
    pd.DataFrame(kept).to_pickle(rows_pkl)
    ckpt.write_text(json.dumps({"next": None, "pages": pages, "seen": seen, "done": True}))
raw = pd.read_pickle(rows_pkl)
print(f"stream complete: {seen:,} rows seen, {len(raw):,} kept", flush=True)
raw["datadate"] = pd.to_datetime(raw["datadate"], errors="coerce")
raw = raw[raw["datadate"] <= pd.Timestamp.today().normalize()]              # a few stray future-dated rows exist in the table
px = P.secd_to_daily(raw, primary_iid=primary if primary else None)
px.to_pickle(G.CACHE / "secd_px_2024_2026.pkl")
print("daily rows:", len(px), "| firms", px.gvkey.nunique(), "| dates", px.date.min().date(), "to", px.date.max().date(), flush=True)

# validation against CRSP v2 on the overlap
crsp = pd.read_pickle(G.CACHE / "ciz_dsf_2023_2025.pkl")
pg = lk.dropna(subset=["lpermno"]).assign(permno=lambda x: x.lpermno.astype("int64")).drop_duplicates("permno", keep="last").set_index("permno")["gvkey"]
crsp = crsp[crsp.date >= "2024-01-01"].assign(gvkey=lambda x: x.permno.map(pg)).dropna(subset=["gvkey"])
rep = P.overlap_vs_crsp(px, crsp[["gvkey", "date", "ret"]], tol=1e-3)
print("OVERLAP Compustat-built vs CRSP v2 returns, 2024-2025:", json.dumps(rep), flush=True)
m = px.merge(crsp[["gvkey", "date", "prc", "shrout"]], on=["gvkey", "date"], suffixes=("_s", "_c"))
lm = np.log(m.prc_s.abs() * m.shrout_s * 1000).sub(np.log(m.prc_c.abs() * m.shrout_c * 1000)).dropna()
rep["lmcap_abs_diff_median"] = float(lm.abs().median())
rep["lmcap_abs_diff_p95"] = float(lm.abs().quantile(0.95))
rep["by_year_share_within"] = {str(y): P.overlap_vs_crsp(px[px.date.dt.year == y], crsp[["gvkey", "date", "ret"]], tol=1e-3).get("share_within") for y in (2024, 2025)}
print("log market cap difference (Compustat vs CRSP): median |diff| %.4f, p95 %.4f" % (rep["lmcap_abs_diff_median"], rep["lmcap_abs_diff_p95"]), flush=True)
print("share within 0.001 by year:", rep["by_year_share_within"], flush=True)
(G.RESULTS / "secd_vs_crsp.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
