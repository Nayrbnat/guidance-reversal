"""Stream Compustat annual fundamentals (comp.funda; the WRDS REST API honours the date filter but not a firm filter) and keep only the standard-format
industrial annual records (indfmt INDL, datafmt STD, consol C, popsrc D) of the firms in the departure events file, with the few columns Stage 14 needs.
Resumable (checkpoint every 50 pages). NEW cache file only.

python scripts/pull_funda.py"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import guide_and_go as G

COLS = ["gvkey", "datadate", "fyear", "ceq", "at", "ib", "indfmt", "datafmt", "consol", "popsrc"]
ev = pd.read_csv(G.DATA / "events.csv", dtype={"gvkey": str})
gv = set(ev["gvkey"].astype(str).str.zfill(6))
print("firms in the events file:", len(gv), flush=True)
db = G.WRDS(refresh=True)
have = db.columns("comp", "funda")
print("columns present:", {c: (c in have) for c in ("ceq", "at", "ib", "indfmt", "datafmt", "consol", "popsrc", "fyear")}, flush=True)
ckpt, rows_pkl = G.CACHE / "funda_stream_ckpt.json", G.CACHE / "funda_stream_rows.pkl"
kept, nxt, pages, seen = [], None, 0, 0
if ckpt.exists() and rows_pkl.exists():
    st = json.loads(ckpt.read_text())
    nxt, pages, seen = st["next"], st["pages"], st["seen"]
    kept = pd.read_pickle(rows_pkl).to_dict("records")
    print(f"resuming at page {pages} ({seen:,} seen, {len(kept):,} kept)", flush=True)
if not (ckpt.exists() and json.loads(ckpt.read_text()).get("done")):
    url = db._url("comp", "funda")
    cur, first, t0 = nxt or url, nxt is None, time.time()
    while cur:
        js = db._get(cur, {"datadate__gte": "2001-01-01", "limit": db.limit} if first else None)
        first = False
        if js is None:
            sys.exit("empty response; rerun to resume")
        recs = js if isinstance(js, list) else js.get("results", [])
        seen += len(recs)
        for r in recs:
            if (str(r.get("gvkey", "")).zfill(6) in gv and r.get("indfmt") == "INDL" and r.get("datafmt") == "STD" and r.get("consol") == "C" and r.get("popsrc") == "D"):
                kept.append({c: r.get(c) for c in COLS})
        pages += 1
        cur = None if isinstance(js, list) else js.get("next")
        if pages % 50 == 0:
            pd.DataFrame(kept).to_pickle(rows_pkl)
            ckpt.write_text(json.dumps({"next": cur, "pages": pages, "seen": seen}))
            print(f"page {pages}: {seen:,} seen, {len(kept):,} kept, {(time.time() - t0) / 60:.0f} min", flush=True)
    pd.DataFrame(kept).to_pickle(rows_pkl)
    ckpt.write_text(json.dumps({"next": None, "pages": pages, "seen": seen, "done": True}))
df = pd.read_pickle(rows_pkl)
for c in ("ceq", "at", "ib"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
df = df.drop_duplicates(["gvkey", "datadate"]).sort_values(["gvkey", "datadate"]).reset_index(drop=True)
df.to_pickle(G.CACHE / "funda_events_2001_plus.pkl")
print(f"DONE: {seen:,} rows seen, {len(df):,} firm-years kept for {df.gvkey.nunique():,} firms; datadate {df.datadate.min().date()} to {df.datadate.max().date()}; ceq non-missing {df.ceq.notna().mean():.3f}", flush=True)
