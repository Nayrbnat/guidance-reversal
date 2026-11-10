"""Stream the Thomson Reuters insider table (tfn.table1, 17.5M rows; the WRDS REST API ignores every filter on it) and keep only open-market sales
and purchases from 2025-06-01 on, so memory stays small. Resumable: the next-page URL and the kept rows are checkpointed. New cache file only.

python scripts/pull_insiders_2026.py"""
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

KEEP_FROM = "2025-06-01"
COLS = ["personid", "owner", "cusip6", "cusip2", "rolecode1", "trandate", "acqdisp", "trancode", "shares", "tprice", "cleanse"]
ckpt, rows_pkl = G.CACHE / "insiders_stream_ckpt.json", G.CACHE / "insiders_stream_rows.pkl"
db = G.WRDS(refresh=True)
url = db._url("tfn", "table1")
kept, nxt, pages, seen = [], None, 0, 0
if ckpt.exists() and rows_pkl.exists():
    st = json.loads(ckpt.read_text())
    nxt, pages, seen = st["next"], st["pages"], st["seen"]
    kept = pd.read_pickle(rows_pkl).to_dict("records")
    print(f"resuming at page {pages} ({seen:,} rows seen, {len(kept):,} kept)", flush=True)
t0 = time.time()
first = nxt is None
cur = nxt or url
while cur:
    js = db._get(cur, {"limit": db.limit} if first else None)
    first = False
    if js is None:
        print("empty response; stopping (rerun to resume)", flush=True)
        break
    recs = js if isinstance(js, list) else js.get("results", [])
    seen += len(recs)
    for r in recs:
        d = str(r.get("trandate") or "")[:10]
        if d >= KEEP_FROM and r.get("trancode") in ("S", "P"):
            kept.append({c: r.get(c) for c in COLS})
    pages += 1
    cur = None if isinstance(js, list) else js.get("next")
    if pages % 100 == 0:
        pd.DataFrame(kept).to_pickle(rows_pkl)
        ckpt.write_text(json.dumps({"next": cur, "pages": pages, "seen": seen}))
        print(f"page {pages}: {seen:,} rows seen, {len(kept):,} kept, {(time.time() - t0) / 60:.0f} min", flush=True)
df = pd.DataFrame(kept)
df.to_pickle(G.CACHE / "tr_insiders_2025_06_plus.pkl")
ckpt.write_text(json.dumps({"next": None, "pages": pages, "seen": seen}))
print(f"DONE: {seen:,} rows seen, {len(df):,} kept; trandate {df.trandate.min()} to {df.trandate.max()}", flush=True)
