"""Score the first filing of every phase with the frozen production scorer (one read each), in parallel shards.

python scripts/score_phases.py [workers]     (resumable: bundles come from the filings_text cache, reads from the shard caches)

Units are taken from the END of the list, so this runs without colliding with the bundle prefetch that works from the start.
Writes validation/transp_scores/shard*.jsonl (reads) and statuses.jsonl (bundle status per unit)."""
import hashlib
import json
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "src")
import pandas as pd
import guide_and_go as G
import gg_filings as F
import gg_transparency as T

MODEL, BATCH = "claude-haiku-5-5", 20
OUT = G.ROOT / "validation" / "transp_scores"
OUT.mkdir(parents=True, exist_ok=True)
FREEZE = G.RESULTS / "stage5_freeze.json"
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def variants(conm):
    c = re.sub(r"\s*-\s*CL\s*[A-Z]$", "", str(conm).strip(), flags=re.I)
    c = re.sub(r"\s*/\w+$", "", c)
    base = re.sub(r"[.,]?\s+(INC|CORP|CORPORATION|CO|COMPANY|LTD|PLC|LLC|LP|HOLDINGS?|GROUP|& CO)\.?$", "", c, flags=re.I)
    return sorted({c.title(), base.title()}, key=len, reverse=True)


PT_NAME = sys.argv[2] if len(sys.argv) > 2 else "phase_table.csv"          # extension run: python scripts/score_phases.py 4 phase_table_2025.csv phase_needed_all.csv
NEED_NAME = sys.argv[3] if len(sys.argv) > 3 else "phase_needed.csv"
pt = pd.read_csv(G.CACHE / PT_NAME, dtype={"gvkey": str, "accession": str, "phase_first": str})
need = pd.read_csv(G.CACHE / NEED_NAME, dtype=str)
firsts = pt[["gvkey", "phase_first"]].drop_duplicates().rename(columns={"phase_first": "accession"}).merge(need[["accession", "cik", "url", "filing_date"]], on="accession")
firsts = firsts.iloc[::-1].reset_index(drop=True)                     # from the end
ev = pd.read_csv(G.DATA / "events.csv", dtype={"gvkey": str})
ev["gvkey"] = ev["gvkey"].str.zfill(6)
names = dict(zip(ev["gvkey"], ev["conm"]))
missing = [g for g in set(firsts.gvkey) if g not in names]
if missing:
    co = G.WRDS().fetch_in("comp", "company", ["gvkey", "conm"], "gvkey", missing, [])
    co["gvkey"] = co["gvkey"].astype(str).str.zfill(6)
    names.update(dict(zip(co["gvkey"], co["conm"])))
db = G.WRDS()
doc_idx = F.build_filing_index(db, gvkeys=sorted(set(firsts.gvkey)), cache_dir=G.CACHE / "stage5_forms_docs", forms=("10-Q", "8-K"))
ua = F.sec_user_agent()
cache = G.CACHE / "filings_text"
cfg = T.measurement_config(MODEL)
T.assert_frozen(FREEZE, cfg)
runner = T.claude_runner(MODEL, tuple(cfg["cli_flags"]))
seed = G.ROOT / "validation" / "transp_stability" / "scores_cache.jsonl"
lock = threading.Lock()
status_path = OUT / "statuses.jsonl"
done_status = {}
if status_path.exists():
    for line in status_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        done_status[r["unit_id"]] = r["status"]
t0 = time.time()
counter = {"done": 0}
if len(sys.argv) > 2:                                                       # extension run: only units never handled before
    firsts = firsts[~firsts["accession"].isin(done_status)].reset_index(drop=True)
print(f"units {len(firsts)}, workers {workers}, already have status for {len(done_status)}", flush=True)


def bundle(r):
    row = pd.Series({"gvkey": r.gvkey, "accession": r.accession, "url": r.url, "cik": r.cik, "filing_date": pd.Timestamp(r.filing_date)})
    try:
        text, status, _ = F.load_bundle(row, doc_idx, variants(names.get(r.gvkey, "")), cache, ua)
    except Exception as e:
        return "", f"error:{type(e).__name__}"
    return text, status


def shard(i):
    cp = OUT / f"shard{i}.jsonl"
    if not cp.exists() and seed.exists():
        shutil.copy(seed, cp)
    mine = [r for k, r in enumerate(firsts.itertuples()) if k % workers == i]
    for b in range(0, len(mine), BATCH):
        batch = mine[b:b + BATCH]
        got = [(r, *bundle(r)) for r in batch]
        with lock:
            with open(status_path, "a", encoding="utf-8") as fh:
                for r, text, st in got:
                    fh.write(json.dumps({"unit_id": r.accession, "status": st}) + "\n")
        ok = [(r, text) for r, text, st in got if st in ("ok", "limited_disclosure")]
        with lock:                                  # the hash of the exact text this pass scored: the assembler and the run check scores against it
            with open(OUT / "current_hashes.jsonl", "a", encoding="utf-8") as fh:
                for r, text in ok:
                    fh.write(json.dumps({"unit_id": r.accession, "sha": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
        if ok:
            units = pd.DataFrame({"unit_id": [r.accession for r, _ in ok], "text": [t for _, t in ok]})
            try:
                T.score_reads(units, cfg=cfg, cache_path=cp, runner=runner, reads=(1,), require_frozen=FREEZE)
            except Exception as e:
                print(f"shard {i} batch {b}: {type(e).__name__} {str(e)[:150]}", flush=True)
        with lock:
            counter["done"] += len(batch)
            print(f"  {counter['done']}/{len(firsts)} units handled ({(time.time() - t0) / 60:.0f} min), tokens in {sum(u['in'] for u in runner.usage):,}", flush=True)


with ThreadPoolExecutor(max_workers=workers) as ex:
    list(ex.map(shard, range(workers)))
print("finished scoring pass", f"({(time.time() - t0) / 60:.0f} min)", flush=True)
