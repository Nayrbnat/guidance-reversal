"""Batch runner for the revenue-split change check (no model tokens).

For every 10-K in a list it downloads the filing, reads the revenue-breakdown labels (gg_phases.revenue_labels), keeps ONLY the
label list and discards the HTML, and appends one JSON line per filing. Resumable: finished filings are skipped on a rerun, failed
downloads are not recorded so a rerun retries them. change_levels() then compares each firm's consecutive filings."""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import gg_filings as F
import gg_phases as P
import gg_transparency as T


def label_filing(row, fetch, user_agent: str) -> dict:
    html = fetch(F._filing_url(pd.Series({"url": row["url"], "cik": row["cik"], "accession": row["accession"]})), user_agent)
    labels = P.revenue_labels(T.select_docs(html, ("10-K", "EX-13")))
    return {"accession": row["accession"], "gvkey": row["gvkey"], "filing_date": str(row["filing_date"])[:10], "labels": labels}


def _done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["accession"])
        except (ValueError, KeyError):
            continue            # a half-written last line after a crash is simply redone
    return done


def run_labels(rows: pd.DataFrame, out_path, fetch=None, user_agent: str = "", workers: int = 4, min_interval: float = 0.12,
               log_every: int = 250) -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fetch = fetch or F._http_get
    done = _done(out_path)
    todo = [r for _, r in rows.iterrows() if r["accession"] not in done]
    stats = {"skipped": len(rows) - len(todo), "done": 0, "failed": 0}
    lock, gate = threading.Lock(), threading.Lock()
    last = [0.0]
    t0 = time.time()

    def work(r):
        with gate:                                  # global request rate, whatever the worker count
            wait = min_interval - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
        try:
            rec = label_filing(r, fetch, user_agent)
        except Exception as exc:                    # network, 5xx, parse: leave it for the next run
            with lock:
                stats["failed"] += 1
                with open(out_path.with_suffix(".errors.log"), "a", encoding="utf-8") as fh:
                    fh.write(f"{r['accession']}\t{type(exc).__name__}\t{str(exc)[:120]}\n")
            return
        with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            stats["done"] += 1
            if log_every and stats["done"] % log_every == 0:
                print(f"{stats['done']}/{len(todo)} done, {stats['failed']} failed, {(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    return stats


def change_levels(df: pd.DataFrame) -> pd.DataFrame:
    """One row per filing: level FIRST / NONE / MINOR / MAJOR / UNKNOWN against the firm's previous filing that had labels."""
    out = []
    for gv, g in df.sort_values(["gvkey", "filing_date"]).groupby("gvkey"):
        prev = None
        for r in g.itertuples():
            labels = list(r.labels)
            if prev is None:
                c = {"level": "FIRST" if labels else "UNKNOWN", "similarity": None, "kept_old": None, "kept_new": None}
            else:
                c = P.compare_labels(prev, labels)
            out.append({"gvkey": gv, "accession": r.accession, "filing_date": r.filing_date, "n_labels": len(labels), **{k: c[k] for k in ("level", "similarity", "kept_old", "kept_new")}})
            if labels:
                prev = labels
    return pd.DataFrame(out)


def triage_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """The MAJOR changes that need the small model check: one row per such filing with the previous and current label lists
    (the previous list is the firm's last filing that had labels, as in change_levels)."""
    lv = change_levels(df).set_index("accession")
    out = []
    for gv, g in df.sort_values(["gvkey", "filing_date"]).groupby("gvkey"):
        prev = None
        for r in g.itertuples():
            labels = list(r.labels)
            if prev is not None and lv.loc[r.accession, "level"] == "MAJOR":
                out.append({"gvkey": gv, "accession": r.accession, "filing_date": r.filing_date, "prev": prev, "cur": labels})
            if labels:
                prev = labels
    return pd.DataFrame(out, columns=["gvkey", "accession", "filing_date", "prev", "cur"])


def phase_table(df: pd.DataFrame, triage: dict) -> pd.DataFrame:
    """Phase number per filing. `triage` maps accession -> True (really changed), False (not), or None (unusable answer).
    A new phase starts at a MAJOR filing only when its triage answer is True; MAJOR filings with no answer yet are flagged pending
    and do not split. phase_first is the accession a phase is scored from."""
    lv = change_levels(df)
    rows = []
    for gv, g in lv.groupby("gvkey", sort=False):
        phase, first = 0, None
        for r in g.sort_values("filing_date", kind="stable").itertuples():     # stable: same-day filings keep change_levels' order
            major = r.level == "MAJOR"
            ans = triage.get(r.accession, "missing") if major else "n/a"
            if major and ans is True:
                phase, first = phase + 1, r.accession
            if first is None:
                first = r.accession
            rows.append({"gvkey": gv, "accession": r.accession, "filing_date": r.filing_date, "level": r.level, "phase": phase, "phase_first": first,
                         "triage_pending": bool(major and ans == "missing"), "triage_unusable": bool(major and ans is None)})
    return pd.DataFrame(rows)


def run_triage(pairs: pd.DataFrame, out_path, runner, workers: int = 3, sleeper=None, wait_seconds: int = 900, max_waits: int = 96,
               log_every: int = 100) -> dict:
    """Ask the small model about every MAJOR pair (gg_phases.triage_change). One JSON line per answered pair; a timeout or other
    failure is not stored, so a rerun retries it. A usage limit waits and retries the same pair."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sleeper = sleeper or time.sleep
    done = _done(out_path)
    todo = [r for _, r in pairs.iterrows() if r["accession"] not in done]
    stats = {"skipped": len(pairs) - len(todo), "done": 0, "failed": 0}
    lock = threading.Lock()

    def work(r):
        for _ in range(max_waits):
            try:
                ans = P.triage_change(list(r["prev"]), list(r["cur"]), runner)
                if ans["major"] is None:                        # unusable reply: one more try before recording it as unusable
                    ans = P.triage_change(list(r["prev"]), list(r["cur"]), runner)
                break
            except T.UsageLimit:
                sleeper(wait_seconds)
            except Exception as exc:                            # timeout or CLI error: leave for the next run
                with lock:
                    stats["failed"] += 1
                    with open(out_path.with_suffix(".errors.log"), "a", encoding="utf-8") as fh:
                        fh.write(f"{r['accession']}\t{type(exc).__name__}\t{str(exc)[:120]}\n")
                return
        else:
            with lock:
                stats["failed"] += 1
            return
        with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"accession": r["accession"], "gvkey": r["gvkey"], **ans}) + "\n")
            stats["done"] += 1
            if log_every and stats["done"] % log_every == 0:
                print(f"triage {stats['done']}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    return stats


def load_triage(out_path) -> dict:
    """accession -> True / False / None, in the form phase_table takes."""
    out_path = Path(out_path)
    if not out_path.exists():
        return {}
    res = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            res[rec["accession"]] = rec["major"]
        except (ValueError, KeyError):
            continue
    return res


def choose_reads(reads: pd.DataFrame, current_hashes: dict, status: dict, config_hash: str) -> pd.DataFrame:
    """One scored row per unit that has a bundle status. A unit's score is the read made on the CURRENT bundle text (its
    input_sha256 equals the current hash), under the frozen config; a read on older text is never used (stale_input), a failed
    bundle keeps no score whatever the cache holds, and a unit never read is not_scored. If a unit has several reads on the
    current text (a resumed run), the latest parseable one is used."""
    reads = reads[reads["config_hash"] == config_hash]
    by_unit = {u: g for u, g in reads.groupby("unit_id", sort=False)}
    rows = []
    for u, st in status.items():
        cur = current_hashes.get(u)
        row = {"accession": u, "t_final": float("nan"), "status": st, "config_hash": config_hash, "input_sha256": cur}
        if st in ("ok", "limited_disclosure"):
            g = by_unit.get(u)
            if g is None:
                row["status"] = "not_scored"
            else:
                m = g[g["input_sha256"] == cur]
                if m.empty:
                    row["status"] = "stale_input"
                else:
                    good = m.dropna(subset=["score"])
                    if len(good):
                        row["t_final"] = float(good["score"].iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows, columns=["accession", "t_final", "status", "config_hash", "input_sha256"])


def scores_from_phases(pt: pd.DataFrame, phase_scores: pd.DataFrame) -> pd.DataFrame:
    """Filing-level score table (the form stage5_scores.pkl takes): every filing gets the score of its phase's first filing.
    `phase_scores` has accession (the phase's first filing), t_final, status, config_hash, input_sha256. A phase with no score row
    stays NaN with status not_scored; a failed extraction keeps its own status."""
    ps = phase_scores.drop_duplicates("accession").set_index("accession")
    first = pt["phase_first"]
    out = pd.DataFrame({"accession": pt["accession"].values, "gvkey": pt["gvkey"].values, "phase": pt["phase"].values, "unit_id": first.values})
    out["t_final"] = first.map(ps["t_final"]).astype(float).values
    out["status"] = first.map(ps["status"]).fillna("not_scored").values
    out["config_hash"] = first.map(ps["config_hash"]).values
    out["input_sha256"] = first.map(ps["input_sha256"]).values
    out["config_hash"] = out["config_hash"].fillna(ps["config_hash"].dropna().iloc[0] if ps["config_hash"].notna().any() else None)
    return out


if __name__ == "__main__":
    # python src/gg_phase_run.py <needed_filings.csv> <out.jsonl> [workers]
    # python src/gg_phase_run.py triage <labels.jsonl> <triage.jsonl> [workers]
    import guide_and_go  # noqa: F401  (loads .env, which holds SEC_CONTACT_EMAIL)
    if sys.argv[1] == "triage":
        lab = pd.DataFrame([json.loads(l) for l in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines()])
        stats = run_triage(triage_pairs(lab), sys.argv[3], T.claude_runner("claude-haiku-5-5", T.DEFAULT_FLAGS),
                           workers=int(sys.argv[4]) if len(sys.argv) > 4 else 3)
        print("finished triage", stats, flush=True)
        sys.exit(0)
    rows = pd.read_csv(sys.argv[1], dtype=str)
    stats = run_labels(rows, sys.argv[2], user_agent=F.sec_user_agent(), workers=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    print("finished", stats, flush=True)
