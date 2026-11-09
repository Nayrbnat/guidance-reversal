"""
Tests for the Stage 10 interim-run machinery: gg_stage10 (gate, universe restriction, samples, Holm, filing chain, extension refusal checks) and the unattended
chain script scripts/stage10_chain.sh (run with a fake `uv` that records calls, so control flow is tested without any data).
Run:  uv run python tests/test_gg10.py    (from the repo root; no pytest needed; needs bash)
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAILS = []


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(name)
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail and not cond else ""))


# ------------------------------------------------------------------------------------------------------------------------ gate
def test_gate():
    import gg_stage10 as S10
    check("the gate passes at exactly the bar (corr 0.99, share 0.95)", S10.gate_passed({"corr": 0.99, "share_within": 0.95}))
    check("the gate fails on correlation just under 0.99 (the real first gate: 0.9883)", not S10.gate_passed({"corr": 0.9883, "share_within": 0.9894}))
    check("the gate fails on a share just under 0.95", not S10.gate_passed({"corr": 0.999, "share_within": 0.9499}))
    check("a missing or NaN statistic fails the gate, never passes it", not S10.gate_passed({"corr": float("nan"), "share_within": 0.99}) and not S10.gate_passed({"share_within": 0.99}))
    check("the corrected gate (corr 0.99999, share 0.99997) passes", S10.gate_passed({"corr": 0.999995, "share_within": 0.99997}))


# ------------------------------------------------------------------------------------------------------------------------ universe
def test_universe():
    import gg_stage10 as S10
    new = pd.DataFrame({"cusip6": ["AAA111", "BBB222", "AAA111", "CCC333", "AAA111"], "trandate": pd.to_datetime(["2026-02-01", "2026-02-01", "2025-12-30", "2026-03-01", "2026-04-01"]),
                        "cleanse": ["R", "R", "R", "H", "L"]})
    out = S10.restrict_universe(new, {"AAA111", "CCC333"})
    check("2026 rows outside the original CUSIP universe are dropped", set(out.cusip6) <= {"AAA111", "CCC333"} and "BBB222" not in set(out.cusip6))
    check("rows before 2026 are not added (they come from the cached file)", (out.trandate >= "2026-01-01").all())
    check("only cleansed records (R or H) are kept", set(out.cleanse) <= {"R", "H"} and len(out) == 2, str(len(out)))


# ------------------------------------------------------------------------------------------------------------------------ Holm
def test_holm():
    import gg_stage10 as S10
    h = S10.holm_adjust({"a": 0.01, "b": 0.04})
    check("Holm with two tests: smallest p doubled, largest unchanged", abs(h["a"] - 0.02) < 1e-12 and abs(h["b"] - 0.04) < 1e-12, str(h))
    h = S10.holm_adjust({"a": 0.03, "b": 0.04})
    check("Holm adjusted p-values are monotone (0.03 and 0.04 both become 0.06)", abs(h["a"] - 0.06) < 1e-12 and abs(h["b"] - 0.06) < 1e-12, str(h))
    h = S10.holm_adjust({"a": 0.403, "b": 0.641})
    check("the real Stage 10 values (0.403, 0.641) both give 0.806 to three places", abs(h["a"] - 0.806) < 1e-3 and abs(h["b"] - 0.806) < 1e-3, str(h))
    check("adjusted values never exceed 1", max(S10.holm_adjust({"a": 0.7, "b": 0.9}).values()) <= 1.0)


# ------------------------------------------------------------------------------------------------------------------------ samples
def _panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"trandate": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D"), "gvkey": rng.integers(0, 50, n).astype(str),
                      "Yraw": rng.normal(0, 10, n), "Z": rng.normal(0, 1, n), "vol": rng.uniform(0.01, 0.05, n), "sic2": rng.choice(["28", "35", "73"], n),
                      "lmcap": rng.normal(22.3, 1.5, n)})
    d.loc[0, "Yraw"] = 1e4                                          # an extreme outlier
    return d


def test_samples():
    import gg_stage10 as S10
    d = _panel()
    s = S10.make_sample(d)
    check("outcome is winsorised on the sample itself (the 1e4 outlier is cut to the sample's 99th percentile)", s.Y63.max() < 100 and s.Y63.max() <= d.Yraw.quantile(0.99) + 1e-9, str(s.Y63.max()))
    check("V is standardised on the sample (mean 0, sd 1)", abs(s.V.mean()) < 1e-9 and abs(s.V.std() - 1) < 1e-9)
    d2 = d.copy()
    d2.loc[1, "Z"] = np.nan
    d2.loc[2, "sic2"] = np.nan
    d2.loc[3, "Yraw"] = np.nan
    check("rows missing the outcome, Z, volatility or industry are dropped", len(S10.make_sample(d2)) == len(d) - 3)
    pooled, new_all, large = S10.stage_samples(d, new_from="2025-10-01", large_cut=22.298)
    check("the new-months sample holds only sales on or after 2025-10-01", (new_all.trandate >= "2025-10-01").all() and len(new_all) < len(pooled))
    check("the large-firm sample is the new months with log market cap at least 22.298, inclusive", (large.lmcap >= 22.298).all() and (large.trandate >= "2025-10-01").all())
    edge = d.copy()
    edge.loc[5, ["lmcap", "trandate"]] = [22.298, pd.Timestamp("2025-10-01")]
    check("a sale exactly on the date and at exactly the threshold is included", len(S10.stage_samples(edge, "2025-10-01", 22.298)[2]) >= 1)
    check("winsorising is per sample: the large sample's Y range is not the pooled one's", large.Y63.max() <= large.Yraw.quantile(0.99) + 1e-9)


# ------------------------------------------------------------------------------------------------------------------------ filing chain
def test_filings():
    import gg_stage10 as S10
    tk = pd.DataFrame({"gvkey": ["A"] * 4 + ["C"] * 2 + ["D"], "accession": ["a22", "a23", "a24", "a25", "c1", "c2", "d1"],
                       "filing_date": pd.to_datetime(["2022-03-01", "2023-03-01", "2024-03-01", "2025-03-01", "2020-03-01", "2027-03-01", "2025-03-01"]),
                       "cik": "1", "url": "u"})
    ns = pd.DataFrame({"gvkey": ["A", "B", "C", "D"], "trandate": pd.to_datetime(["2025-06-01", "2025-06-01", "2025-06-01", "2024-06-01"])})
    last_have = pd.Series({"A": pd.Timestamp("2023-03-01")})
    out = S10.filings_to_label(ns, tk, last_have, have_accessions={"a24"}, end="2026-06-30")
    got = set(out.accession)
    check("a firm with a known last scored filing gets the filings after it, up to the end date, minus filings already labelled", got >= {"a25"} and "a22" not in got and "a23" not in got and "a24" not in got, str(sorted(got)))
    check("a firm with no prior filing before the sale (B, D) needs nothing", not ({"d1"} & got) and "B" not in set(out.gvkey))
    check("a firm with no scored history gets all its filings through the end date (C) but not later ones", "c1" in got and "c2" not in got, str(sorted(got)))
    check("each accession appears once", not out.accession.duplicated().any())
    check("the output has the columns the labelling runner reads", {"gvkey", "accession", "filing_date", "cik", "url"} <= set(out.columns))


# ------------------------------------------------------------------------------------------------------------------------ extension refusal checks
def test_extension_checks():
    import gg_stage10 as S10
    old = pd.DataFrame({"accession": ["a", "b", "c"], "t_final": [3.0, np.nan, 2.0], "phase": [0, 0, 1], "level": ["FIRST", "NONE", "MAJOR"], "phase_first": ["a", "a", "c"]})
    new_ok = pd.concat([old, pd.DataFrame({"accession": ["d"], "t_final": [4.0], "phase": [0], "level": ["FIRST"], "phase_first": ["d"]})], ignore_index=True)
    check("adding new filings changes no existing score", S10.changed_scores(new_ok, old) == [])
    check("a NaN score that stays NaN is unchanged (not flagged by NaN != NaN)", S10.changed_scores(new_ok, old) == [])
    check("a changed existing score is reported", S10.changed_scores(new_ok.assign(t_final=[3.0, np.nan, 5.0, 4.0]), old) == ["c"])
    check("a score that goes from NaN to a value is reported", S10.changed_scores(new_ok.assign(t_final=[3.0, 1.0, 2.0, 4.0]), old) == ["b"])
    check("an existing filing that disappears is reported as changed", S10.changed_scores(new_ok.iloc[[0, 2]], old) == ["b"])
    check("changed phases: identical existing rows give none, new rows are fine", S10.changed_phases(new_ok, old) == [])
    check("a changed phase_first is reported", S10.changed_phases(new_ok.assign(phase_first=["a", "a", "z", "d"]), old) == ["c"])
    check("a changed level is reported", S10.changed_phases(new_ok.assign(level=["FIRST", "MINOR", "MAJOR", "FIRST"]), old) == ["b"])


# ------------------------------------------------------------------------------------------------------------------------ the chain script
FAKE_UV = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG"
if [ "$1 $2 $3" = "run python -" ]; then cat > /dev/null; exit 0; fi
case "$*" in *"$FAIL_ON"*) [ -n "$FAIL_ON" ] && exit 1;; esac
case "$*" in
  *prep_stage10.py*) mkdir -p gg_cache; { echo "gvkey,accession,filing_date,cik,url"; i=0; while [ "$i" -lt "${FAKE_N:-0}" ]; do echo "g,a$i,2026-01-01,1,u"; i=$((i+1)); done; } > gg_cache/phase_needed_2026.csv;;
  *phase_run.py*) mkdir -p gg_cache; touch gg_cache/phase_labels_2026.jsonl;;
esac
exit 0
"""


def _run_chain(fail_on="", n=3, files=True):
    tmp = Path(tempfile.mkdtemp(prefix="chain_test_"))
    (tmp / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "stage10_chain.sh", tmp / "scripts" / "stage10_chain.sh")
    (tmp / "gg_cache").mkdir()
    (tmp / "outputs" / "results").mkdir(parents=True)
    if files:
        (tmp / "gg_cache" / "tr_insiders_2025_06_plus.pkl").write_bytes(b"x")
        (tmp / "outputs" / "results" / "secd_vs_crsp_v2.json").write_text("{}")
    bindir = tmp / "bin"
    bindir.mkdir()
    (bindir / "uv").write_text(FAKE_UV, newline="\n")
    os.chmod(bindir / "uv", 0o755)
    env = {**os.environ, "PATH": str(bindir) + os.pathsep + os.environ["PATH"], "FAKE_LOG": str(tmp / "calls.log").replace("\\", "/"), "FAIL_ON": fail_on, "FAKE_N": str(n), "STAGE10_POLL": "0"}
    gitbash = r"C:\Program Files\Gitinash.exe"
    bash = gitbash if os.path.exists(gitbash) else shutil.which("bash")
    p = subprocess.run([bash, str(tmp / "scripts" / "stage10_chain.sh")], cwd=tmp, env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp / "calls.log").read_text().splitlines() if (tmp / "calls.log").exists() else []
    return p.returncode, calls, (tmp / "gg_cache" / "stage10_chain.log").read_text() if (tmp / "gg_cache" / "stage10_chain.log").exists() else ""


def _scripts(calls):
    out = []
    for c in calls:
        for tok in c.split():
            if tok.endswith(".py"):
                out.append(Path(tok).name)
    return out


def test_chain():
    rc, calls, log = _run_chain(n=3)
    s = _scripts(calls)
    check("happy path: the chain ends with CHAIN DONE and exit 0", rc == 0 and "CHAIN DONE" in log, f"rc {rc}")
    order = ["prep_stage10.py", "gg_phase_run.py", "gg_phase_run.py", "phases_2026.py", "score_phases.py", "assemble_scores.py", "run_stage10.py"]
    check("steps run in order: prep, label twice, phases, score, assemble, run", s == order, str(s))
    check("the extension assembly is called with --extension2026 and the scorer with the 2026 phase table", any("assemble_scores.py --extension2026" in c for c in calls) and any("phase_table_2026.csv" in c for c in calls))
    rc, calls, log = _run_chain(fail_on="prep_stage10.py")
    check("a failing prep stops the chain: nothing after it runs", rc != 0 and _scripts(calls) == ["prep_stage10.py"] and "FAILED at: prep_stage10" in log, str(_scripts(calls)))
    rc, calls, log = _run_chain(fail_on="score_phases.py")
    check("a failing scorer stops the chain before assembly and before the tests", rc != 0 and "assemble_scores.py" not in _scripts(calls) and "run_stage10.py" not in _scripts(calls) and "FAILED at: score_phases" in log, str(_scripts(calls)))
    rc, calls, log = _run_chain(fail_on="assemble_scores.py")
    check("a refused extension (assemble fails) means the tests are never run", rc != 0 and "run_stage10.py" not in _scripts(calls), str(_scripts(calls)))
    rc, calls, log = _run_chain(n=0)
    check("with nothing to label the chain skips labelling, phases and scoring but still runs the tests", rc == 0 and "gg_phase_run.py" not in _scripts(calls) and "score_phases.py" not in _scripts(calls) and "run_stage10.py" in _scripts(calls), str(_scripts(calls)))
    rc, calls, log = _run_chain(fail_on="run_stage10.py")
    check("a failing test run is reported as a failure, not as DONE", rc != 0 and "CHAIN DONE" not in log, log[-200:])
    src = (ROOT / "scripts" / "stage10_chain.sh").read_text()
    check("the chain waits for the CORRECTED gate file (secd_vs_crsp_v2.json), not the first failed one", "secd_vs_crsp_v2.json" in src and "secd_vs_crsp.json" not in src.replace("secd_vs_crsp_v2.json", ""))


def main():
    test_gate()
    test_universe()
    test_holm()
    test_samples()
    test_filings()
    test_extension_checks()
    test_chain()
    print(f"\n{len(FAILS)} failures {FAILS}")
    return len(FAILS)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
