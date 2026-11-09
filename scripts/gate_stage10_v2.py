"""Stage 10 validation gate, corrected (docs/STAGE10_SPEC.md, Amendment 2): choose one Compustat issue per CRSP security on 2024 returns, then validate on 2025 only
(out of sample) against the unchanged bar (correlation >= 0.99 and >= 95% of returns within 0.001). Writes outputs/results/secd_vs_crsp_v2.json and the permno-keyed
Compustat daily file. No outcome is used.

python scripts/gate_stage10_v2.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_compustat_px as P

px = pd.read_pickle(G.CACHE / "secd_px_2024_2026.pkl")                      # every issue of every firm (the original primary-issue map was empty)
crsp = pd.read_pickle(G.CACHE / "ciz_dsf_2023_2025.pkl")
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
lk = link[link["linkprim"].isin(["P", "C"]) & link["linktype"].isin(["LC", "LU"]) & link.lpermno.notna()].copy()
lk["gvkey"] = lk["gvkey"].astype(str).str.zfill(6)
lk["linkdt"] = pd.to_datetime(lk["linkdt"], errors="coerce")
pg = dict(lk.sort_values("linkdt").drop_duplicates("lpermno", keep="last").assign(permno=lambda x: x.lpermno.astype("int64")).set_index("permno")["gvkey"])
_cp, _pg = set(crsp.permno.astype(int)), set(px.gvkey)                    # build the sets once (rebuilding them per link row made this quadratic)
pg = {p: g for p, g in pg.items() if p in _cp or g in _pg}
chosen = P.choose_issues(px, crsp, pg, train_year=2024)
print(f"securities: {len(pg)} | with a chosen issue: {len(chosen)} ({chosen.rate.notna().sum()} by matching on 2024, {chosen.rate.isna().sum()} single-issue fallback)", flush=True)
pp = P.by_permno(px, chosen)
pp.to_pickle(G.CACHE / "secd_px_by_permno_2024_2026.pkl")
chosen.to_pickle(G.CACHE / "secd_issue_choice.pkl")

c25 = crsp[crsp.date.dt.year == 2025][["permno", "date", "ret", "prc", "shrout"]]
m = pp[pp.date.dt.year == 2025].merge(c25, on=["permno", "date"], suffixes=("_s", "_c"))
mr = m.dropna(subset=["ret_s", "ret_c"])
d = (mr.ret_s - mr.ret_c).abs()
rep = {"scope": "2025 only, out of sample (issue chosen on 2024)", "n": int(len(mr)), "corr": float(np.corrcoef(mr.ret_s, mr.ret_c)[0, 1]), "share_within": float((d <= 1e-3).mean()),
       "p99_abs_diff": float(d.quantile(0.99)), "max_abs_diff": float(d.max()), "securities": int(mr.permno.nunique()), "rows_diff_gt_0.05": int((d > 0.05).sum())}
pr = (m.prc_s.abs() / m.prc_c.abs() - 1).abs().replace([np.inf, -np.inf], np.nan).dropna()
rep["price_share_within_0.1pct"] = float((pr <= 0.001).mean())
lm = (np.log(m.prc_s.abs() * m.shrout_s * 1000) - np.log(m.prc_c.abs() * m.shrout_c * 1000)).replace([np.inf, -np.inf], np.nan).dropna()
rep["lmcap_abs_diff_median"], rep["lmcap_abs_diff_p95"] = float(lm.abs().median()), float(lm.abs().quantile(0.95))
import gg_stage10 as S10
rep["gate_passed"] = S10.gate_passed(rep)
print(json.dumps(rep, indent=1), flush=True)
(G.RESULTS / "secd_vs_crsp_v2.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print("GATE", "PASSED" if rep["gate_passed"] else "FAILED", flush=True)
