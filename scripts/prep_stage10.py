"""Stage 10 preparation (docs/STAGE10_SPEC.md, Amendments 1 and 2): validation gate on the Compustat prices, link the 2026 insider sales, pull CRSP history for new
firms, splice CRSP and Compustat prices, and list the 10-K filings that still need labelling and scoring. Counts only; no outcome is computed.

python scripts/prep_stage10.py"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import guide_and_go as G
import gg_compustat_px as P
import gg_crsp_ciz as C
import gg_stage5 as S5
import gg_stage8 as S8
import gg_stage10 as S10

START, END = "2025-01-01", "2026-06-30"
last = lambda p: sorted(glob.glob(str(G.CACHE / p)))[-1]

# 1. the validation gate, fixed in the spec before any 2026 data was touched
v = json.loads((G.RESULTS / "secd_vs_crsp_v2.json").read_text())      # Amendment 2: one issue per CRSP security, validated on 2025 out of sample
print("VALIDATION Compustat-built vs CRSP v2 returns:", {k: v.get(k) for k in ("n", "corr", "share_within", "p99_abs_diff", "max_abs_diff")}, flush=True)
if not S10.gate_passed(v):
    (G.RESULTS / "stage10_cancelled.json").write_text(json.dumps({"reason": "Compustat validation failed the spec's bar (corr >= 0.99 and share within 0.001 >= 0.95)", **v}, indent=1))
    sys.exit("CANCELLED: the Compustat price route failed validation; Stage 10 is not run (spec Amendments 1 and 2).")
print("  gate passed", flush=True)

# 2. insiders: the cached file (to 2025-12-31) plus the streamed rows for 2026
old = pd.read_pickle(last("tr_insiders_2004_2025_*.pkl"))
new = pd.read_pickle(G.CACHE / "tr_insiders_2025_06_plus.pkl")
new["trandate"] = pd.to_datetime(new["trandate"], errors="coerce")
old["trandate"] = pd.to_datetime(old["trandate"], errors="coerce")
for df in (old, new):
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["tprice"] = pd.to_numeric(df["tprice"], errors="coerce")
a = old[(old.trancode == "S") & (old.trandate.between("2025-06-01", "2025-12-31"))]
b = new[(new.trancode == "S") & (new.trandate.between("2025-06-01", "2025-12-31"))]
print(f"CONSISTENCY 2025-06..12 sales rows: cached {len(a):,}, streamed {len(b):,} (cleanse flag in streamed: {new.cleanse.value_counts(dropna=False).head(4).to_dict()})", flush=True)
add = S10.restrict_universe(new, set(old.cusip6))                        # the cached file covers the original firm universe; keep 2026 to the same firms
ins = pd.concat([old, add[[c for c in old.columns if c in add.columns]]], ignore_index=True)
ins.to_pickle(G.CACHE / "tr_insiders_combined_2026.pkl")
print(f"2026 rows added: {len(add):,} (sales {int((add.trancode == 'S').sum()):,}); latest trade date {add.trandate.max().date()}", flush=True)

# 3. names and links; the CRSP name rows active at 2025-12-31 are treated as active through 2026 (disclosed in the spec)
names = pd.read_pickle(G.CACHE / "ciz_names_merged.pkl")
names.loc[names["nameenddt"] == pd.Timestamp("2025-12-31"), "nameenddt"] = pd.Timestamp("2026-12-31")
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
for c in ("linkdt", "linkenddt"):
    link[c] = pd.to_datetime(link[c], errors="coerce")
d = S8.insider_sale_days(ins, names, link, min_value=100_000.0, start=START, end=END)
d["gvkey"] = d["gvkey"].astype(str).str.zfill(6)
s = ins[(ins.trancode == "S")].copy()
s["role"] = s.rolecode1.map(S8.role_group)
s = s[s.role.notna() & s.trandate.between(START, END)]
s["val"] = s.shares.abs() * s.tprice
rd = s.groupby(["personid", "cusip6", "cusip2", "trandate"], as_index=False).agg(val=("val", "sum"))
rd = rd[rd.val >= 100_000]
q_raw = rd.groupby(rd.trandate.dt.to_period("Q")).size()
q_lnk = d.groupby(d.trandate.dt.to_period("Q")).size()
print("LINK RATE by quarter (linked / raw insider-days >= $100k):", {str(k): round(float(q_lnk.get(k, 0) / v), 3) for k, v in q_raw.items()}, flush=True)
print("linked insider-days:", len(d), "| firms", d.gvkey.nunique(), "| by role", d.role.value_counts().to_dict(), flush=True)
d.to_pickle(G.DATA / "stage10_insider_days_prelink.pkl")

# 4. CRSP history for permnos not yet pulled (needed for the 252-day betas and size controls)
permnos = sorted(set(d.permno.astype(int)))
crsp_path = G.CACHE / "ciz_dsf_2023_2025.pkl"
crsp = pd.read_pickle(crsp_path)
todo = sorted(set(permnos) - set(crsp.permno.astype(int)))
print(f"permnos in sample {len(permnos)}, already pulled {len(permnos) - len(todo)}, to pull now {len(todo)}", flush=True)
if todo:
    db = G.WRDS(refresh=True)
    pd.to_pickle(crsp, G.CACHE / "ciz_dsf_2023_2025_pre2026.pkl")
    raw_new = db.fetch_in("crsp", "dsf_v2", ["permno", "dlycaldt", "dlyret", "dlyprc", "shrout"], "permno", todo, G._range("dlycaldt", "2023-01-01", "2025-12-31"))
    crsp = pd.concat([crsp, C.daily_v2_to_v1(raw_new)], ignore_index=True).drop_duplicates(["permno", "date"]).sort_values(["permno", "date"]).reset_index(drop=True)
    crsp.to_pickle(crsp_path)
    print("CRSP daily rows now:", len(crsp), "permnos", crsp.permno.nunique(), flush=True)

# 5. splice CRSP (through 2025-12-31) and Compustat (after), keyed by permno
px = pd.read_pickle(G.CACHE / "secd_px_by_permno_2024_2026.pkl")          # permno-keyed, one chosen issue per security (Amendment 2)
px["gvkey"] = px["permno"].astype(str).str.zfill(6)
lk = link[link["linkprim"].isin(["P", "C"]) & link["linktype"].isin(["LC", "LU"]) & link.lpermno.notna()].copy()
lk["gvkey"] = lk["gvkey"].astype(str).str.zfill(6)
g2p = {str(p).zfill(6): int(p) for p in px["permno"].unique()}              # identity map: the file is already keyed by permno
spl = P.splice_returns(crsp, px, g2p, cutoff="2025-12-31")
spl.to_pickle(G.CACHE / "spliced_prices_2023_2026.pkl")
need_p = set(permnos)
cov = spl[spl.permno.isin(need_p) & (spl.date > "2025-12-31")].permno.nunique()
print(f"SPLICED prices: {len(spl):,} rows; sample permnos with 2026 Compustat prices: {cov} of {len(permnos)}; latest date {spl.date.max().date()}", flush=True)

# 6. which 10-Ks still need labelling and scoring
scores = pd.read_pickle(G.DATA / "stage5_scores_2025.pkl")
meta = S5.score_meta(pd.read_pickle(G.DATA / "stage5_scores.pkl"))
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
x = S5.attach_T(d, scores, idx, meta, date_col="trandate", strict=True)
print("T coverage of the new window with current scores:", int(x["T"].notna().sum()), "of", len(x), "| reasons", x["T_missing_reason"].value_counts().to_dict(), flush=True)
tk = idx[idx.form.isin(["10-K", "10-K405"])].copy()
ns = x[x.T_missing_reason == "not_scored"][["gvkey", "trandate"]].copy()
have = pd.concat([pd.read_csv(G.CACHE / "phase_needed_all.csv", dtype=str)])
have["gvkey"] = have.gvkey.astype(str).str.zfill(6)
last_have = have.groupby("gvkey").filing_date.max()
chain = S10.filings_to_label(ns, tk, last_have, set(have.accession), END)
firms = chain.gvkey.unique()
chain.to_csv(G.CACHE / "phase_needed_2026.csv", index=False)
print(f"FILINGS to label for 2026: {len(chain)} from {chain.gvkey.nunique()} firms (unscored insider-days {len(ns)}, firms {len(firms)})", flush=True)
print("PREP DONE", flush=True)
