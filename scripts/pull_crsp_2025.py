"""Pull the CRSP v2 (CIZ) names, daily returns and Fama-French factors that extend past 2024-12-31, through the WRDS REST API, into NEW cache files
(nothing existing is overwritten): gg_cache/ciz_names_2024plus.pkl, ciz_dsf_2023_2025.pkl, ciz_ff_daily.pkl. Counts and an overlap check only; no outcomes.

python scripts/pull_crsp_2025.py"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
import guide_and_go as G
import gg_crsp_ciz as C
import gg_stage8 as S8
import gg_tests as GT

db = G.WRDS(refresh=True)
last = lambda p: sorted(glob.glob(str(G.CACHE / p)))[-1]

# 1. names valid in 2024 or later, v2 -> v1 layout
n2 = db.fetch("crsp", "stocknames_v2", ["permno", "cusip", "ticker", "issuernm", "siccd", "namedt", "nameenddt"], [("nameenddt", "gte", "2024-01-01")])
n2 = C.names_v2_to_v1(n2)
n2.to_pickle(G.CACHE / "ciz_names_2024plus.pkl")
print("names v2 rows:", len(n2), "max nameenddt", n2.nameenddt.max(), flush=True)

# 2. permnos of the 2025 insider-days, linked with old plus new names
ins = pd.read_pickle(last("tr_insiders_2004_2025_*.pkl"))
old = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(G.CACHE / "crsp_stocknames_v2_*.pkl")))], ignore_index=True)
old["namedt"], old["nameenddt"] = pd.to_datetime(old["namedt"]), pd.to_datetime(old["nameenddt"])
names = C.merge_names(old, n2)
names.to_pickle(G.CACHE / "ciz_names_merged.pkl")
link = pd.read_pickle(G.CACHE / "ccm_link.pkl")
for c in ("linkdt", "linkenddt"):
    link[c] = pd.to_datetime(link[c], errors="coerce")
d = S8.insider_sale_days(ins, names, link, min_value=100_000.0, start="2024-06-01", end="2025-12-31")
print("linked insider-days by half-year:", d.groupby(d.trandate.dt.year.astype(str) + "H" + ((d.trandate.dt.month > 6) + 1).astype(str)).size().to_dict())
permnos = sorted(set(d.permno.astype(int)))
print("insider-days 2024-06..2025-12 linked:", len(d), "| permnos:", len(permnos), flush=True)

# 3. daily returns, v2, for those permnos
prev = pd.read_pickle(G.CACHE / "ciz_dsf_2023_2025.pkl") if (G.CACHE / "ciz_dsf_2023_2025.pkl").exists() else pd.DataFrame(columns=["permno", "date", "ret", "prc", "shrout"])
todo = sorted(set(permnos) - set(prev.permno.astype(int)))
print("permnos already pulled:", prev.permno.nunique(), "| to pull now:", len(todo), flush=True)
raw = db.fetch_in("crsp", "dsf_v2", ["permno", "dlycaldt", "dlyret", "dlyprc", "shrout"], "permno", todo, G._range("dlycaldt", "2023-01-01", "2025-12-31")) if todo else None
new = pd.concat([prev, C.daily_v2_to_v1(raw)], ignore_index=True) if raw is not None else prev
new = new.drop_duplicates(["permno", "date"]).sort_values(["permno", "date"]).reset_index(drop=True)
new.to_pickle(G.CACHE / "ciz_dsf_2023_2025.pkl")
print("daily v2 rows:", len(new), "permnos:", new.permno.nunique(), "date range", new.date.min().date(), new.date.max().date(), flush=True)

# 4. does v2 agree with the v1 file we already used, on the overlap (2023-01 to 2024-12)?
v1 = pd.read_pickle(last("crsp_dsf_v2_*.pkl"))
v1["date"] = pd.to_datetime(v1["date"])
v1["permno"] = pd.to_numeric(v1["permno"]).astype("int64")
v1["ret"] = pd.to_numeric(v1["ret"], errors="coerce")
ov = C.overlap_report(v1[v1.permno.isin(permnos) & (v1.date >= "2023-01-01")], new[new.date <= "2024-12-31"])
print("OVERLAP v1 vs v2 on 2023-01..2024-12:", ov, flush=True)

# 5. Fama-French factors: do they reach 2025?
loc = db.find([("ff", "fivefactors_daily"), ("ff", "factors_daily")])
f = db.fetch(*loc, ["date", "mktrf", "smb", "hml", "rmw", "cma", "umd", "rf"])
f["date"] = pd.to_datetime(f["date"])
f.to_pickle(G.CACHE / "ciz_ff_daily.pkl")
print("FF factors from", loc, "rows", len(f), "max date", f.date.max().date(), flush=True)
