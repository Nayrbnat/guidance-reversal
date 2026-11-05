"""Stage 6 moderator follow-up (docs/STAGE6_SPEC.md section 9): litigation-prone industry L and pre-sale return volatility V, each interacted with
guided and left, entered one at a time and together beside the primary Z x G x X term. Reads the saved panel outputs/data/stage6_panel.pkl.

python scripts/run_stage6_moderators.py"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_stage3 as S3
import gg_stage6 as S6
import gg_tests as GT

t0 = time.time()
d = pd.read_pickle(G.DATA / "stage6_panel.pkl")
base = d[(d["val"] >= 100_000) & d["Z"].notna() & d["Y63"].notna()].copy().reset_index(drop=True)
print("primary sample:", len(base), flush=True)

# litigation-prone industry from the SIC on each firm's filings (the latest filing's code)
idx = pd.read_pickle(G.CACHE / "filing_index_stage5.pkl")
sic = idx.sort_values("filing_date").drop_duplicates("gvkey", keep="last").set_index("gvkey")["sic"]
base["sic"] = base["gvkey"].map(sic)
base["L"] = S6.fps_industry(base["sic"]).astype(float)
print("litigation-prone share of sale-days:", round(base["L"].mean(), 3), "| with SIC:", int(base["sic"].notna().sum()), flush=True)

# pre-sale volatility from CRSP daily returns
db = G.WRDS(refresh=False)
ep_g = pd.read_pickle(G.DATA / "panel_guid.pkl")
crsp, mkt, _ = G.load_crsp(db, sorted(ep_g.permno.unique()), S3.START, S3.END)
fac = GT.load_factors(db)
print(f"crsp loaded ({(time.time() - t0) / 60:.1f} min)", flush=True)
base = S6.pre_sale_vol(base, crsp, fac)
ok = base["vol"].notna()
base["V"] = (base["vol"] - base.loc[ok, "vol"].mean()) / base.loc[ok, "vol"].std()
print("volatility available:", int(ok.sum()), "of", len(base), flush=True)
sub = base[ok].copy()


def one_sided_neg(c):
    return c["p_two"] / 2 if c["coef"] < 0 else 1 - c["p_two"] / 2


res = {"spec": "docs/STAGE6_SPEC.md section 9", "n_primary": int(len(base)), "n_with_vol": int(ok.sum()), "share_L": float(base["L"].mean())}
res["primary_same_sample"] = S6.run_primary(sub, "Y63")
for name, mods in (("L", ["L"]), ("V", ["V"]), ("L_and_V", ["L", "V"])):
    r = S6.run_primary(sub, "Y63", moderators=mods)
    res[name] = r
    out = {"ZGX": (round(r["ZGX"]["coef"], 3), round(r["ZGX"]["se"], 3), round(r["p_one"], 3))}
    for m in mods:
        c = r[m + "GX"]
        out[m + "GX"] = (round(c["coef"], 3), round(c["se"], 3), "one-sided(neg) p=%.3f" % one_sided_neg(c) if m == "L" else "two-sided p=%.3f" % c["p_two"])
    print(name, out, flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


(G.RESULTS / "stage6_moderators.json").write_text(json.dumps(clean(res), indent=1), encoding="utf-8")
print("wrote", G.RESULTS / "stage6_moderators.json", f"({(time.time() - t0) / 60:.1f} min)", flush=True)
