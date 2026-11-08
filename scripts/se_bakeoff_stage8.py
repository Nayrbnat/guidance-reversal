"""STANDARD-ERROR-ONLY comparison of design variants for the Stage 8 mechanism test. Prints ONLY the standard error, n, the sd of the outcome, and the minimum
detectable effect. The coefficients are never printed or saved, so the choice of design cannot depend on the answer. Exploratory (the sample was seen in Stage 8).

python scripts/se_bakeoff_stage8.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_stage8 as S8

CTRL = ["lmcap", "ret12", "lval", "V", "role_CEO", "role_CFO", "role_DIRECTOR"]
p = pd.read_pickle(G.DATA / "stage8_panel.pkl")
b = p[p["Y63"].notna() & p["Z"].notna() & p["V"].notna() & p["sic2"].notna()].copy().reset_index(drop=True)
b["sicq"] = b["sic2"].astype(str) + "_" + b["yq"].astype(str)
b["lmcap2"] = b["lmcap"] ** 2
b["ret12sq"] = b["ret12"] ** 2
b["V2"] = b["V"] ** 2
vol_pp = 100.0 * b["vol"] * np.sqrt(63) if "vol" in b else None                   # pre-sale daily vol scaled to a 63-day horizon, in percentage points
if vol_pp is not None:
    b["Y63_vs"] = b["Y63"] / vol_pp                                                  # avoided loss per unit of the stock's own risk
rows = []


def se_only(name, d, y, **kw):
    r = S8.run_mechanism(d, y, controls=kw.pop("controls", CTRL), **kw)
    sd = float(d[y].std())
    rows.append({"variant": name, "n": int(r["n"]), "se": float(r["Z"]["se"]), "sd_y": sd, "mde": float(2.49 * r["Z"]["se"]), "mde_in_sd_of_y": float(2.49 * r["Z"]["se"] / sd)})


se_only("A. Stage 8 baseline (SIC2 + quarter, firm and quarter clusters)", b, "Y63")
se_only("B. baseline, firm clusters only", b, "Y63", two_way=False)
se_only("C. industry-by-quarter effects, firm clusters", b, "Y63", firm="sicq", period=None, two_way=False)
se_only("D. flexible controls (squares of size, return, volatility)", b, "Y63", controls=CTRL + ["lmcap2", "ret12sq", "V2"], two_way=False)
se_only("E. D plus industry-by-quarter effects", b, "Y63", firm="sicq", period=None, controls=CTRL + ["lmcap2", "ret12sq", "V2"], two_way=False)
if vol_pp is not None:
    se_only("F. outcome scaled by the stock's own 63-day volatility (new units)", b, "Y63_vs", two_way=False)
    se_only("G. F plus industry-by-quarter effects", b, "Y63_vs", firm="sicq", period=None, two_way=False)
for h in (21, 126):
    if f"Y{h}" in b:
        se_only(f"H{h}. horizon {h} trading days", b[b[f"Y{h}"].notna()], f"Y{h}", two_way=False)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False))
(G.RESULTS / "stage8_se_bakeoff.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
