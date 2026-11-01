"""
gg_stage3_calibrate.py -- null calibration of the Stage-3 regressions on the REAL treatment structure.

Outcomes (exit indicator, miss) are randomly permuted within year-quarter, which destroys any true relation to optimism, sale and exit
but keeps the panel's size, clustering, rare-event structure and treatment overlap. Reports how often P1 and P2 reject at one-sided 5%.
No real outcome is used or printed.

    python -u gg_stage3_calibrate.py --reps 100
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg_panel_data as PD
import gg_stage3 as S3
import guide_and_go as G

DATA, RESULTS = G.DATA, G.RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    ep = pd.read_pickle(DATA / "panel_episodes.pkl"); sales = pd.read_pickle(DATA / "panel_sales.pkl")
    g = pd.read_pickle(DATA / "panel_guid.pkl"); Q = pd.read_pickle(DATA / "panel_A_quarters.pkl")
    db = G.WRDS(refresh=False)
    crsp, mkt, _ = G.load_crsp(db, sorted(g.permno.unique()), S3.START, S3.END)
    tab = S3.market_table(crsp); age_tab = S3.load_age(db)
    si, si1m = PD.SaleIndex(sales), PD.SaleIndex(sales, 1e6)
    B = S3.release_frame(g, ep, si, si1m)
    B = B[B.has_actual].reset_index(drop=True)
    B = S3.add_exec_controls(S3.add_market_controls(B, tab, "t"), ep, age_tab, "t"); B["gvkey"] = B.gvkey.astype(str)
    Q = Q.copy(); Q["permno"] = Q.permno.astype("int64"); Q["tq"] = Q.qs.dt.to_period("Q").astype(str)
    Q["opt_pos"] = (Q.opt > 0).astype(float)
    Q = S3.add_exec_controls(S3.add_market_controls(Q, tab, "qs"), ep, age_tab, "qs"); Q["gvkey"] = Q.gvkey.astype(str)
    rows = []
    for r in range(a.reps):
        row = {"rep": r}
        for name, fn, panel, key in (("P1", S3.run_p1, Q, "OS"), ("P2", S3.run_p2, B, "OSX")):
            for tag, c2 in (("two", "tq"), ("firm", None)):
                rr = fn(panel, cluster2=c2, rng=np.random.default_rng(1000 + r), dry=True)      # SAME randomised outcome for both SE types
                row[f"{name}_{tag}_p"] = rr["p_one"]; row[f"{name}_{tag}_se"] = rr[key]["se"]; row[f"{name}_{tag}_coef"] = rr[key]["coef"]
        rows.append(row)
        if (r + 1) % 10 == 0:
            d = pd.DataFrame(rows)
            print(f"after {r + 1}: " + " | ".join(f"{k} rejects {np.mean(d[k] < .05):.0%}" for k in ("P1_two_p", "P1_firm_p", "P2_two_p", "P2_firm_p")), flush=True)
    d = pd.DataFrame(rows)
    print("NULL CALIBRATION, paired draws (one-sided 5%, nominal 5%):")
    for nm in ("P1", "P2"):
        for tag in ("two", "firm"):
            v = d[f"{nm}_{tag}_p"]
            print(f"  {nm} {tag:<5s} rejects {np.mean(v < .05):.1%} | share p<0.10 {np.mean(v < .10):.1%} | mean se {d[f'{nm}_{tag}_se'].mean():.4f}")
        print(f"  {nm} se ratio two-way/firm-only: mean {np.mean(d[f'{nm}_two_se'] / d[f'{nm}_firm_se']):.2f}, min {np.min(d[f'{nm}_two_se'] / d[f'{nm}_firm_se']):.2f}, max {np.max(d[f'{nm}_two_se'] / d[f'{nm}_firm_se']):.2f}")
        print(f"  {nm} sd of coef across draws {d[f'{nm}_two_coef'].std():.4f} (a correct se should be close to this)")
    out = d
    out.to_csv(RESULTS / "stage3_null_calibration.csv", index=False)


if __name__ == "__main__":
    main()
