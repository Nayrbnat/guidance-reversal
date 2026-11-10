"""Stage 13 COUNTS ONLY (docs/STAGE13_SPEC.md): how many events the wider sale condition adds to the Guidance Reversal flag. No return or alpha is computed or printed;
only whether a return window exists (not its value).

python scripts/count_stage13.py"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import guide_and_go as G
import gg_stage13 as S13

last = lambda p: sorted(glob.glob(str(G.CACHE / p)))[-1]
ev = pd.read_csv(G.DATA / "events.csv", dtype={"gvkey": str}, parse_dates=["dep_ann"])
ev["guided_recent"] = (ev.days_since_guid <= 180).fillna(False).astype(int)
ev["abrupt"] = (ev.abrupt_rule == 1).astype(int)
ev["gr_candidate_old"] = ((ev.sell_val_post_lastguid > 0) & (ev.guided_recent == 1)).astype(int)
lab = pd.read_csv(G.ROOT / "validation" / "candidate_labels.csv").set_index("event_id")
ev["abrupt_ann"] = ev["abrupt_rule"]
has = ev.event_id.isin(lab.index) & ev.event_id.map(lab.abrupt_ann_label).notna()
ev.loc[has, "abrupt_ann"] = ev.loc[has, "event_id"].map(lab.abrupt_ann_label).astype(float)
ev["flag_old"] = ((ev.gr_candidate_old == 1) & (ev.abrupt_ann == 1)).astype(int)

# insiders in the layout insider_features expects (permno attached by the 8-character CUSIP, as the original build did)
raw = pd.read_pickle(last("tr_insiders_2004_2025_*.pkl"))
names = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(str(G.CACHE / "crsp_stocknames_v2_*.pkl")))], ignore_index=True)
names["ncusip"] = names["ncusip"].astype(str).str[:8]
raw["ncusip"] = raw.cusip6.astype(str) + raw.cusip2.astype(str)
ins = raw.merge(names[["ncusip", "permno"]].drop_duplicates(), on="ncusip")
ins["trandate"] = pd.to_datetime(ins.trandate)
ins["permno"] = ins.permno.astype(int)
for c in ("shares", "tprice"):
    ins[c] = pd.to_numeric(ins[c], errors="coerce")
crsp = pd.read_pickle(last("crsp_dsf_v2_*.pkl"))
crsp["date"] = pd.to_datetime(crsp.date)
crsp["permno"] = pd.to_numeric(crsp.permno).astype(int)
for c in ("ret", "prc", "shrout"):
    crsp[c] = pd.to_numeric(crsp[c], errors="coerce")
crsp["prc"] = crsp.prc.abs()
ev["permno"] = ev.permno.astype("Int64")
e = ev[ev.permno.notna() & ev.days_since_guid.notna()].copy()
e["permno"] = e.permno.astype(int)
e["sell_val_win9"] = S13.window_features(e, ins, crsp, 9)
ev["sell_val_win9"] = e["sell_val_win9"].reindex(ev.index).fillna(0.0)
ev = S13.add_flags(ev)
ev["has_window"] = ev.bhar_252.notna()
ev.to_pickle(G.DATA / "stage13_events_counts.pkl")

n = lambda m: int(m.sum())
new_only = (ev.gr_candidate9 == 1) & (ev.gr_candidate_old == 0)
rep = {
    "events": int(len(ev)), "guided_recent": n(ev.guided_recent == 1),
    "old_candidates": n(ev.gr_candidate_old == 1), "new_candidates": n(ev.gr_candidate9 == 1), "added_candidates": n(new_only),
    "old_flag_annotated": n(ev.flag_old == 1), "old_flag_with_window": n((ev.flag_old == 1) & ev.has_window),
    "new_candidates_abrupt_by_rule": n((ev.gr_candidate9 == 1) & (ev.abrupt_rule == 1)), "added_abrupt_by_rule": n(new_only & (ev.abrupt_rule == 1)),
    "added_abrupt_by_rule_with_window": n(new_only & (ev.abrupt_rule == 1) & ev.has_window),
    "added_needing_annotation": n(new_only & (ev.abrupt_rule == 1) & ~ev.event_id.isin(lab.index)),
    "added_by_role": ev[new_only & (ev.abrupt_rule == 1)].role.value_counts().to_dict(),
    "added_by_year": {int(k): int(v) for k, v in ev[new_only & (ev.abrupt_rule == 1)].dep_ann.dt.year.value_counts().sort_index().items()},
    "windows_available_events": n(ev.has_window), "last_event_with_window": str(ev.loc[ev.has_window, "dep_ann"].max().date()),
}
fis = ev[ev.event_id == "012635_35314_20250506"].iloc[0]
rep["fiserv"] = {"old_candidate": int(fis.gr_candidate_old), "new_candidate": int(fis.gr_candidate9), "new_rule_flag": int(fis.gr_rule9), "sell_val_win9": float(fis.sell_val_win9), "has_window": bool(fis.has_window)}
print(json.dumps(rep, indent=1))
(G.RESULTS / "stage13_counts.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
