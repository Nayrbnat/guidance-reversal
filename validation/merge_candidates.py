"""
Merge annotator labels for candidate events into validation/candidate_labels.csv.

Rules (fixed in advance, see PREREGISTRATION.md):
  * an event is ABRUPT only if the pass-1 annotator AND the pass-2 annotator both say abrupt = true
    (events pass 2 did not re-read were not abrupt in pass 1);
  * 'unclear' (abrupt = null) counts as NOT abrupt;
  * announcement date = date of the item the (pass-1) annotator identified as announcing this executive's departure;
    the pass-2 item is used only if pass 1 found none.
Disagreements between the two passes are written to validation/candidate_disagreements.csv for adjudication.
"""
import sys, json, glob
import pandas as pd

TYPES = {"retirement_planned", "resigned_abrupt", "terminated_forced", "left_for_other_job", "health_personal",
         "death", "government", "other_unclear", "not_a_departure"}


def read(prefix):
    rows = []
    for f in sorted(glob.glob(f"validation/cand_labels_raw/{prefix}_*.jsonl")):
        for ln in open(f, encoding="utf-8"):
            if ln.strip():
                rows.append(json.loads(ln))
    return pd.DataFrame(rows)


def item_dates(prefix):
    d = {}
    for f in glob.glob(f"validation/cand_tasks/{prefix}_*.jsonl"):
        for ln in open(f, encoding="utf-8"):
            t = json.loads(ln)
            d[t["id"]] = {k["i"]: k["date"] for k in t["items"]}
    return d


p1, p2 = read("pass1"), read("pass2")
print("pass 1 labels:", len(p1), "| pass 2 labels:", len(p2), "| invalid dep_type:", int((~p1.dep_type.isin(TYPES)).sum()))
d1, d2 = item_dates("pass1"), item_dates("pass2")
p1 = p1.drop_duplicates("id").set_index("id")
p2 = p2.drop_duplicates("id").set_index("id") if len(p2) else pd.DataFrame(columns=p1.columns)
out = []
for eid, r in p1.iterrows():
    a1 = str(r.abrupt) == "True"
    in2 = eid in p2.index
    a2 = (str(p2.loc[eid, "abrupt"]) == "True") if in2 else None
    abrupt = bool(a1 and (a2 if in2 else False))
    item = None if pd.isna(r.ann_item) else int(r.ann_item)
    date = d1.get(eid, {}).get(item) if item is not None else None
    found = item is not None
    if date is None and in2 and not pd.isna(p2.loc[eid, "ann_item"]):
        date = d2.get(eid, {}).get(int(p2.loc[eid, "ann_item"]))
        found = date is not None
    out.append({"event_id": eid, "ann_date": date, "ann_found_ann": int(bool(found and date)), "dep_type_ann": r.dep_type,
                "abrupt_p1": int(a1), "abrupt_p2": (None if a2 is None else int(a2)), "abrupt_ann_label": int(abrupt),
                "abrupt_unclear_p1": int(pd.isna(r.abrupt)), "gov_ann": int(str(r.government_destination) == "True"),
                "confidence": r.confidence})
res = pd.DataFrame(out)
res.to_csv("validation/candidate_labels.csv", index=False)
dis = res[(res.abrupt_p1 == 1) & (res.abrupt_p2 == 0)]
dis.to_csv("validation/candidate_disagreements.csv", index=False)
print(f"candidates labelled: {len(res)} | pass-1 abrupt: {int(res.abrupt_p1.sum())} | confirmed abrupt (both passes): {int(res.abrupt_ann_label.sum())} "
      f"| pass-1 vs pass-2 disagreements on abrupt: {len(dis)} | unclear: {int(res.abrupt_unclear_p1.sum())}")
print(res.dep_type_ann.value_counts().to_dict())
