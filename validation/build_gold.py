"""Merge annotator files into validation/gold_labels.csv and report inter-annotator agreement."""
import json, glob, sys
import pandas as pd, numpy as np
TYPES = {"retirement_planned","resigned_abrupt","terminated_forced","left_for_other_job","health_personal","death","government","other_unclear","not_a_departure"}
key = pd.read_csv("validation/gold_key.csv")
rows = []
for f in sorted(glob.glob("validation/labels_raw/annotator_*.jsonl")):
    k = int(f.split("_")[-1].split(".")[0])
    for ln in open(f, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            d = json.loads(ln); d["annotator"] = k; rows.append(d)
raw = pd.DataFrame(rows)
bad = raw[~raw.dep_type.isin(TYPES)]
print("annotator files:", sorted(raw.annotator.unique()), "| rows:", len(raw), "| invalid dep_type rows:", len(bad))
prim = raw.merge(key[["id", "annotator"]].rename(columns={"annotator": "prim_annot"}), on="id")
gold = prim[prim.annotator == prim.prim_annot].drop(columns=["prim_annot"])
second = prim[(prim.annotator == 8)].set_index("id")
gold = gold.merge(key.drop(columns=["annotator"]), on="id")
print("gold rows:", len(gold), "of", len(key), "| missing ids:", sorted(set(key.id) - set(gold.id))[:10])
# agreement on the 40 overlapping tasks: annotator 8 vs the primary annotator
ov = gold[gold.id.isin(second.index)].set_index("id")
def agree(col, f=lambda x: x):
    a = ov[col].map(f); b = second.loc[ov.index, col].map(f)
    return (a == b).mean()
def kappa(a, b):
    a, b = pd.Series(a).astype(str).values, pd.Series(b).astype(str).values
    cats = sorted(set(a) | set(b)); po = np.mean(a == b)
    pe = sum((np.mean(a == c) * np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
n = len(ov)
if n:
    print(f"inter-annotator (n={n}): ann_item agree {agree('ann_item', lambda x: -1 if pd.isna(x) else int(x)):.0%} | dep_type agree {agree('dep_type'):.0%} "
          f"(kappa {kappa(ov.dep_type, second.loc[ov.index, 'dep_type']):.2f}) | abrupt agree {agree('abrupt', lambda x: 'na' if pd.isna(x) else str(x)):.0%} | gov agree {agree('government_destination'):.0%}")
    dis = ov[(ov.dep_type != second.loc[ov.index, 'dep_type'])]
    print("dep_type disagreements:", len(dis), "|", list(zip(dis.dep_type, second.loc[dis.index, 'dep_type']))[:12])
gold.to_csv("validation/gold_labels.csv", index=False)
print(gold.dep_type.value_counts().to_dict())
print("abrupt:", gold.abrupt.astype(str).value_counts().to_dict(), "| gov destination true:", int((gold.government_destination == True).sum()))
