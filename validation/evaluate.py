"""Evaluate gg_labels.classify_event against the hand-labelled gold set. usage: python validation/evaluate.py dev|test|all [-v]"""
import sys, json
import pandas as pd, numpy as np
sys.path.insert(0, "src")
import gg_labels as L
which = sys.argv[1] if len(sys.argv) > 1 else "dev"
verbose = "-v" in sys.argv
g = pd.read_csv("validation/gold_labels.csv")
if which != "all":
    g = g[g.split == which]
tasks = {}
for k in range(1, 9):
    for ln in open(f"validation/tasks/annotator_{k}.jsonl", encoding="utf-8"):
        d = json.loads(ln); tasks.setdefault(d["id"], d)
rows = []
for _, r in g.iterrows():
    t = tasks[r.id]
    c = L.classify_event(t["items"], t["exec_name"])
    gold_item = None if pd.isna(r.ann_item) else int(r.ann_item)
    rows.append(dict(id=r.id, stratum=r.stratum, gold_dep=gold_item is not None, pred_dep=c["departing"],
                     item_ok=(gold_item == c["item"]), gold_type=r.dep_type, pred_type=c["dep_type"],
                     gold_abrupt=int(str(r.abrupt) == "True"), pred_abrupt=int(c["abrupt"] is True),
                     gold_gov=int(str(r.government_destination) == "True"), pred_gov=int(c["gov"]), old_abrupt=r.old_abrupt))
e = pd.DataFrame(rows)
def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return f"P={pr:.0%} R={rc:.0%} F1={2*pr*rc/max(pr+rc,1e-9):.0%} (tp {tp} fp {fp} fn {fn})"
print(f"=== {which}: n={len(e)} ===")
print("departure detected      :", prf(e.gold_dep.astype(int), e.pred_dep.astype(int)), "| same item as gold:", f"{e.item_ok.mean():.0%}")
print("ABRUPT (new)            :", prf(e.gold_abrupt, e.pred_abrupt))
print("ABRUPT (old pipeline)   :", prf(e.gold_abrupt, e.old_abrupt.astype(int)))
print("GOVERNMENT              :", prf(e.gold_gov, e.pred_gov))
both = e[e.gold_dep & e.pred_dep]
print("dep_type exact match (both found a departure):", f"{(both.gold_type == both.pred_type).mean():.0%} of {len(both)}")
print(pd.crosstab(e.gold_type, e.pred_type).to_string())
if verbose:
    bad = e[(e.gold_abrupt != e.pred_abrupt)]
    print("\nABRUPT errors:", len(bad))
    for _, r in bad.head(25).iterrows():
        t = tasks[r.id]; print(f"\n[{r.id}] {t['exec_name']} | gold={r.gold_type} abrupt={r.gold_abrupt} | pred={r.pred_type} abrupt={r.pred_abrupt}")
        for k in t["items"]:
            if r.gold_dep or True:
                print(f"   item {k['i']} {k['date']}: {k['text'][:380]}")
