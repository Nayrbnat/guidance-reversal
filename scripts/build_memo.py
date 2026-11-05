"""Build MEMO.md from docs/MEMO_TEMPLATE.md: every {{TABLE:name}} placeholder is replaced by the table that scripts/memo_tables.py
generates from outputs/results/*.json, so no number in the tables is typed by hand.

python scripts/build_memo.py"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import memo_tables as MT

text = (ROOT / "docs" / "MEMO_TEMPLATE.md").read_text(encoding="utf-8")
names = re.findall(r"\{\{TABLE:(\w+)\}\}", text)
unknown = [n for n in names if n not in MT.TABLES]
if unknown:
    sys.exit(f"unknown table placeholder(s): {unknown}")
for n in set(names):
    text = text.replace("{{TABLE:" + n + "}}", MT.TABLES[n]())
if "{{" in text:
    sys.exit("unfilled placeholder left in the memo")
(ROOT / "MEMO.md").write_text(text, encoding="utf-8")
print(f"MEMO.md written: {len(text.splitlines())} lines, {len(names)} tables ({', '.join(names)})")
