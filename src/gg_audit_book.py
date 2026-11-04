"""Blind audit workbook for the transparency (T) hand audit.

One Excel file, one row per filing. The only thing the rater clicks is the OPEN link in each row: it opens a local reader page
holding the filing's text (10-K business description, segment and KPI text, other furnished materials) with the scoring cheat sheet
beside it and previous/next buttons. The rater types the 0-5 score into the highlighted column. The filing id sits in a hidden column
and never appears on a reader page; company names are already masked in the text."""
from __future__ import annotations

import html as H
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

import gg_transparency as T

FIRST_ROW = 5          # rows 1-3 hold the title, instructions and progress counter; row 4 the header
COLS = ["Row", "OPEN (click)", "YOUR SCORE 0-5", "Note (optional)", "Done", "id (hidden)"]

TERMS = ["available seat miles", "revenue per available seat mile", "load factor", "revenue per user", "average revenue per",
         "same-store sales", "same store sales", "revenue passenger miles", "price per unit", "average selling price",
         "subscribers", "subscriptions", "procedures", "passengers", "shipments", "connections", "customers", "volumes",
         "units sold", "units", "tons", "carloads", "yield", "bookings", "backlog", "installed base", "price", "volume"]
_TERM_RE = re.compile("|".join(re.escape(t) for t in sorted(TERMS, key=len, reverse=True)), re.I)
_HEADING = re.compile(r"^=== .* ===$")

CHEAT = [
    ("5", "Volume and price (stated or derivable) for every line that makes the profit."),
    ("4", "Volume and price for about three quarters of the profit. One meaningful line is dollars only."),
    ("3", "About half the profit covered, or volume OR price (not both) for the main engine."),
    ("2", "Dollars only: revenue and profit by line, no volume or price for the main engine."),
    ("1", "Almost opaque: total or regional revenue; drivers only in words."),
    ("0", "Nothing: one revenue number, no breakdown."),
]
CAPS = ("Razor-and-blade cap: price and quantity only for the low-margin line (the razor), dollars only for the high-margin recurring "
        "line (the blades): at most 2. Cannot tell where the profit comes from: at most 2. Number of segments does not matter. "
        "Unsure between two scores: pick the lower one and add a note. Judge only this text.")

CSS = """
body{font:16px/1.55 Georgia,serif;margin:0;color:#1d1d1f;background:#fafaf7}
.bar{position:sticky;top:0;background:#1f3a5f;color:#fff;padding:10px 20px;display:flex;gap:16px;align-items:center;z-index:2}
.bar a{color:#fff;background:#3b5f8f;padding:6px 14px;border-radius:5px;text-decoration:none;font:14px sans-serif}
.bar .gap{flex:1}
.wrap{display:flex;gap:24px;max-width:1300px;margin:0 auto;padding:18px}
main{flex:1;min-width:0}
aside{width:310px;flex:none;position:sticky;top:64px;align-self:flex-start;background:#fff;border:1px solid #d8d8d0;border-radius:8px;
 padding:14px;font:14px/1.45 sans-serif}
aside h3{margin:0 0 8px}
aside td{vertical-align:top;padding:3px 6px 3px 0}
aside td:first-child{font-weight:bold;font-size:18px}
h2{background:#e9eef5;padding:6px 10px;border-left:5px solid #1f3a5f;font:bold 17px sans-serif;margin-top:30px}
h3.src{font:bold 15px sans-serif;color:#1f3a5f;border-bottom:1px solid #ccd;margin-top:22px}
p{margin:0 0 9px;white-space:pre-wrap}
mark{background:#ffe27a;padding:0 2px}
.caps{margin-top:10px;font-size:13px;color:#444}
"""


def _highlight(escaped: str) -> str:
    return _TERM_RE.sub(lambda m: "<mark>" + m.group(0) + "</mark>", escaped)


def _render_block(text: str) -> str:
    """Paragraphs of one section; '=== ... ===' lines become small headings; terms are highlighted."""
    out = []
    for para in re.split(r"\n\s*\n|\n(?=\s*=== )", text.strip()):
        para = para.strip()
        if not para:
            continue
        if _HEADING.match(para.splitlines()[0].strip()):
            first, _, rest = para.partition("\n")
            out.append('<h3 class="src">' + H.escape(first.strip("= ").strip()) + "</h3>")
            para = rest.strip()
            if not para:
                continue
        out.append("<p>" + _highlight(H.escape(para)) + "</p>")
    return "\n".join(out)


def _sections(text: str) -> list:
    """(title, body) pairs: Item 1 business text, segment and KPI text, other furnished materials."""
    tenk, _, other = text.partition(T.BUNDLE_MARKER)
    item1, _, seg = tenk.partition(T.SEGMENT_MARKER)
    secs = [("1. Business description (10-K Item 1)", item1), ("2. Segment and operating-statistics text (10-K)", seg),
            ("3. Other furnished materials (10-Qs, earnings releases, presentations)", other)]
    return [(t, b) for t, b in secs if b.strip()]


def render_row_html(n: int, total: int, text: str, tag: str, prev_href: str | None, next_href: str | None) -> str:
    nav = ""
    if prev_href:
        nav += f'<a href="{prev_href}">&larr; Previous</a>'
    nav += '<span class="gap"></span>'
    if next_href:
        nav += f'<a href="{next_href}">Next &rarr;</a>'
    body = "\n".join(f"<h2>{H.escape(t)}</h2>\n{_render_block(b)}" for t, b in _sections(text))
    cheat = "".join(f"<tr><td>{s}</td><td>{H.escape(d)}</td></tr>" for s, d in CHEAT)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Row {n} of {total}</title><style>{CSS}</style></head><body>"
            f'<div class="bar"><strong>Row {n} of {total}</strong>{nav}</div>'
            f'<div class="wrap"><main>{body}</main>'
            f'<aside><h3>Your score = share of the profit you could rebuild as price x quantity</h3><table>{cheat}</table>'
            f'<div class="caps">{H.escape(CAPS)}</div>'
            f"<p><b>Go back to the Excel row and type your score (0 to 5).</b></p></aside></div></body></html>")


def _guide(wb: openpyxl.Workbook, sheets: dict) -> None:
    ws = wb.active
    ws.title = "Guide"
    lines = ["How to do the audit", "",
             "1. Open the 'Representative' tab first (100 filings). Do the 'Diagnostic' tab only afterwards.",
             "2. In each row click the blue link in the OPEN column. The filing opens in your browser. Nothing else to click.",
             "3. Ask: could an outside investor rebuild the profit-making lines as price x quantity from this text alone?",
             "4. Type one whole number 0 to 5 into the yellow YOUR SCORE column. Add a note if you hesitated between two scores.",
             "5. Use the Previous / Next buttons in the browser, or just click the next row's link. Save the workbook often.",
             "6. Do NOT open audit_KEY_do_not_open.csv: it holds the model's scores and would bias you.",
             "7. Keep this workbook and the 'reader' folder together in the same folder, or the links will break.", "",
             "Scale"] + [f"   {s} = {d}" for s, d in CHEAT] + ["", CAPS, "",
             "Company names are masked on purpose. Judge only the text in front of you, not what you know about the company."]
    for i, line in enumerate(lines, 1):
        ws.cell(i, 1, line)
    ws["A1"].font = Font(bold=True, size=16)
    ws.column_dimensions["A"].width = 130
    for r in range(1, len(lines) + 1):
        ws.cell(r, 1).alignment = Alignment(wrap_text=True, vertical="top")


def build_audit_workbook(sheets: dict, out_dir: Path, name: str = "AUDIT.xlsx") -> Path:
    """sheets: {tab name: (file tag, DataFrame with unit_id and text)}. Writes reader pages to out_dir/reader and the workbook
    to out_dir/name; returns the workbook path. Hyperlinks are relative so the folder can be moved or zipped."""
    out_dir = Path(out_dir)
    reader = out_dir / "reader"
    reader.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    _guide(wb, sheets)
    yellow = PatternFill("solid", fgColor="FFF2A8")
    for tab, (tag, df) in sheets.items():
        total = len(df)
        ws = wb.create_sheet(tab)
        last = FIRST_ROW + total - 1
        ws["A1"] = f"{tab}: {total} filings"
        ws["A1"].font = Font(bold=True, size=14)
        ws["A2"] = "Click OPEN, read, type a whole number 0-5 in the yellow column. Company names are masked."
        ws["A3"] = f'=IF(COUNT(C{FIRST_ROW}:C{last})=0,"Scored: 0 of {total}","Scored: "&COUNT(C{FIRST_ROW}:C{last})&" of {total}")'
        ws["A3"].font = Font(bold=True, color="1F3A5F")
        for c, h in enumerate(COLS, 1):
            cell = ws.cell(4, c, h)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F3A5F")
        dv = DataValidation(type="whole", operator="between", formula1="0", formula2="5", allow_blank=True,
                            showErrorMessage=True, errorTitle="Score", error="Enter a whole number from 0 to 5.")
        ws.add_data_validation(dv)
        for i, row in enumerate(df.itertuples(index=False), 1):
            r = FIRST_ROW + i - 1
            page = f"{tag}_{i:03d}.html"
            prev_href = f"{tag}_{i - 1:03d}.html" if i > 1 else None
            next_href = f"{tag}_{i + 1:03d}.html" if i < total else None
            (reader / page).write_text(render_row_html(i, total, row.text, tag, prev_href, next_href), encoding="utf-8")
            ws.cell(r, 1, i)
            link = ws.cell(r, 2, f"Open {tab.lower()} {i}")
            link.hyperlink = f"reader/{page}"
            link.font = Font(color="0563C1", underline="single")
            ws.cell(r, 3).fill = yellow
            dv.add(ws.cell(r, 3))
            ws.cell(r, 5, f'=IF(C{r}<>"","done","")')
            ws.cell(r, 6, row.unit_id)
        for col, w in zip("ABCDEF", (6, 22, 18, 50, 8, 12)):
            ws.column_dimensions[col].width = w
        ws.column_dimensions[get_column_letter(COLS.index("id (hidden)") + 1)].hidden = True
        ws.freeze_panes = ws.cell(FIRST_ROW, 1)
    path = out_dir / name
    wb.save(path)
    return path
