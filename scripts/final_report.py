"""W3-12: the 12-page technical report, built from docs/report/final_report.md and checked (ADR-047).

Usage:
    python scripts/final_report.py            # check, then print the PDF with Chrome and record the build
    python scripts/final_report.py --check    # no browser: the checks below, and that the committed PDF is current

The report is written by hand in Markdown (docs/report/final_report.md). This script
  1. checks that every number in a section appears in a file the section cites on its "Sources:" line (commas
     ignored), so every figure is traceable to a committed file; small integers (up to 12), years and numbers that
     are part of a name (W3-03, ADR-039, Qwen3-8B, FY2024, R1) are not checked;
  2. checks that every cited path, and every path the rubric checklist (docs/report/rubric_checklist.md) names,
     exists in the repository;
  3. renders the Markdown to one self-contained HTML page (the figures inlined as data URIs, a print stylesheet)
     and prints it to docs/report/final_report.pdf with headless Chrome, once whole and once without the appendix;
  4. records in docs/report/final_report.build.json the SHA-256 of that HTML and both page counts.
--check repeats 1 and 2, renders the HTML again and compares its hash with the record (so a report edited without
rebuilding the PDF fails), checks the PDF's page count against the record and the body (everything before the
appendix) against the 12-page limit. Only the build needs Chrome; CI runs --check (the tests do the same).
Exit codes: 0 = built (or current with --check), 1 = a check failed, 2 = no Chrome found.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO / "docs" / "report"
SOURCE = REPORT_DIR / "final_report.md"
PDF = REPORT_DIR / "final_report.pdf"
RECORD = REPORT_DIR / "final_report.build.json"
CHECKLIST = REPORT_DIR / "rubric_checklist.md"
MAX_BODY_PAGES = 12
APPENDIX = "<!-- appendix -->"
PAGEBREAK = "<!-- pagebreak -->"

# A number as written in prose or a table: 1,563 · 0.774 · 42 (in 42/48) · 5.54 · 48.25 (in 48.25M). Not part of a
# name: no letter, digit, hyphen, dot or comma directly before it (W3-03, ADR-039, FY2024, Qwen3-8B, v1.0), and never
# a piece of a longer number (the match runs to the number's last digit).
NUMBER = re.compile(r"(?<![A-Za-z0-9\-.,])\d[\d,]*(?:\.\d+)?(?!\d)(?!\.\d)")
# A count out of a total, 42/48 or 42 / 48, or "13 of 13": checked as one claim, wherever its parts are small
FRACTION = re.compile(r"(?<![A-Za-z0-9\-.,/])(\d[\d,]*)\s*(?:/|of)\s*(\d[\d,]*)(?![\d/]|\.\d)")
SOURCES_LINE = re.compile(r"^\*Sources:\*\s*(.*)$", re.M)
NUMERIC_CELL = re.compile(r"[\d\s.,/%()\-+<=>*–]*")    # a table cell holding only figures (bold allowed)
PATH = re.compile(r"`([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)`")


def norm(token: str) -> str:
    """A number without its thousands separators and trailing zeros after the point: 1,563 -> 1563, 3.50 -> 3.5."""
    t = token.replace(",", "")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t


def checked(token: str) -> bool:
    """Whether a number in the report must be found in its sources: not a small count (up to 12), and not a year (a
    bare four-digit number from 1990 to 2030; "2,000" with its separator is a count and is checked)."""
    t = norm(token)
    if "." in t:
        return True
    value = int(t)
    return value > 12 and not ("," not in token and 1990 <= value <= 2030)


def numbers_in(text: str) -> set[str]:
    """The text's numbers, and its fractions as "a/b" (so 42/48 is found only where a source states 42/48)."""
    found = {norm(m.group(0)) for m in NUMBER.finditer(text)}
    return found | {f"{norm(m.group(1))}/{norm(m.group(2))}" for m in FRACTION.finditer(text)}


def claims_in(text: str) -> set[str]:
    """The numbers a section claims: each fraction whole, and every other number that checked() selects."""
    fractions = {f"{norm(m.group(1))}/{norm(m.group(2))}" for m in FRACTION.finditer(text)}
    rest = FRACTION.sub(" ", text)
    return fractions | {norm(m.group(0)) for m in NUMBER.finditer(rest) if checked(m.group(0).rstrip(","))}


@dataclass
class Section:
    title: str
    text: str
    sources: list[str] = field(default_factory=list)


def sections(markdown: str) -> list[Section]:
    """The report's level-2 sections, each with the paths on its Sources line(s). Text before the first one is the
    front matter (its numbers are checked against the paths it cites, if any)."""
    out, title, lines = [], "front matter", []
    for line in markdown.splitlines():
        if line.startswith("## "):
            out.append(Section(title, "\n".join(lines)))
            title, lines = line[3:].strip(), []
        else:
            lines.append(line)
    out.append(Section(title, "\n".join(lines)))
    for s in out:
        s.sources = [p for m in SOURCES_LINE.finditer(s.text) for p in PATH.findall(m.group(1))]
    return out


def strip_non_claims(text: str) -> str:
    """The text whose numbers are claims: without the Sources and Owner lines, HTML comments, code spans holding
    paths or commands, and link targets."""
    text = SOURCES_LINE.sub("", text)
    text = re.sub(r"^\*Owner[^\n]*$", "", text, flags=re.M)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return text


def untraced(markdown: str, root: Path = REPO) -> list[str]:
    """Every problem with the report's traceability: a cited path that does not exist, a section with numbers but no
    sources, a number its sources do not contain."""
    problems = []
    cache: dict[str, set[str]] = {}
    for s in sections(markdown):
        claims = claims_in(strip_non_claims(s.text))
        missing = [p for p in s.sources if not (root / p).is_file()]
        problems += [f"{s.title}: cites {p}, which does not exist" for p in missing]
        if claims and not s.sources:
            problems.append(f"{s.title}: states numbers ({', '.join(sorted(claims)[:5])} ...) but cites no source")
            continue
        found: set[str] = set()
        for p in s.sources:
            if p not in cache and (root / p).is_file():
                cache[p] = numbers_in((root / p).read_text(encoding="utf-8", errors="replace"))
            found |= cache.get(p, set())
        problems += [f"{s.title}: {n} is in none of its sources" for n in sorted(claims - found)]
    return problems


def checklist_problems(text: str, root: Path = REPO) -> list[str]:
    """Paths the rubric checklist names that do not exist (a directory may be named with a trailing slash)."""
    out = []
    for p in sorted(set(re.findall(r"`([A-Za-z0-9_./\-]+)`", text))):
        if "/" not in p and "." not in p:
            continue                                   # a command or a word in code style, not a path
        if not (root / p.rstrip("/")).exists():
            out.append(f"rubric checklist: {p} does not exist")
    return out


# ----------------------------------------------------------------------------------------------- Markdown -> HTML
def inline(text: str) -> str:
    """Escaped text with `code`, **bold**, *italic* and [links](url)."""
    parts = re.split(r"(`[^`]*`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        t = html.escape(part)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
        out.append(t)
    return "".join(out)


def data_uri(path: Path) -> str:
    kind = {".svg": "image/svg+xml", ".png": "image/png"}[path.suffix.lower()]
    data = path.read_bytes()
    if kind == "image/svg+xml":
        data = data.replace(b"\r\n", b"\n")          # the same hash on a checkout with CRLF line endings
    return f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"


def to_html(markdown: str, base: Path = REPORT_DIR) -> str:
    """The report's Markdown subset as HTML: headings, paragraphs, bullet and numbered lists (two levels), tables,
    figures, the Sources and Owner lines, page breaks and the appendix marker."""
    out: list[str] = []
    para: list[str] = []
    lists: list[str] = []                               # open list tags, innermost last
    table: list[list[str]] = []

    def close_para() -> None:
        if para:
            text = " ".join(para)
            cls = ' class="meta"' if re.match(r"\*(Sources|Owner)", text) else ""
            out.append(f"<p{cls}>{inline(text)}</p>")
            para.clear()

    def close_lists(depth: int = 0) -> None:
        while len(lists) > depth:
            out.append("</li>")
            out.append(f"</{lists.pop()}>")

    def close_table() -> None:
        if table:
            head, *body = table
            # A column of figures is right-aligned, a column with words left-aligned
            num = [bool(body) and all(NUMERIC_CELL.fullmatch(r[i] if i < len(r) else "") for r in body)
                   for i in range(len(head))]

            def cell(tag: str, i: int, c: str) -> str:
                return f'<{tag} class="num">{inline(c)}</{tag}>' if num[i] else f"<{tag}>{inline(c)}</{tag}>"

            out.append("<table><thead><tr>" + "".join(cell("th", i, c) for i, c in enumerate(head)) +
                       "</tr></thead><tbody>")
            out.extend("<tr>" + "".join(cell("td", i, c) for i, c in enumerate(r)) + "</tr>" for r in body)
            out.append("</tbody></table>")
            table.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            close_para()
            close_lists()
            if not re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", line):
                table.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        close_table()
        if line == PAGEBREAK:
            close_para()
            close_lists()
            out.append('<div class="pagebreak"></div>')
        elif line == APPENDIX:
            close_para()
            close_lists()
            out.append('<div class="appendix"></div>')
        elif line.startswith('<p class="') and line.endswith("</p>"):
            close_para()
            close_lists()
            out.append(line)                             # a styled line written as HTML (the subtitle)
        elif not line or line.startswith("<!--"):
            close_para()
            if not line:
                continue
        elif m := re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line):
            close_para()
            close_lists()
            out.append(f'<figure><img src="{data_uri((base / m.group(2)).resolve())}" alt="{html.escape(m.group(1))}">'
                       f"<figcaption>{inline(m.group(1))}</figcaption></figure>")
        elif m := re.fullmatch(r"(#{1,4})\s+(.*)", line):
            close_para()
            close_lists()
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif m := re.fullmatch(r"( *)(-|\d+\.)\s+(.*)", line):
            close_para()
            depth = len(m.group(1)) // 2 + 1
            tag = "ol" if m.group(2)[0].isdigit() else "ul"
            if len(lists) < depth:
                while len(lists) < depth:
                    out.append(f"<{tag}>")
                    lists.append(tag)
            else:
                close_lists(depth)
                out.append("</li>")
            out.append(f"<li>{inline(m.group(3))}")
        elif lists and raw.startswith("  "):
            out[-1] += " " + inline(line.strip())        # a list item's next line
        else:
            close_lists()
            para.append(line.strip())
    close_para()
    close_lists()
    close_table()
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm;
        @bottom-center { content: counter(page); font: 8pt "Segoe UI", Arial, sans-serif; color: #777; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 10.4pt/1.38 Cambria, Georgia, "Times New Roman", serif; color: #1e2b2f; margin: 0; }
h1 { font: bold 19pt/1.15 "Segoe UI", Arial, sans-serif; color: #12343b; margin: 0 0 2mm; }
h2 { font: bold 13pt/1.2 "Segoe UI", Arial, sans-serif; color: #12343b; margin: 4.2mm 0 1.4mm;
     border-bottom: 1.2pt solid #1f7a72; padding-bottom: 0.6mm; break-after: avoid; }
h3 { font: bold 10pt/1.2 "Segoe UI", Arial, sans-serif; color: #1f7a72; margin: 2.6mm 0 0.8mm; break-after: avoid; }
p { margin: 0 0 1.5mm; text-align: justify; hyphens: auto; }
p.meta { font: 7.9pt/1.3 "Segoe UI", Arial, sans-serif; color: #5b6b73; text-align: left; margin: 0.4mm 0 1.6mm; }
p.meta code { font-size: 7.2pt; }
ul, ol { margin: 0 0 1.6mm; padding-left: 5mm; }
li { margin: 0 0 0.6mm; }
code { font: 8.2pt Consolas, "Courier New", monospace; background: #eef4f3; padding: 0 0.6mm; border-radius: 1mm; }
table { border-collapse: collapse; width: 100%; margin: 1mm 0 2.4mm; font: 8.6pt/1.28 "Segoe UI", Arial, sans-serif;
        break-inside: avoid; }
th { background: #12343b; color: #fff; text-align: left; padding: 0.9mm 1.4mm; font-weight: 600; }
td { padding: 0.7mm 1.4mm; border-bottom: 0.5pt solid #d9e3e1; vertical-align: top; }
tr:nth-child(even) td { background: #f3f8f7; }
td.num, th.num { text-align: right; white-space: nowrap; }
figure { margin: 1.5mm 0 2.6mm; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 74mm; }
th code { background: transparent; color: inherit; padding: 0; }
figcaption { font: italic 7.8pt "Segoe UI", Arial, sans-serif; color: #5b6b73; margin-top: 0.8mm; }
a { color: #1f7a72; text-decoration: none; }
.subtitle { font: 10pt "Segoe UI", Arial, sans-serif; color: #5b6b73; margin-bottom: 1mm; text-align: left; }
.pagebreak, .appendix { break-before: page; height: 0; }
"""


def page(markdown: str) -> str:
    """The report as one HTML document (deterministic: the same Markdown and figures give the same bytes)."""
    body = to_html(markdown)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>FinSight technical report</title>"
            f"<style>{CSS}</style></head><body>\n{body}\n</body></html>\n")


def body_only(markdown: str) -> str:
    """The report without its appendix (the part the page limit applies to)."""
    return markdown.split(APPENDIX, 1)[0]


def pdf_pages(data: bytes) -> int:
    """The number of pages of a PDF Chrome printed (its page objects are not inside object streams)."""
    return len(re.findall(rb"/Type\s*/Page(?![a-zA-Z])", data))


def html_sha256(markdown: str) -> str:
    return hashlib.sha256(page(markdown).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------- build and check
def problems(markdown: str) -> list[str]:
    out = untraced(markdown)
    if APPENDIX not in markdown:
        out.append(f"the report has no appendix marker ({APPENDIX})")
    if CHECKLIST.exists():
        out += checklist_problems(CHECKLIST.read_text(encoding="utf-8"))
    else:
        out.append(f"{CHECKLIST.relative_to(REPO).as_posix()} is missing")
    return out


def chrome() -> str | None:
    """A Chrome or Edge executable: CHROME, then the usual install paths, then the PATH."""
    candidates = [os.environ.get("CHROME", ""),
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        if found := shutil.which(name):
            return found
    return None


def print_pdf(browser: str, document: str, out: Path) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_bytes(document.encode("utf-8"))
        target = Path(tmp) / "report.pdf"
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={target}", src.as_uri()],
                       check=True, capture_output=True, timeout=180)
        data = target.read_bytes()
    out.write_bytes(data)
    return data


def build() -> int:
    markdown = SOURCE.read_text(encoding="utf-8")
    written = {f"{p.relative_to(REPO).as_posix()} does not exist" for p in (PDF, RECORD)}
    if found := [p for p in problems(markdown) if not any(p.endswith(w) for w in written)]:
        print("\n".join(f"ERROR: {p}" for p in found), file=sys.stderr)
        return 1
    browser = chrome()
    if browser is None:
        print("ERROR: no Chrome or Edge found (set CHROME to its path)", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        body_pages = pdf_pages(print_pdf(browser, page(body_only(markdown)), Path(tmp) / "body.pdf"))
    total = pdf_pages(print_pdf(browser, page(markdown), PDF))
    record = {"html_sha256": html_sha256(markdown), "pages_body": body_pages, "pages_total": total}
    RECORD.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    print(f"written {PDF.relative_to(REPO).as_posix()}: {total} pages ({body_pages} before the appendix)")
    if body_pages > MAX_BODY_PAGES:
        print(f"ERROR: the body is {body_pages} pages, over the {MAX_BODY_PAGES}-page limit", file=sys.stderr)
        return 1
    return 0


def check() -> list[str]:
    markdown = SOURCE.read_text(encoding="utf-8")
    out = problems(markdown)
    if not RECORD.exists() or not PDF.exists():
        return out + ["the PDF or its build record is missing: run python scripts/final_report.py"]
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    if record.get("html_sha256") != html_sha256(markdown):
        out.append("the PDF is stale (the report changed since it was built): run python scripts/final_report.py")
    if pdf_pages(PDF.read_bytes()) != record.get("pages_total"):
        out.append("the PDF's page count differs from its build record: run python scripts/final_report.py")
    if record.get("pages_body", MAX_BODY_PAGES + 1) > MAX_BODY_PAGES:
        out.append(f"the body is {record.get('pages_body')} pages, over the {MAX_BODY_PAGES}-page limit")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="check only; no browser needed")
    args = ap.parse_args(argv)
    if args.check:
        found = check()
        print("\n".join(f"ERROR: {p}" for p in found) or "final report current", file=sys.stderr if found else sys.stdout)
        return 1 if found else 0
    return build()


if __name__ == "__main__":
    sys.exit(main())
