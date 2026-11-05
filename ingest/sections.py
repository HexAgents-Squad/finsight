"""Split a filing's HTML into Item-level section documents (docs/week0_plan.txt §2.1, docs/spec.md §3).

How it works
1. HTML -> text: scripts, styles and the hidden inline-XBRL header are removed; each table row becomes
   "cell | cell | cell" so numbers keep their row and column context.
2. Candidate headings are lines that START with "Item <n>" (optionally "Part I/II"). Cross-references inside
   sentences ("see Item 7") do not start a line.
3. Every heading appears twice (table of contents + body): for each Item the occurrence with the LONGEST section
   wins. TOC rows ending in a page number are ignored, and a bare "Item 7" line counts only if the next line is
   that Item's title (some filers print "Item 7" as a running page header).
4. 10-Q item numbers repeat in Part I and II; the Part is taken from the title ("Risk Factors" -> Part II).
5. Section-mapping rule (spec §3): when the MD&A or the financial statements are not under their Item heading
   (bank annual reports appended after Item 15 or filed as Exhibit 13, NVIDIA's statements after Item 15, no
   Item headings at all), the regions starting at "Management's Discussion and Analysis" and at the financial
   statements are labelled Item 7 / Item 8 (10-Q: Part I Item 2 / Item 1), with source_location telling where
   the text physically is. Text that maps to no Item is dropped and counted in the parse report.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

MAIN_DOCUMENT = "main document"
EXHIBIT = "Exhibit 13"
MIN_SECTION_CHARS = 300          # shorter Items ("Not applicable", "[Reserved]") are dropped (spec §3)
MIN_ITEMS_FOR_HEADINGS = 4       # fewer Item headings than this = the filing does not use them
STUB_CHARS = 3_000               # an MD&A / financial-statement Item shorter than this is only a pointer
MIN_MDNA_CHARS = 20_000          # financial statements start at least this far after the MD&A heading
HEADING_MAX_CHARS = 160          # a heading line is short; longer lines must start with the Item's title
REGION_HEADING_MAX_CHARS = 150
TITLE_WORDS = 3                  # a bare "Item N" line is a heading only if the next line starts with this many
                                 # whole words of the Item's title

TENK_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity "
         "Securities",
    "6": "[Reserved]", "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance", "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services", "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
TENQ_TITLES = {
    ("I", "1"): "Financial Statements",
    ("I", "2"): "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ("I", "3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("I", "4"): "Controls and Procedures",
    ("II", "1"): "Legal Proceedings", ("II", "1A"): "Risk Factors",
    ("II", "2"): "Unregistered Sales of Equity Securities and Use of Proceeds",
    ("II", "3"): "Defaults Upon Senior Securities", ("II", "4"): "Mine Safety Disclosures",
    ("II", "5"): "Other Information", ("II", "6"): "Exhibits",
}
MDNA_KEY = {"10-K": "Item 7", "10-Q": "Part I Item 2"}
FS_KEY = {"10-K": "Item 8", "10-Q": "Part I Item 1"}

ITEM_LINE = re.compile(r"^\s*(?:part\s+(i{1,3}|iv)\b[\s.,:\-–—]*)?item[\s\xa0]*(\d{1,2}[a-c]?)\b\s*[.:\-–—|]*\s*(.{0,120})",
                       re.IGNORECASE)
PART_LINE = re.compile(r"^\s*part\s+(i{1,3}|iv)\b\.?\s*(?:[-–—:]\s*)?(?:financial information|other information)?\s*$",
                       re.IGNORECASE)
# Table-of-contents rows end in a page reference: "23", "23-25", "F-1", or "N/A"/"None" for an Item not in use
TOC_PAGE_NUMBER = re.compile(r"\|\s*(?:\d{1,3}(?:\s*[-–]\s*\d{1,3})?|[A-Z]-\d{1,3}|n/?a|none)\s*$", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9']+")
# A table-of-contents ROW has only text cells and page-reference cells, in any order ("Executive Summary | 27",
# "48 | Introduction | 171 | Notes", "Note 26-Leases | 212"); a figure ("391,035", "12.5", "$") or an exhibit
# number ("101.INS", an exhibit-index row) makes it data.
TOC_PAGE_CELL = re.compile(r"^(?:\d{1,3}(?:\s*[-–]\s*\d{1,3})?|[A-Z]-\d{1,3}|n/?a|none)$", re.IGNORECASE)
DATA_FIGURE = re.compile(r"\d,\d{3}|\d\.\d|[$%()]|\b\d{1,3}\.[A-Z]{3}\b")
HAS_LETTER = re.compile(r"[A-Za-z]")
LEADING_NUMBER = re.compile(r"\d+")
# A table-of-contents BLOCK is at least MIN_TOC_ROWS TOC rows in a row whose page numbers mostly go up (at least
# TOC_ASCENDING_SHARE of consecutive pairs: two-column TOCs restart once; an executive table "Name | 63 | CEO" has
# ages in any order)
MIN_TOC_ROWS = 4
TOC_ASCENDING_SHARE = 0.75
TITLE_LOOKAHEAD_LINES = 3        # lines after a bare "Item N" searched for its title
TITLE_MAX_LINES = 2              # an Item title wraps over at most this many lines
# Words that TOC entries use. A block recognised only by its position at a section's edge must use them in most
# rows, so a small data table there ("Santa Clara, California | 12") is never taken for a TOC.
TOC_TITLE_WORD = re.compile(r"\b(?:statements?|notes?|management|discussion|analysis|overview|results|report|"
                            r"financial|balance|income|cash\s+flows?|equity|risk|summary|highlights|controls|"
                            r"introduction|segments?|glossary|index|item|part|liquidity|capital|critical|"
                            r"accounting|supplementary)\b", re.IGNORECASE)   # not "executive": officer tables
TOC_LOOKAHEAD_LINES = 6          # lines after a region heading checked for the start of a TOC block
TOC_LOOKAHEAD_MIN_ROWS = 3
TOC_EDGE_LINES = 8               # a TOC block starting within this many lines of a section's start/end
TOC_HEADER = re.compile(r"^(?:table\s+of\s+contents|index(?:\s+to\b.*)?|contents|page(?:\s+no\.?)?)$", re.IGNORECASE)
TOC_ENTRY_WORD = re.compile(r"^\s*(?:\d{1,3}\s*\|\s*)?(?:item|note|part)\s+[\divx]", re.IGNORECASE)
TOC_SUBHEADING = re.compile(r"^(?:part\s+i{1,3}\b.*|table\s+of\s+contents|page)$", re.IGNORECASE)
NOT_A_TITLE = re.compile(r"^(?:of|and|in|to|above|below|is|are)\b|^\d", re.IGNORECASE)
# 10-Q item titles are unique, so they tell the Part even when a TOC leaves the running Part wrong
TENQ_PART_BY_TITLE = [
    (re.compile(r"^(?:condensed\s+)?(?:consolidated\s+)?financial\s+statements", re.I), "I"),
    (re.compile(r"^management", re.I), "I"),
    (re.compile(r"^quantitative", re.I), "I"),
    (re.compile(r"^controls", re.I), "I"),
    (re.compile(r"^legal", re.I), "II"),
    (re.compile(r"^risk\s+factors", re.I), "II"),
    (re.compile(r"^unregistered", re.I), "II"),
    (re.compile(r"^defaults", re.I), "II"),
    (re.compile(r"^mine", re.I), "II"),
    (re.compile(r"^other\s+information", re.I), "II"),
    (re.compile(r"^exhibits", re.I), "II"),
]
MDNA_HEADING = re.compile(r"^management[’'`]?s?\s+discussion\s+and\s+analysis\b", re.IGNORECASE)
FS_HEADING = re.compile(r"^(?:report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm"
                        r"|(?:condensed\s+)?consolidated\s+(?:statements?\s+of\s+(?:income|operations|earnings)"
                        r"|income\s+statements?))\b", re.IGNORECASE)
# Risk factors printed inside a bank's MD&A/annual report (Citi, Wells Fargo): the region from a heading line that
# is exactly "Risk Factors" to the next major heading. Their tables of contents are not reliable enough to find
# the end (Wells Fargo's exhibit has none), so the end headings are listed explicitly.
RISK_FACTORS_HEADING = re.compile(r"^risk\s+factors$", re.IGNORECASE)
RISK_FACTORS_END = re.compile(r"^(?:controls\s+and\s+procedures|report\s+of\s+independent\s+registered"
                              r"|(?:condensed\s+)?consolidated\s+(?:statements?|balance\s+sheets?)\b"
                              r"|financial\s+statements|glossary|managing\s+global\s+risk|human\s+capital"
                              # Citi's ESG sections, as whole lines only: "Climate Change Presents Various ...
                              # Risks to Citi" is a risk factor and must not end the section
                              r"|(?:climate\s+change\s+and\s+net\s+zero|net\s+zero\s+and\s+sustainability"
                              r"|sustainability)$)", re.IGNORECASE)
# A relocated financial-statements region (Citi) is followed by other 10-K sections and the exhibit index;
# those end it (whole heading lines only, so phrases inside the notes do not)
STATEMENTS_END = re.compile(r"^(?:properties|supervision,?\s+(?:and\s+)?regulation\b.*|unregistered\s+sales\b.*"
                            r"|exhibit\s+index|exhibits?|legal\s+proceedings)$", re.IGNORECASE)
RISK_KEY = {"10-K": "Item 1A", "10-Q": "Part II Item 1A"}
# Controls and procedures at the end of a bank's MD&A (Wells Fargo); it usually runs to the end of that region
CONTROLS_HEADING = re.compile(r"^controls\s+and\s+procedures$", re.IGNORECASE)
CONTROLS_END = re.compile(r"^(?:report\s+of\s+independent\s+registered|(?:condensed\s+)?consolidated\s+"
                          r"(?:statements?|balance\s+sheets?)\b|financial\s+statements|glossary|risk\s+factors$)",
                          re.IGNORECASE)
CONTROLS_KEY = {"10-K": "Item 9A", "10-Q": "Part I Item 4"}
# An Item 1A/9A is only a pointer when it is short AND refers elsewhere ("can be found ... under Risk Factors");
# a genuine short Item (MSFT/GS 10-Q Item 4, ~800 chars) is never replaced
POINTER_CHARS = 1_000
POINTER_WORDING = re.compile(r"\b(?:see|refer(?:red)?\s+to|incorporated\s+(?:herein\s+)?by\s+reference|"
                             r"set\s+forth|can\s+be\s+found|included\s+(?:in|under)|appears?\s+(?:in|under)|"
                             r"page\s+\d+)", re.IGNORECASE)
# The signature page (heading + its standard legend) ends a filing's text; without this the last Item, or a
# relocated Item 8, would carry the signatures and anything printed after them
SIGNATURE_PAGE = re.compile(r"^[ \t]*signatures?[ \t]*\n+[ \t]*(?:pursuant\s+to|under)\s+the\s+requirements",
                            re.IGNORECASE | re.MULTILINE)          # Pfizer writes "Under the requirements"
TABLE_FILLER_CELLS = frozenset({"$", "%", ")"})
SPACE_RUN = re.compile(r"[ \t]+")
SPACE_AROUND_NEWLINE = re.compile(r" *\n *")
BLANK_LINE_RUN = re.compile(r"\n{3,}")
TENQ_KEY = re.compile(r"Part (\w+) Item (\w+)$")
ZERO_WIDTH = "​"


def _norm_quote(s: str) -> str:
    return s.replace("’", "'").lower()


# ----------------------------------------------------------------------------------------------- HTML -> text
def html_to_text(html: bytes | str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    hidden = soup.find_all(lambda t: t.name and (t.name.lower() == "ix:header"
                                                 or "display:none" in (t.get("style") or "").replace(" ", "").lower()))
    for tag in hidden:
        tag.decompose()
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c and c not in TABLE_FILLER_CELLS]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n" if rows else "\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]):
        block.insert_after("\n")
    text = soup.get_text().replace("\xa0", " ").replace(ZERO_WIDTH, "")
    text = SPACE_RUN.sub(" ", text)
    text = SPACE_AROUND_NEWLINE.sub("\n", text)
    return BLANK_LINE_RUN.sub("\n\n", text).strip()


# ----------------------------------------------------------------------------------------------- Item headings
def item_title(form: str, key: str) -> str:
    if form == "10-K":
        return TENK_TITLES.get(key.removeprefix("Item "), "")
    found = TENQ_KEY.match(key)
    return TENQ_TITLES.get((found.group(1), found.group(2)), "") if found else ""


def find_headings(lines: list[str], form: str) -> list[tuple[int, str]]:
    """(line index, Item key) for every line that looks like an Item heading."""
    out, part = [], "I"
    for i, line in enumerate(lines):
        if TOC_PAGE_NUMBER.search(line) or is_toc_row(line):
            continue                         # a TOC row ("Item 7. | MD&A | 30", "Item 7 (continued) | 31")
        part_line = PART_LINE.match(line) if len(line) < HEADING_MAX_CHARS else None
        if part_line:
            part = part_line.group(1).upper()
            continue
        m = ITEM_LINE.match(line)
        if not m:
            continue
        if m.group(1):
            part = m.group(1).upper()
        item = m.group(2).upper()
        rest = m.group(3).strip().lstrip("|").strip()
        if not rest:                         # bare "Item 7": a heading only if the next line is its title
            following = [ln.strip() for ln in lines[i + 1:i + 1 + TITLE_LOOKAHEAD_LINES] if ln.strip()]
            nxt = following[0] if following else ""
            titles = [TENK_TITLES.get(item, "")] if form == "10-K" else [
                t for (_, it), t in TENQ_TITLES.items() if it == item]
            if not any(t and starts_with_title(nxt, t) for t in titles):
                continue
            # A title ending in a page number is a TOC entry, also when the title wraps and the page number is
            # on its second line (GS: "Changes in ... on Accounting" / "and Financial Disclosure | 240")
            title_end = next((ln for ln in following[:TITLE_MAX_LINES] if "|" in ln), "")
            if is_toc_row(title_end) or TOC_PAGE_NUMBER.search(title_end):
                continue
            rest = nxt
        if NOT_A_TITLE.match(_norm_quote(rest)):
            continue                         # sentence fragment or cross-reference, not a heading
        if form == "10-K":
            if item not in TENK_TITLES:
                continue
            title, key = TENK_TITLES[item], f"Item {item}"
        else:
            for pattern, inferred in TENQ_PART_BY_TITLE:
                if pattern.match(rest):
                    part = inferred
                    break
            if (part, item) not in TENQ_TITLES:
                continue
            title, key = TENQ_TITLES[(part, item)], f"Part {part} Item {item}"
        if len(line) > HEADING_MAX_CHARS and not starts_with_title(rest, title):
            continue                         # long line: a heading only if it starts with the Item's own title
        out.append((i, key))
    return out


def starts_with_title(text: str, title: str) -> bool:
    """True if `text` begins with the first TITLE_WORDS whole words of `title` ("Financial Instruments" is not
    "Financial Statements and Supplementary Data")."""
    words, title_words = WORD.findall(_norm_quote(text)), WORD.findall(_norm_quote(title))
    n = min(TITLE_WORDS, len(title_words))
    return n > 0 and words[:n] == title_words[:n]


def heading_offsets(text: str, pattern: re.Pattern, include_toc: bool = False) -> list[int]:
    """Character offsets of short lines matching `pattern` (region headings such as "Management's Discussion").
    Unless `include_toc`, lines inside a table of contents (the row itself, or TOC_LOOKAHEAD_MIN_ROWS TOC rows
    among the next TOC_LOOKAHEAD_LINES lines) are not headings."""
    lines, pos, out = text.split("\n"), 0, []
    for n, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < REGION_HEADING_MAX_CHARS and pattern.match(stripped) and include_toc:
            out.append(pos)
        elif len(stripped) < REGION_HEADING_MAX_CHARS and pattern.match(stripped) and not is_toc_row(line):
            following = [ln for ln in lines[n + 1:n + 1 + TOC_LOOKAHEAD_LINES] if ln.strip()]
            if sum(map(is_toc_row, following)) < TOC_LOOKAHEAD_MIN_ROWS:
                out.append(pos)
        pos += len(line) + 1
    return out


def split_items(text: str, form: str) -> tuple[dict[str, str], str]:
    """({Item key: text}, unassigned text before the first Item, i.e. the cover page)."""
    lines = text.split("\n")
    offsets, pos = [], 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1
    headings = find_headings(lines, form)
    longest: dict[str, tuple[int, int]] = {}
    for n, (line_no, key) in enumerate(headings):
        start = offsets[line_no]
        end = offsets[headings[n + 1][0]] if n + 1 < len(headings) else len(text)
        if key not in longest or end - start > longest[key][1]:
            longest[key] = (start, end - start)
    starts = sorted((start, key) for key, (start, _) in longest.items())
    sections: dict[str, str] = {}
    unassigned = text[:starts[0][0]] if starts else text
    for n, (start, key) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(text)
        sections[key] = text[start:end].strip()
    return sections, unassigned


def split_last_item(sections: dict[str, str]) -> str:
    """When the MD&A or the statements are not under their Items, a bank's annual report (JPM) or NVIDIA's
    statements follow the exhibit list in Item 15: cut Item 15 where that report starts and return the rest as
    unassigned text for relocation. The report usually opens with its own contents list naming the MD&A, so the
    cut also counts MD&A / statements lines inside a TOC; otherwise its front matter (JPM's contents, financial
    highlights, stock-performance chart) would stay in Item 15. Item 15 is left alone when nothing follows it."""
    body = sections.get("Item 15", "")
    starts = [p for p in heading_offsets(body, MDNA_HEADING, include_toc=True)
              + heading_offsets(body, FS_HEADING, include_toc=True) if p > MIN_SECTION_CHARS]
    if not starts:
        return ""
    cut = min(starts)
    # The MD&A line may be the 3rd row of the report's contents list: start at its "Table of contents" header
    header = heading_offsets(body[:cut], TOC_HEADER, include_toc=True)
    preceding = body[header[-1]:cut].count("\n") if header else None
    if preceding is not None and preceding <= TOC_LOOKAHEAD_LINES and header[-1] > MIN_SECTION_CHARS:
        cut = header[-1]
    sections["Item 15"] = body[:cut].strip()
    return body[cut:]


# ----------------------------------------------------------------------------------------------- relocation
@dataclass
class Section:
    key: str
    text: str
    location: str = MAIN_DOCUMENT


@dataclass
class ParsedFiling:
    sections: list[Section]
    headings_found: int
    relocated: list[str] = field(default_factory=list)     # Item keys taken from outside their heading
    unassigned_chars: int = 0


def toc_row_page(line: str) -> int | None:
    """The first page number of a table-of-contents row, -1 for a row whose page is "N/A"/"F-1", or None if the
    line is not a TOC row (see TOC_PAGE_CELL)."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if len(cells) < 2 or DATA_FIGURE.search(line):
        return None
    is_page = [bool(TOC_PAGE_CELL.match(c)) for c in cells]
    if any(a and b for a, b in zip(is_page, is_page[1:], strict=False)):
        return None                          # two numbers side by side ("97 | 95 | 102") = a data table row
    pages = [c for c, p in zip(cells, is_page, strict=True) if p]
    texts = [c for c, p in zip(cells, is_page, strict=True) if not p]
    if not pages or not texts or not all(HAS_LETTER.search(t) for t in texts):
        return None
    first = LEADING_NUMBER.match(pages[0])
    return int(first.group()) if first else -1


def is_toc_row(line: str) -> bool:
    return toc_row_page(line) is not None


def toc_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """(first, last) line indexes of table-of-contents blocks: MIN_TOC_ROWS or more TOC rows, possibly with blank
    lines, "Page"/"Table of Contents"/"Part II" lines between them, whose page numbers mostly go up (at least
    TOC_ASCENDING_SHARE of consecutive pairs; two-column TOCs restart once). A small data table can have the same
    shape ("Cash and equivalents | 361"), so a block also has to look like navigation by its place: near the
    start or end of the text, under a "Table of Contents"/"Index"/"Page" line, or listing Items/Notes/Parts."""
    filled = [i for i, ln in enumerate(lines) if ln.strip()]
    near_start = set(filled[:TOC_EDGE_LINES])
    near_end = set(filled[-TOC_EDGE_LINES:])
    blocks, n = [], 0
    while n < len(lines):
        if not is_toc_row(lines[n]):
            n += 1
            continue
        first, last, pages = n, n, []
        while n < len(lines):
            stripped = lines[n].strip()
            page = toc_row_page(lines[n])
            if page is not None:
                pages.append(page)
                last = n
            elif stripped and not (TOC_SUBHEADING.match(stripped) or PART_LINE.match(stripped)):
                break
            n += 1
        numbered = [p for p in pages if p >= 0]
        pairs = list(zip(numbered, numbered[1:], strict=False))
        ascending = sum(a <= b for a, b in pairs) / len(pairs) if pairs else 1.0
        if len(pages) < MIN_TOC_ROWS or ascending < TOC_ASCENDING_SHARE:
            continue
        while first > 0 and TOC_SUBHEADING.match(lines[first - 1].strip()):
            first -= 1                           # take the "Table of Contents" / "Page" header along
        before = next((lines[i].strip() for i in range(first - 1, -1, -1) if lines[i].strip()), "")
        rows = [ln for ln in lines[first:last + 1] if is_toc_row(ln)]
        headed = bool(TOC_SUBHEADING.match(lines[first].strip()) or TOC_HEADER.match(before))
        lists_entries = sum(bool(TOC_ENTRY_WORD.match(r)) for r in rows) * 2 >= len(rows)
        at_edge = first in near_start or last in near_end
        titled = sum(bool(TOC_TITLE_WORD.search(r)) for r in rows) * 2 >= len(rows)
        if headed or lists_entries or (at_edge and titled):
            blocks.append((first, last))
    return blocks


def strip_toc(text: str) -> str:
    """The text without the blocks toc_blocks() takes for tables of contents. Those rules are built to leave
    data tables alone (checked block by block on the real corpus, and by tests), but they are heuristics.
    A heading on the first line is always kept."""
    lines = text.split("\n")
    drop = {i for first, last in toc_blocks(lines) for i in range(max(first, 1), last + 1)}
    if not drop:
        return text
    return BLANK_LINE_RUN.sub("\n\n", "\n".join(ln for i, ln in enumerate(lines) if i not in drop)).strip()


def carve_region(text: str, start_heading: re.Pattern, end_heading: re.Pattern,
                 to_end: bool = False) -> tuple[str, str] | None:
    """(text without the region, region) for the region from the first `start_heading` line to the next
    `end_heading` line (or to the end of the text if `to_end`), or None if there is no such region of at least
    MIN_SECTION_CHARS."""
    starts = heading_offsets(text, start_heading)
    if not starts:
        return None
    start = starts[0]
    ends = [p for p in heading_offsets(text, end_heading) if p > start + MIN_SECTION_CHARS]
    end = ends[0] if ends else (len(text) if to_end else None)
    if end is None or end - start < MIN_SECTION_CHARS:
        return None                                      # no clear end: leave the text where it is
    return (text[:start].rstrip() + "\n\n" + text[end:].lstrip()), text[start:end].strip()


def _regions(unassigned: str, exhibit_text: str | None) -> list[tuple[str, str, bool]]:
    """(location, text, is_exhibit) places where relocated MD&A / statements may sit, in search order."""
    out = [(MAIN_DOCUMENT, unassigned, False)]
    if exhibit_text:
        out.append((EXHIBIT, exhibit_text, True))
    return out


def parse_filing(main_text: str, form: str, exhibit_text: str | None = None) -> ParsedFiling:
    mdna_key, fs_key = MDNA_KEY[form], FS_KEY[form]
    sections, unassigned = split_items(main_text, form)
    headings_found = len(sections)
    if headings_found < MIN_ITEMS_FOR_HEADINGS:          # no usable Item headings: everything is unassigned
        sections, unassigned = {}, main_text
    if form == "10-K" and any(len(sections.get(k, "")) < STUB_CHARS for k in (mdna_key, fs_key)):
        unassigned += "\n\n" + split_last_item(sections)
    result = {key: Section(key, body) for key, body in sections.items()}
    relocated: list[str] = []
    used: dict[str, int] = {}                            # chars taken from each region

    def is_stub(key: str) -> bool:
        return len(result[key].text) < STUB_CHARS if key in result else True

    if not is_stub(mdna_key) and is_stub(fs_key):
        body = result[mdna_key].text
        inside = [p for p in heading_offsets(body, FS_HEADING) if p > MIN_MDNA_CHARS]
        if inside:                                       # bank 10-Qs: statements printed inside the MD&A Item
            result[mdna_key] = Section(mdna_key, body[:inside[0]].strip())
            result[fs_key] = Section(fs_key, body[inside[0]:].strip())
            relocated.append(fs_key)
        else:                                            # NVIDIA: Item 8 points to statements after Item 15
            for location, text, _ in _regions(unassigned, exhibit_text):
                found = heading_offsets(text, FS_HEADING)
                if found:
                    result[fs_key] = Section(fs_key, text[found[0]:].strip(), location)
                    used[location] = len(text) - found[0]
                    relocated.append(fs_key)
                    break
    elif is_stub(mdna_key):                              # bank 10-Ks: MD&A + statements outside the Items
        for location, text, is_exhibit in _regions(unassigned, exhibit_text):
            starts = heading_offsets(text, MDNA_HEADING)
            if not starts and is_exhibit:                # an annual-report exhibit opens with its MD&A
                starts = [0]
            if not starts:
                continue
            start = starts[0]
            fs = [p for p in heading_offsets(text, FS_HEADING) if p > start + MIN_MDNA_CHARS]
            end = fs[0] if fs else len(text)
            result[mdna_key] = Section(mdna_key, text[start:end].strip(), location)
            relocated.append(mdna_key)
            if fs and is_stub(fs_key):
                result[fs_key] = Section(fs_key, text[end:].strip(), location)
                relocated.append(fs_key)
            used[location] = len(text) - start
            break

    # Risk factors and controls & procedures printed inside a bank's MD&A / annual report get their own Items
    # (spec §3 section-mapping rule): only when the Item is absent or a pointer AND the MD&A was relocated or the
    # Item is a pointer (Controls also when absent: Wells Fargo 10-Qs have no Part I Item 4), so a normal filing
    # never loses MD&A text to a sub-heading.
    for key, heading, end, to_end, when_absent in (
            (RISK_KEY[form], RISK_FACTORS_HEADING, RISK_FACTORS_END, False, False),
            (CONTROLS_KEY[form], CONTROLS_HEADING, CONTROLS_END, True, True)):
        present = key in result
        is_pointer = (present and len(result[key].text) < POINTER_CHARS
                      and bool(POINTER_WORDING.search(result[key].text)))
        if mdna_key not in result or (present and not is_pointer):
            continue
        if not (mdna_key in relocated or is_pointer or (when_absent and not present)):
            continue
        carved = carve_region(result[mdna_key].text, heading, end, to_end=to_end)
        if carved:
            rest, region = carved
            location = result[mdna_key].location
            result[mdna_key] = Section(mdna_key, rest, location)
            result[key] = Section(key, region, location)
            relocated.append(key)

    trimmed_chars = 0                                    # cut off sections below, counted as unassigned
    if fs_key in relocated:                              # other 10-K sections after relocated statements (Citi)
        body = result[fs_key].text
        ends = [p for p in heading_offsets(body, STATEMENTS_END) if p > MIN_MDNA_CHARS]
        if ends:
            trimmed_chars += len(body) - ends[0]
            result[fs_key] = Section(fs_key, body[:ends[0]].rstrip(), result[fs_key].location)
    for key, section in result.items():
        body = strip_toc(section.text)
        trimmed_chars += len(section.text) - len(body)
        section = result[key] = Section(key, body, section.location)
        page = SIGNATURE_PAGE.search(section.text)
        if page:
            trimmed_chars += len(section.text) - page.start()
            result[key] = Section(key, section.text[:page.start()].rstrip(), section.location)

    kept = [s for s in result.values() if len(s.text) >= MIN_SECTION_CHARS]
    total_unassigned = len(unassigned) - used.get(MAIN_DOCUMENT, 0) + trimmed_chars
    if exhibit_text:
        total_unassigned += len(exhibit_text) - used.get(EXHIBIT, 0)
    return ParsedFiling(kept, headings_found, relocated, max(total_unassigned, 0))


# ----------------------------------------------------------------------------------------------- documents
def doc_id(ticker: str, form: str, fiscal_year: int, fiscal_period: str, key: str) -> str:
    """'AAPL_10-K_2024_Item7', 'AAPL_10-Q_2025Q2_PartIItem2' (week0_plan §2.1)."""
    period = "" if fiscal_period == "FY" else fiscal_period
    return f"{ticker}_{form}_{fiscal_year}{period}_{key.replace(' ', '')}"


def section_records(filing: dict, parsed: ParsedFiling) -> list[dict]:
    """sections.jsonl lines for one manifest entry (schema: week0_plan §2.1)."""
    out = []
    for s in sorted(parsed.sections, key=lambda s: section_order(filing["form"], s.key)):
        exhibit = s.location == EXHIBIT
        out.append({
            "doc_id": doc_id(filing["ticker"], filing["form"], filing["fiscal_year"], filing["fiscal_period"], s.key),
            "ticker": filing["ticker"], "cik": filing["cik"], "sector": filing["sector"], "form": filing["form"],
            "fiscal_year": filing["fiscal_year"], "fiscal_period": filing["fiscal_period"],
            "period_end": filing["period_end"], "filing_date": filing["filing_date"],
            "accession_no": filing["accession_no"], "item": s.key, "item_title": item_title(filing["form"], s.key),
            "source_url": filing["exhibit_url"] if exhibit else filing["source_url"],
            "source_location": s.location, "text": s.text, "n_chars": len(s.text), "has_tables": " | " in s.text,
        })
    return out


def section_order(form: str, key: str) -> int:
    keys = [f"Item {k}" for k in TENK_TITLES] if form == "10-K" else [f"Part {p} Item {i}" for p, i in TENQ_TITLES]
    return keys.index(key) if key in keys else len(keys)


# ----------------------------------------------------------------------------------------------- checks
DOC_COUNT_RANGE = (500, 2_000)                 # spec §3
MIN_CORE_ITEM_CHARS = 3_000                    # week0_plan WP2 step 11: Item 7 and Item 8 of every 10-K


def corpus_problems(docs: list[dict], expected_10k: int) -> list[str]:
    """The WP2 acceptance checks over the whole corpus."""
    out = []
    low, high = DOC_COUNT_RANGE
    if not low <= len(docs) <= high:
        out.append(f"{len(docs)} documents; the corpus must have between {low} and {high}")
    ids = [d["doc_id"] for d in docs]
    if len(set(ids)) != len(ids):
        out.append("duplicate doc_id values")
    tenk = {(d["ticker"], d["fiscal_year"]) for d in docs if d["form"] == "10-K"}
    if len(tenk) != expected_10k:
        out.append(f"{len(tenk)} 10-Ks have sections; expected {expected_10k}")
    toc_like = sorted(d["doc_id"] for d in docs if "text" in d and toc_blocks(d["text"].split("\n")))
    if toc_like:
        out.append(f"documents containing a table-of-contents block: {toc_like}")
    for key in ("Item 7", "Item 8"):
        long_enough = {(d["ticker"], d["fiscal_year"]) for d in docs
                       if d["form"] == "10-K" and d["item"] == key and d["n_chars"] > MIN_CORE_ITEM_CHARS}
        short = sorted(f"{t} FY{y}" for t, y in tenk - long_enough)
        if short:
            out.append(f"10-Ks without a {key} longer than {MIN_CORE_ITEM_CHARS:,} characters: {short}")
    return out
