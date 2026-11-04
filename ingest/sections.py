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
MIN_TOC_ROWS = 3                 # a block under a heading with this many page-reference rows is a TOC
TOC_LOOKAHEAD_LINES = 6          # how many lines after a region heading are checked for TOC rows
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
                              r"|financial\s+statements|glossary|managing\s+global\s+risk|sustainability"
                              r"|human\s+capital)", re.IGNORECASE)
# A relocated financial-statements region (Citi) is followed by other 10-K sections and the exhibit index;
# those end it (whole heading lines only, so phrases inside the notes do not)
STATEMENTS_END = re.compile(r"^(?:properties|supervision,?\s+(?:and\s+)?regulation\b.*|unregistered\s+sales\b.*"
                            r"|exhibit\s+index|exhibits?|legal\s+proceedings)$", re.IGNORECASE)
RISK_KEY = {"10-K": "Item 1A", "10-Q": "Part II Item 1A"}
# The signature page (heading + its standard legend) ends a filing's text; without this the last Item, or a
# relocated Item 8, would carry the signatures and anything printed after them
SIGNATURE_PAGE = re.compile(r"^[ \t]*signatures?[ \t]*\n+[ \t]*pursuant\s+to\s+the\s+requirements",
                            re.IGNORECASE | re.MULTILINE)
TABLE_FILLER_CELLS = frozenset({"$", "%", ")"})
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
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ----------------------------------------------------------------------------------------------- Item headings
def item_title(form: str, key: str) -> str:
    if form == "10-K":
        return TENK_TITLES.get(key.removeprefix("Item "), "")
    found = re.match(r"Part (\w+) Item (\w+)$", key)
    return TENQ_TITLES.get((found.group(1), found.group(2)), "") if found else ""


def find_headings(lines: list[str], form: str) -> list[tuple[int, str]]:
    """(line index, Item key) for every line that looks like an Item heading."""
    out, part = [], "I"
    for i, line in enumerate(lines):
        if TOC_PAGE_NUMBER.search(line):
            continue
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
            nxt = next((ln.strip() for ln in lines[i + 1:i + 4] if ln.strip()), "")
            titles = [TENK_TITLES.get(item, "")] if form == "10-K" else [
                t for (_, it), t in TENQ_TITLES.items() if it == item]
            if not any(t and starts_with_title(nxt, t) for t in titles):
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


def heading_offsets(text: str, pattern: re.Pattern) -> list[int]:
    """Character offsets of short lines matching `pattern` (region headings such as "Management's Discussion").
    Lines inside a table of contents (the row itself, or MIN_TOC_ROWS page-reference rows among the next
    lines) are not headings."""
    lines, pos, out = text.split("\n"), 0, []
    for n, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < REGION_HEADING_MAX_CHARS and pattern.match(stripped) and not is_toc_row(line):
            following = [ln for ln in lines[n + 1:n + 1 + TOC_LOOKAHEAD_LINES] if ln.strip()]
            if sum(map(is_toc_row, following)) < MIN_TOC_ROWS:
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
    statements follow the exhibit list in Item 15: cut Item 15 at the first MD&A / statements heading and
    return the rest as unassigned text for relocation. Item 15 is left alone when nothing follows it."""
    body = sections.get("Item 15", "")
    starts = [p for p in heading_offsets(body, MDNA_HEADING) + heading_offsets(body, FS_HEADING)
              if p > MIN_SECTION_CHARS]
    if not starts:
        return ""
    cut = min(starts)
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


def is_toc_row(line: str) -> bool:
    return bool(TOC_PAGE_NUMBER.search(line))


def drop_leading_toc(text: str) -> str:
    """Remove a table of contents printed right under a section's heading (JPM and Wells Fargo 10-Qs: the TOC
    entry is the only 'Item 2' heading, so the section starts with the rest of the TOC). The heading line is
    kept; the block is removed only if it has at least MIN_TOC_ROWS rows ending in a page reference."""
    lines = text.split("\n")
    end, rows = 1, 0
    while end < len(lines) and (not lines[end].strip() or is_toc_row(lines[end]) or PART_LINE.match(lines[end])
                                or TOC_SUBHEADING.match(lines[end].strip())):
        rows += is_toc_row(lines[end])
        end += 1
    if rows < MIN_TOC_ROWS:
        return text
    return "\n".join(lines[:1] + lines[end:]).strip()


def carve_region(text: str, start_heading: re.Pattern, end_heading: re.Pattern) -> tuple[str, str] | None:
    """(text without the region, region) for the region from the first `start_heading` line to the next
    `end_heading` line, or None if there is no such region of at least MIN_SECTION_CHARS."""
    starts = heading_offsets(text, start_heading)
    if not starts:
        return None
    start = starts[0]
    ends = [p for p in heading_offsets(text, end_heading) if p > start + MIN_SECTION_CHARS]
    end = ends[0] if ends else None
    if end is None:
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

    risk_key = RISK_KEY[form]
    risk_is_pointer = risk_key in result and is_stub(risk_key)
    if mdna_key in result and is_stub(risk_key) and (mdna_key in relocated or risk_is_pointer):
        # risk factors printed inside the MD&A / annual report (spec §3 section-mapping rule)
        carved = carve_region(result[mdna_key].text, RISK_FACTORS_HEADING, RISK_FACTORS_END)
        if carved:
            rest, risk_text = carved
            location = result[mdna_key].location
            result[mdna_key] = Section(mdna_key, rest, location)
            result[risk_key] = Section(risk_key, risk_text, location)
            relocated.append(risk_key)

    trimmed_chars = 0                                    # cut off sections below, counted as unassigned
    if fs_key in relocated:                              # other 10-K sections after relocated statements (Citi)
        body = result[fs_key].text
        ends = [p for p in heading_offsets(body, STATEMENTS_END) if p > MIN_MDNA_CHARS]
        if ends:
            trimmed_chars += len(body) - ends[0]
            result[fs_key] = Section(fs_key, body[:ends[0]].rstrip(), result[fs_key].location)
    for key, section in result.items():
        body = drop_leading_toc(section.text)
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
    toc_like = sorted(d["doc_id"] for d in docs if "text" in d
                      and sum(1 for line in d["text"].split("\n") if ITEM_LINE.match(line) and is_toc_row(line))
                      >= MIN_TOC_ROWS)
    if toc_like:
        out.append(f"documents containing table-of-contents rows: {toc_like}")
    for key in ("Item 7", "Item 8"):
        long_enough = {(d["ticker"], d["fiscal_year"]) for d in docs
                       if d["form"] == "10-K" and d["item"] == key and d["n_chars"] > MIN_CORE_ITEM_CHARS}
        short = sorted(f"{t} FY{y}" for t, y in tenk - long_enough)
        if short:
            out.append(f"10-Ks without a {key} longer than {MIN_CORE_ITEM_CHARS:,} characters: {short}")
    return out
