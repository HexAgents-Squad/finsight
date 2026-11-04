"""ingest/sections.py: HTML cleaning, Item splitting, the section-mapping rule and corpus checks (offline).

Each test filing is a few lines of synthetic HTML reproducing one pitfall from week0_plan WP2 step 9.
"""
import pytest

from eval.schema import TENK_ITEMS, TENQ_ITEMS
from ingest.sections import (
    EXHIBIT,
    MAIN_DOCUMENT,
    MIN_SECTION_CHARS,
    TENK_TITLES,
    TENQ_TITLES,
    corpus_problems,
    doc_id,
    html_to_text,
    parse_filing,
    section_records,
    split_items,
)

LONG = 400          # paragraphs longer than MIN_SECTION_CHARS
BIG = 25_000        # longer than MIN_MDNA_CHARS / STUB_CHARS


def body(word: str, n: int = LONG) -> str:
    return (word + " ") * (n // (len(word) + 1) + 1)


def tenk_html(items: dict[str, str], toc: bool = True, extra: str = "") -> str:
    """A 10-K: optional table of contents (rows ending in page numbers), then each Item heading + text."""
    parts = ["<html><body><p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION - Form 10-K</p>"]
    if toc:
        rows = "".join(f"<tr><td>Item {k}.</td><td>{TENK_TITLES[k]}</td><td>{n + 3}</td></tr>"
                       for n, k in enumerate(items))
        parts.append(f"<table>{rows}</table>")
    for k, text in items.items():
        parts.append(f"<p>Item {k}. {TENK_TITLES[k]}</p><p>{text}</p>")
    parts.append(extra + "</body></html>")
    return "".join(parts)


STANDARD = {"1": body("business"), "1A": body("risk"), "2": body("property"), "7": body("mdna", BIG),
            "7A": body("market"), "8": body("statements", BIG), "9A": body("controls"), "15": body("exhibits")}


# --- contract -------------------------------------------------------------------------------------------

def test_item_names_match_the_gold_set_contract():
    assert {f"Item {k}" for k in TENK_TITLES} == TENK_ITEMS
    assert {f"Part {p} Item {i}" for p, i in TENQ_TITLES} == TENQ_ITEMS


@pytest.mark.parametrize("args, expected", [
    (("AAPL", "10-K", 2024, "FY", "Item 7"), "AAPL_10-K_2024_Item7"),
    (("AAPL", "10-Q", 2025, "Q2", "Part I Item 2"), "AAPL_10-Q_2025Q2_PartIItem2"),
])
def test_doc_id_format(args, expected):
    assert doc_id(*args) == expected


# --- HTML -> text ---------------------------------------------------------------------------------------

def test_html_to_text_drops_hidden_xbrl_and_flattens_tables():
    html = """<html><head><style>p{}</style></head><body>
      <div style="display: none"><ix:header>dei:EntityName secret</ix:header></div>
      <script>var x=1;</script>
      <p>Net sales grew.</p>
      <table><tr><td>Net sales</td><td>$</td><td>391,035</td><td>$</td><td>383,285</td></tr>
             <tr><td></td><td></td></tr></table>
      <p>Done&#160;here.</p></body></html>"""
    text = html_to_text(html)
    assert "secret" not in text and "var x" not in text and "p{}" not in text
    assert "Net sales | 391,035 | 383,285" in text and "Done here." in text


# --- splitting ------------------------------------------------------------------------------------------

def test_table_of_contents_is_ignored_and_the_body_wins():
    sections, _ = split_items(html_to_text(tenk_html(STANDARD)), "10-K")
    assert set(sections) == {f"Item {k}" for k in STANDARD}
    assert sections["Item 7"].startswith("Item 7. Management") and "mdna mdna" in sections["Item 7"]
    assert "risk" not in sections["Item 7"]


def test_running_page_header_is_not_a_heading():
    item7 = body("mdna", 2_000) + "<p>Item 7</p><p>continued discussion of results</p>" + body("more", 2_000)
    text = html_to_text(tenk_html({**STANDARD, "7": item7}, toc=False))
    sections, _ = split_items(text, "10-K")
    assert "continued discussion of results" in sections["Item 7"]      # no new section started there


def test_cross_reference_inside_text_does_not_split():
    item1 = body("business") + "<p>Item 7 of this report discusses results.</p>" + body("business")
    sections, _ = split_items(html_to_text(tenk_html({**STANDARD, "1": item1}, toc=False)), "10-K")
    assert "Item 7 of this report" in sections["Item 1"]


def test_tenq_part_comes_from_the_title():
    html = ("<p>PART I. FINANCIAL INFORMATION</p>"
            f"<p>Item 1. Financial Statements</p><p>{body('fs')}</p>"
            f"<p>Item 2. Management's Discussion and Analysis</p><p>{body('mdna')}</p>"
            f"<p>Item 4. Controls and Procedures</p><p>{body('controls')}</p>"
            # no "PART II" line before these, as in some filings: the titles still say Part II
            f"<p>Item 1. Legal Proceedings</p><p>{body('legal')}</p>"
            f"<p>Item 1A. Risk Factors</p><p>{body('risk')}</p>")
    sections, _ = split_items(html_to_text(html), "10-Q")
    assert set(sections) == {"Part I Item 1", "Part I Item 2", "Part I Item 4", "Part II Item 1", "Part II Item 1A"}
    assert "legal" in sections["Part II Item 1"] and "fs" in sections["Part I Item 1"]


def test_short_items_are_dropped():
    parsed = parse_filing(html_to_text(tenk_html({**STANDARD, "4": "Not applicable."})), "10-K")
    keys = {s.key for s in parsed.sections}
    assert "Item 4" not in keys and "Item 7" in keys
    assert all(len(s.text) >= MIN_SECTION_CHARS for s in parsed.sections)


# --- section-mapping rule -------------------------------------------------------------------------------

def test_bank_10k_mdna_and_statements_appended_after_item_15_are_relocated():
    stubs = {**STANDARD, "7": "See Management's Discussion and Analysis on page 50.",
             "8": "See the Consolidated Financial Statements on page 150."}
    report = ("<p>Management's discussion and analysis</p><p>" + body("bankmdna", BIG) + "</p>"
              "<p>Consolidated statements of income</p><p>" + body("bankfs", BIG) + "</p>")
    parsed = parse_filing(html_to_text(tenk_html(stubs, extra=report)), "10-K")
    by_key = {s.key: s for s in parsed.sections}
    assert set(parsed.relocated) == {"Item 7", "Item 8"}
    assert "bankmdna" in by_key["Item 7"].text and "bankfs" not in by_key["Item 7"].text
    assert by_key["Item 8"].text.startswith("Consolidated statements of income")
    assert by_key["Item 7"].location == MAIN_DOCUMENT
    assert "exhibits" in by_key["Item 15"].text and "bankmdna" not in by_key["Item 15"].text


def test_annual_report_exhibit_supplies_mdna_and_statements():
    stubs = {**STANDARD, "7": "Incorporated by reference to the Annual Report.", "8": "Incorporated by reference."}
    exhibit = html_to_text("<p>Overview</p><p>" + body("exmdna", BIG) + "</p><p>Report of Independent Registered "
                           "Public Accounting Firm</p><p>" + body("exfs", BIG) + "</p>")
    parsed = parse_filing(html_to_text(tenk_html(stubs)), "10-K", exhibit)
    by_key = {s.key: s for s in parsed.sections}
    assert by_key["Item 7"].location == EXHIBIT and by_key["Item 8"].location == EXHIBIT
    assert "exmdna" in by_key["Item 7"].text and "exfs" in by_key["Item 8"].text


def test_statements_after_item_15_fill_a_pointer_item_8():
    items = {**STANDARD, "8": "The information required by this Item is set forth in Item 15."}
    statements = "<p>Consolidated Statements of Income</p><p>" + body("nvfs", BIG) + "</p>"
    parsed = parse_filing(html_to_text(tenk_html(items, extra=statements)), "10-K")
    by_key = {s.key: s for s in parsed.sections}
    assert parsed.relocated == ["Item 8"] and "nvfs" in by_key["Item 8"].text
    assert "nvfs" not in by_key["Item 15"].text


def test_bank_10q_statements_inside_the_mdna_item_are_split_out():
    html = (f"<p>Item 2. Management's Discussion and Analysis</p><p>{body('qmdna', BIG)}</p>"
            f"<p>Consolidated Statements of Income</p><p>{body('qfs', 5_000)}</p>"
            f"<p>Item 1. Financial Statements</p><p>See page 70.</p>"
            f"<p>Item 4. Controls and Procedures</p><p>{body('controls')}</p>"
            f"<p>Item 1A. Risk Factors</p><p>{body('risk')}</p>"
            f"<p>Item 6. Exhibits</p><p>{body('exhibits')}</p>")
    parsed = parse_filing(html_to_text(html), "10-Q")
    by_key = {s.key: s for s in parsed.sections}
    assert "Part I Item 1" in parsed.relocated
    assert "qfs" in by_key["Part I Item 1"].text and "qfs" not in by_key["Part I Item 2"].text


def test_filing_without_item_headings_keeps_only_mapped_regions():
    html = ("<p>Annual report</p><p>" + body("letter", 5_000) + "</p><p>Management's Discussion and Analysis</p><p>"
            + body("citimdna", BIG) + "</p><p>Consolidated Statement of Income</p><p>" + body("citifs", BIG) + "</p>")
    parsed = parse_filing(html_to_text(html), "10-K")
    assert {s.key for s in parsed.sections} == {"Item 7", "Item 8"}
    assert parsed.unassigned_chars >= 5_000                    # the letter is dropped, and counted


# --- records and checks ---------------------------------------------------------------------------------

FILING = {"ticker": "JPM", "cik": "0000019617", "sector": "financials", "form": "10-K", "fiscal_year": 2024,
          "fiscal_period": "FY", "period_end": "2024-12-31", "filing_date": "2025-02-14",
          "accession_no": "0000019617-25-000270", "source_url": "https://www.sec.gov/main.htm",
          "exhibit_url": "https://www.sec.gov/ex13.htm"}
SCHEMA_KEYS = {"doc_id", "ticker", "cik", "sector", "form", "fiscal_year", "fiscal_period", "period_end",
               "filing_date", "accession_no", "item", "item_title", "source_url", "source_location", "text",
               "n_chars", "has_tables"}


def test_records_follow_the_section_schema_and_cite_the_right_file():
    stubs = {**STANDARD, "7": "See the Annual Report.", "8": "See the Annual Report."}
    exhibit = html_to_text("<p>Overview</p><p>" + body("exmdna", BIG) + "</p><p>Consolidated Statements of Income"
                           "</p><table><tr><td>Revenue</td><td>100</td></tr></table><p>" + body("exfs", BIG) + "</p>")
    records = section_records(FILING, parse_filing(html_to_text(tenk_html(stubs)), "10-K", exhibit))
    assert all(set(r) == SCHEMA_KEYS for r in records)
    item7 = next(r for r in records if r["item"] == "Item 7")
    item1 = next(r for r in records if r["item"] == "Item 1")
    assert item7["doc_id"] == "JPM_10-K_2024_Item7" and item7["source_location"] == EXHIBIT
    assert item7["source_url"] == FILING["exhibit_url"] and item1["source_url"] == FILING["source_url"]
    assert item7["item_title"].startswith("Management's Discussion")
    assert [r["item"] for r in records] == sorted((r["item"] for r in records),
                                                  key=lambda k: list(TENK_TITLES).index(k.removeprefix("Item ")))
    item8 = next(r for r in records if r["item"] == "Item 8")
    assert item8["has_tables"] and item8["n_chars"] == len(item8["text"])


def doc(ticker, year, item, n=5_000, form="10-K"):
    return {"doc_id": f"{ticker}_{form}_{year}_{item}", "ticker": ticker, "fiscal_year": year, "form": form,
            "item": item, "n_chars": n}


def test_corpus_checks():
    docs = [doc(f"T{i}", 2024, item) for i in range(300) for item in ("Item 7", "Item 8")]
    assert corpus_problems(docs, expected_10k=300) == []
    short = docs[:-1] + [doc("T299", 2024, "Item 8", n=100)]
    assert any("Item 8" in p and "T299 FY2024" in p for p in corpus_problems(short, 300))
    assert any("between 500 and 2000" in p for p in corpus_problems(docs[:10], 5))
    assert any("expected 301" in p for p in corpus_problems(docs, 301))
    assert "duplicate doc_id values" in corpus_problems(docs + docs[:1], 300)


# --- risk factors inside a bank's MD&A (section-mapping rule) ---------------------------------------------

def bank_report(risk_end_heading: str = "Controls and Procedures") -> str:
    return ("<p>Management's Discussion and Analysis</p><p>" + body("overview", BIG) + "</p>"
            "<p>Risk Factors</p><p>" + body("riskfactor", 5_000) + "</p>"
            f"<p>{risk_end_heading}</p><p>" + body("tail", 2_000) + "</p>"
            "<p>Consolidated Statements of Income</p><p>" + body("fs", BIG) + "</p>")


def test_risk_factors_inside_a_relocated_mdna_become_item_1a():
    stubs = {**STANDARD, "1A": "See Risk Factors in the Annual Report.", "7": "See the Annual Report.",
             "8": "See the Annual Report."}
    parsed = parse_filing(html_to_text(tenk_html(stubs)), "10-K", html_to_text(bank_report()))
    by_key = {s.key: s for s in parsed.sections}
    assert "Item 1A" in parsed.relocated and by_key["Item 1A"].location == EXHIBIT
    assert "riskfactor" in by_key["Item 1A"].text and "riskfactor" not in by_key["Item 7"].text
    assert "overview" in by_key["Item 7"].text and "tail" in by_key["Item 7"].text   # MD&A keeps both sides


def test_risk_factors_without_a_clear_end_stay_in_place():
    stubs = {**STANDARD, "1A": "See the Annual Report.", "7": "See the Annual Report.", "8": "See the Annual Report."}
    parsed = parse_filing(html_to_text(tenk_html(stubs)), "10-K", html_to_text(bank_report("Our Strategy")))
    by_key = {s.key: s for s in parsed.sections}
    assert "Item 1A" not in by_key and "riskfactor" in by_key["Item 7"].text


def test_a_normal_filing_keeps_its_mdna_whole():
    item7 = body("mdna", BIG) + "<p>Risk Factors</p><p>" + body("discussion") + "</p><p>Controls and Procedures</p>"
    no_1a = {k: v for k, v in STANDARD.items() if k != "1A"}
    parsed = parse_filing(html_to_text(tenk_html({**no_1a, "7": item7})), "10-K")
    assert "Item 1A" not in {s.key for s in parsed.sections} and parsed.relocated == []


def test_signature_page_is_cut_from_the_last_item():
    item16 = (body("summary") + "<p>SIGNATURES</p><p>Pursuant to the requirements of Section 13 or 15(d) of the "
              "Securities Exchange Act of 1934, the registrant has duly caused this report to be signed.</p>"
              "<p>/s/ Jane Doe, Director</p>")
    parsed = parse_filing(html_to_text(tenk_html({**STANDARD, "16": item16})), "10-K")
    item = next(s for s in parsed.sections if s.key == "Item 16")
    assert "summary" in item.text and "Jane Doe" not in item.text and "SIGNATURES" not in item.text
    assert parsed.unassigned_chars > 100


def test_the_word_signatures_alone_does_not_cut():
    item1 = body("business") + "<p>Signatures</p><p>We collect electronic signatures from customers.</p>"
    parsed = parse_filing(html_to_text(tenk_html({**STANDARD, "1": item1})), "10-K")
    assert "electronic signatures" in next(s.text for s in parsed.sections if s.key == "Item 1")


# --- regressions from the code review on real filings ----------------------------------------------------

def test_running_header_whose_next_line_only_shares_a_word_is_not_a_heading():
    # MSFT FY2023: a bare "Item 8" page header followed by the note "Financial Instruments" split Item 8
    item8 = (body("statements", 3_000) + "<p>Item 8</p><p>Financial Instruments</p>" + body("notes", 3_000))
    sections, _ = split_items(html_to_text(tenk_html({**STANDARD, "8": item8}, toc=False)), "10-K")
    assert "statements" in sections["Item 8"] and "notes" in sections["Item 8"]
    assert "statements" not in sections["Item 7A"]


@pytest.mark.parametrize("page", ["N/A", "None", "F-1", "12-14", "23"])
def test_toc_rows_with_any_page_reference_are_ignored(page):
    # PFE: "Item 4. | Mine Safety Disclosures | N/A" rows won when the real Item was short
    toc = f"<table><tr><td>Item 4.</td><td>Mine Safety Disclosures</td><td>{page}</td></tr></table>"
    html = tenk_html({**STANDARD, "4": "Not applicable."}).replace("<body>", "<body>" + toc, 1)
    assert "Item 4" not in {s.key for s in parse_filing(html_to_text(html), "10-K").sections}


def test_item_15_of_a_normal_filer_is_not_cut():
    # KO: its exhibit list names "Consolidated Statements of Income"; nothing is relocated, so keep all of it
    item15 = ("<p>(a) Financial statements:</p><p>Consolidated Statements of Income</p><p>" + body("exhibit", BIG)
              + "</p><p>Exhibit 99.9 last entry of the list</p>")
    parsed = parse_filing(html_to_text(tenk_html({**STANDARD, "15": item15})), "10-K")
    item = next(s for s in parsed.sections if s.key == "Item 15")
    assert "last entry of the list" in item.text and parsed.relocated == []


def test_relocated_statements_end_before_trailing_10k_sections():
    # Citi: Properties, regulation and the exhibit index follow the statements in the same report
    html = ("<p>Management's Discussion and Analysis</p><p>" + body("mdna", BIG) + "</p><p>Consolidated Statement of "
            "Income</p><p>" + body("fs", BIG) + "</p><p>SUPERVISION, REGULATION AND OTHER</p><p>" + body("reg", 3_000)
            + "</p><p>Exhibit Index</p><p>" + body("exhibitlist") + "</p>")
    parsed = parse_filing(html_to_text(html), "10-K")
    item8 = next(s for s in parsed.sections if s.key == "Item 8")
    assert "fs fs" in item8.text and "reg" not in item8.text and "exhibitlist" not in item8.text


def test_risk_factors_subheadings_do_not_end_the_section():
    # Citi: "Climate Change Presents Various ... Risks" is a risk factor, not the end of the risk factors
    report = ("<p>Management's Discussion and Analysis</p><p>" + body("overview", BIG) + "</p><p>RISK FACTORS</p><p>"
              + body("riska", 2_000) + "</p><p>Climate Change Presents Various Risks</p><p>" + body("riskb", 2_000)
              + "</p><p>SUSTAINABILITY</p><p>" + body("tail", 2_000) + "</p><p>Consolidated Statement of Income</p>"
              "<p>" + body("fs", BIG) + "</p>")
    parsed = parse_filing(html_to_text(report), "10-K")
    item1a = next(s for s in parsed.sections if s.key == "Item 1A")
    assert "riska" in item1a.text and "riskb" in item1a.text and "tail" not in item1a.text


def test_toc_under_a_heading_is_removed_and_toc_headings_are_skipped():
    # JPM/WFC 10-Q: the TOC entry is the only "Item 2" heading; JPM 10-K: the annual report's own contents list
    from ingest.sections import drop_leading_toc
    text = ("Item 2. Management's Discussion and Analysis\nExecutive Overview | 3\nResults | 9\n"
            "Item 3. Market Risk | 201\nPart II - Other information\nItem 1. Legal Proceedings | 202\n\nBody text.")
    assert drop_leading_toc(text) == "Item 2. Management's Discussion and Analysis\nBody text."
    html = ("<p>Management's discussion and analysis:</p><p>Overview | 1</p><p>Results | 5</p><p>Liquidity | 9</p>"
            "<p>Management's discussion and analysis</p><p>" + body("realmdna", BIG) + "</p>"
            "<p>Consolidated statements of income</p><p>" + body("fs", BIG) + "</p>")
    item7 = next(s for s in parse_filing(html_to_text(html), "10-K").sections if s.key == "Item 7")
    assert "Overview | 1" not in item7.text and "realmdna" in item7.text


def test_corpus_check_flags_documents_holding_toc_rows():
    docs = [doc(f"T{i}", 2024, item) for i in range(300) for item in ("Item 7", "Item 8")]
    docs[0] = {**docs[0], "text": "Item 1. Business | 3\nItem 1A. Risk Factors | 9\nItem 7. MD&A | 30\nreal text"}
    assert any("table-of-contents rows" in p and docs[0]["doc_id"] in p for p in corpus_problems(docs, 300))
