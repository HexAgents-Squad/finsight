"""ingest/sections.py page furniture (W1-04): what is removed, what stays, and sentences cut by a page break.

Each line below is a shape measured in the corpus before W1-04 (eval/results/furniture_removal.md)."""
import pytest

from ingest.edgar import company_names
from ingest.sections import (
    MIN_FURNITURE_REPEATS,
    MIN_SECTION_CHARS,
    TENK_TITLES,
    Furniture,
    corpus_problems,
    find_furniture,
    html_to_text,
    parse_filing,
    strip_furniture,
)
from tests.test_sections import STANDARD, body, tenk_html

NAMES = company_names()


def furniture_of(ticker: str, *lines: str, repeats: int = MIN_FURNITURE_REPEATS) -> Furniture:
    """The furniture of a filing in which each line appears `repeats` times (with changing page numbers)."""
    text = "\n".join(line.replace("{n}", str(10 + i)) for i in range(repeats) for line in lines)
    return find_furniture([text], NAMES[ticker])


@pytest.mark.parametrize("ticker, line", [
    ("WFC", "Wells Fargo & Company | {n}"), ("WFC", "{n} | Wells Fargo & Company"),
    ("BAC", "Bank of America {n}"), ("BAC", "{n} Bank of America"),
    ("JPM", "{n} | JPMorgan Chase & Co./2024 Form 10-K"), ("JPM", "JPMorgan Chase & Co./2024 Form 10-K | {n}"),
    ("GS", "Goldman Sachs 2024 Form 10-K | {n}"), ("GS", "{n} | Goldman Sachs September 2025 Form 10-Q"),
    ("PFE", "Pfizer Inc. | 2024 Form 10-K | {n}"), ("AAPL", "Apple Inc. | 2024 Form 10-K | {n}"),
    ("ABBV", "{n} | | 2024 Form 10-K"), ("ABBV", "2025 Form 10-Q | | {n}"),
    ("JNJ", "2024 Annual Report | {n}"), ("JNJ", "Form 10-Q | {n}"),
    ("PG", "{n} The Procter & Gamble Company"),
    ("GS", "THE GOLDMAN SACHS GROUP, INC. AND SUBSIDIARIES"), ("PFE", "Pfizer Inc. and Subsidiary Companies"),
    ("NVDA", "NVIDIA Corporation and Subsidiaries"), ("BAC", "Bank of America Corporation and Subsidiaries"),
])
def test_running_footers_and_headers_are_furniture(ticker, line):
    furniture = furniture_of(ticker, line)
    assert line.replace("{n}", "27") in furniture


@pytest.mark.parametrize("line", ["Table of Contents", "  table of contents ", "Table of Contents | Alphabet Inc.",
                                  "31", "7", "131", "37."])
def test_contents_links_and_page_numbers_are_furniture_in_any_filing(line):
    assert line in Furniture()


@pytest.mark.parametrize("ticker, line", [
    ("BAC", "Bank of America Corporation"),          # an issuer label above its senior notes (measured)
    ("JNJ", "Johnson & Johnson"),                    # a chart legend (measured)
    ("JPM", "Chase | {n}"),                          # a one-word alias: a table row
    ("GS", "Total | {n}"), ("PG", "NOTE {n}"), ("BAC", "December {n}"), ("COST", "{n} Weeks Ended"),
    ("MSFT", "Item {n}"), ("GS", "Note {n}."),
    ("WFC", "Wells Fargo & Company | {n} | {n}"),    # two numbers: a data row, not a page
    ("WFC", "Wells Fargo & Company reported net income of {n} billion"),
])
def test_content_that_looks_like_furniture_stays(ticker, line):
    furniture = furniture_of(ticker, line)
    assert line.replace("{n}", "27") not in furniture


@pytest.mark.parametrize("line", ["1,234", "2024", "1234", "(12)", "12%", "Table of Contents and Index of Notes",
                                  "Part II", "$12"])
def test_numbers_and_headings_are_not_page_numbers(line):
    assert line not in Furniture()


def test_a_footer_must_repeat_in_the_filing():
    assert "Wells Fargo & Company | 27" not in furniture_of("WFC", "Wells Fargo & Company | {n}",
                                                            repeats=MIN_FURNITURE_REPEATS - 1)


def test_a_sentence_cut_by_a_page_break_is_joined():
    text = ("Item 1A. Risk Factors\nOur products may not be successful or may be claimed to be non-compliant. "
            "For example,\n\n31\n\nTable of Contents\nprotecting our intellectual property is costly.")
    out, removed = strip_furniture(text, Furniture())
    assert removed == 2
    assert out == ("Item 1A. Risk Factors\nOur products may not be successful or may be claimed to be "
                   "non-compliant. For example, protecting our intellectual property is costly.")


@pytest.mark.parametrize("before, after, joined", [
    ("The year ended.", "next paragraph starts here", False),     # the sentence had ended
    ("Revenue | 1,234 | 5,678", "continued table text", False),    # a table row
    ("For example,", "Risk Factors", False),                       # a capitalised heading follows
    ("For example,", "the next words", True),
])
def test_a_join_only_mends_a_cut_sentence(before, after, joined):
    out, _ = strip_furniture(f"{before}\n12\nTable of Contents\n{after}", Furniture())
    assert (out == f"{before} {after}") is joined
    assert before in out and after in out


def test_text_without_furniture_is_returned_unchanged():
    text = "Item 7. MD&A\n\nRevenue grew 12% to $391,035 million.\n\n| a | b |"
    assert strip_furniture(text, Furniture()) == (text, 0)


def test_parse_filing_removes_furniture_with_the_companys_names():
    footer = "<p>Wells Fargo & Company | {}</p><p>Table of Contents</p><p>{}</p>"
    pages = "".join(footer.format(n, n) for n in range(40, 45))
    items = {**STANDARD, "7": body("mdna", 25_000) + "</p>" + pages + "<p>" + body("more")}
    parsed = parse_filing(html_to_text(tenk_html(items)), "10-K", names=NAMES["WFC"])
    mdna = next(s for s in parsed.sections if s.key == "Item 7").text
    assert "Wells Fargo & Company |" not in mdna and "Table of Contents" not in mdna
    assert parsed.furniture_lines == 15
    without_names = parse_filing(html_to_text(tenk_html(items)), "10-K")
    assert "Wells Fargo & Company | 40" in next(s for s in without_names.sections if s.key == "Item 7").text


def test_furniture_never_changes_which_items_exist():
    # a pointer Item just over the minimum only because of its footer stays, as before W1-04
    heading = len(f"Item 12. {TENK_TITLES['12']}") + 1                 # its line break
    pointer = "x" * (MIN_SECTION_CHARS - heading - 8) + "</p><p>Table of Contents</p><p>73"
    parsed = parse_filing(html_to_text(tenk_html({**STANDARD, "12": pointer})), "10-K")
    item = next(s for s in parsed.sections if s.key == "Item 12")
    assert len(item.text) < MIN_SECTION_CHARS and "Table of Contents" not in item.text


def test_the_corpus_check_flags_furniture_left_in_a_document():
    def doc(ticker, item, text):
        return {"doc_id": f"{ticker}_10-K_2024_{item.replace(' ', '')}", "ticker": ticker, "form": "10-K",
                "fiscal_year": 2024, "item": item, "n_chars": 5_000, "text": text}
    docs = [doc(f"T{i}", item, "clean text") for i in range(300) for item in ("Item 7", "Item 8")]
    assert corpus_problems(docs, expected_10k=300) == []
    docs[0]["text"] = "clean text\n42\nTable of Contents\nmore"
    assert any("still holding 'Table of Contents' lines or page numbers" in p for p in corpus_problems(docs, 300))


def test_company_names_keep_full_names_and_multiword_aliases_only():
    assert NAMES["JPM"][0] == "JPMorgan Chase & Co." and "Chase" not in NAMES["JPM"]
    assert "Bank of America" in NAMES["BAC"] and "BofA" not in NAMES["BAC"]
    assert set(NAMES) >= {"AAPL", "WFC", "GS"} and all(NAMES.values())
