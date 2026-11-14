"""scripts/compare_sections.py on small corpora: each check catches what it is for, independently of the rules."""
import pytest

import scripts.compare_sections as cs

NAMES = {"WFC": ("Wells Fargo & Company",), "MRK": ("Merck & Co., Inc.",)}
BODY = "Net interest income rose because of higher rates and loan growth.\n"


def doc(text, ticker="WFC", item="Item 7", year=2024):
    return {"doc_id": f"{ticker}_10-K_{year}_{item.replace(' ', '')}", "ticker": ticker, "form": "10-K",
            "fiscal_year": year, "fiscal_period": "FY", "item": item, "text": text, "n_chars": len(text)}


def corpus(*docs):
    return {d["doc_id"]: d for d in docs}


def run(old_text, new_text, items=(), ticker="WFC"):
    return cs.compare(corpus(doc(old_text, ticker)), corpus(doc(new_text, ticker)), NAMES, list(items))[0]


OLD = (BODY + "For example,\n\n31\n\nWells Fargo & Company | 31\n\nTable of Contents\nthe rate rose.\n"
       + BODY + "\n32\n\nWells Fargo & Company | 32\n")
NEW = BODY + "For example, the rate rose.\n" + BODY


def test_removing_only_furniture_passes_and_is_counted():
    problems, report = cs.compare(corpus(doc(OLD)), corpus(doc(NEW)), NAMES, [])
    assert problems == []
    assert any("page number 2" in line and "running footer 2" in line for line in report)


@pytest.mark.parametrize("new_text, problem", [
    (NEW.replace("loan growth.", "loan growth"), "text changed beyond the removed lines"),
    (NEW + "An extra line.\n", "text changed"),       # aligned with the removed footer block
    ("An extra line.\n" + NEW, "text inserted"),
    (NEW.replace("For example, the rate rose.\n", ""), "removed line that does not look like furniture"),
])
def test_content_changes_are_caught(new_text, problem):
    assert any(problem in p for p in run(OLD, new_text))


def test_a_lost_document_is_caught():
    old = corpus(doc(OLD), doc(BODY * 3, item="Item 1A"))
    assert any("1 documents lost" in p for p in cs.compare(old, corpus(doc(NEW)), NAMES, [])[0])


def test_a_value_taken_for_a_page_number_is_caught():
    old = BODY + "\n88\n\nTable of Contents\nNet income attributable | 3\n704\n\n" + BODY + "\n89\n"
    new = BODY + "Net income attributable | 3\n\n" + BODY
    problems = run(old, new, ticker="MRK")
    assert any("removed line next to a table row" in p for p in problems)
    assert any("page number out of the page sequence" in p and "704" in p for p in problems)


def test_a_sentence_joined_onto_a_repeated_title_is_caught():
    title = "Notes to Consolidated Financial Statements"
    old = "".join(f"Any liability\n{n}\n{title}\nresulting from it.\n" for n in (10, 11, 12))
    new = "".join(f"Any liability\n{title} resulting from it.\n" for _ in range(3))
    assert any("joined onto a repeated line" in p for p in run(old, new))


def test_a_lost_gold_figure_is_caught():
    item = {"id": "A99", "gold_answer": "Revenue was $82,296 million.",
            "gold_sources": [{"ticker": "WFC", "fiscal_year": 2024, "fiscal_period": "FY", "item": "Item 7"}]}
    old = BODY + "Total revenue | 82,296\n"
    problems = run(old, BODY, items=[item])
    assert any("gold figures no longer found" in p and "82,296" in p for p in problems)
    assert any("removed line that does not look like furniture" in p for p in problems)


def test_joined_pairs_maps_old_pieces_to_new_lines():
    assert cs.joined_pairs(["a b", "c", "d"], ["a b c", "d"]) == [("a b", "c")]


def test_a_removed_line_next_to_a_table_row_is_caught():
    # a statement row that names the company, taken for a footer (review): the row above stays
    old = BODY + "Total revenue | 8,597\nNet income attributable to Wells Fargo & Company | 822\n\n" + BODY
    new = BODY + "Total revenue | 8,597\n\n" + BODY
    assert any("removed line next to a table row" in p for p in run(old, new))


def test_a_single_page_number_is_checked_against_the_filing():
    # Item 9 has one page number only; the filing's pages around it are 40-41 and 43
    old = corpus(doc(BODY + "\n40\n\n" + BODY + "\n41\n\n" + BODY), doc(BODY + "\n704\n\n" + BODY, item="Item 9"),
                 doc(BODY + "\n43\n\n" + BODY, item="Item 10"))
    new = corpus(doc(BODY * 3), doc(BODY * 2, item="Item 9"), doc(BODY * 2, item="Item 10"))
    problems = cs.compare(old, new, NAMES, [])[0]
    assert any("page number out of the page sequence" in p and "704" in p for p in problems)
    old["WFC_10-K_2024_Item9"]["text"] = BODY + "\n42\n\n" + BODY
    assert cs.compare(old, new, NAMES, [])[0] == []


def test_the_checker_shares_no_rule_with_the_remover():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(cs.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "ingest.sections" not in imported


@pytest.mark.parametrize("line, kind", [
    ("Table of Contents", "table of contents"), ("Table of Contents | Wells Fargo", "table of contents"),
    ("31", "page number"), ("37.", "page number"), ("- 3 -", "page number"),
    ("Wells Fargo & Company | 27", "running footer"), ("27 | Wells Fargo & Company", "running footer"),
    ("Wells Fargo & Company 2024 Form 10-K | 27", "running footer"), ("2024 Annual Report | 27", "running footer"),
    ("Wells Fargo & Company and Subsidiaries", "running header"),
    ("Wells Fargo & Company", None), ("Wells Fargo & Company | 27 | 28", None), ("Wells Fargo & Company | 1,234", None),
    ("Wells Fargo & Company and Subsidiaries 27", "running footer"), ("Total | 27", None), ("2024", None),
    ("(12)", None), ("1,234", None), ("Subsidiaries of the registrant", None), ("Managing Global Risk", None),
])
def test_the_checker_has_its_own_idea_of_furniture(line, kind):
    assert cs.Looks(NAMES["WFC"]).kind(line) == kind
