"""retrieval/statement.py (W1-05): the financial statements first for a question about a reported figure."""
import pytest

from retrieval.common import NO_FILTERS, Filters, SearchResult
from retrieval.statement import StatementFirstIndex, asks_for_figure, statement_item


def hit(cid):
    return SearchResult(cid, "t", "AAPL", "10-K", 2024, "FY", "Item 8", "u", 1.0)


class ByItem:
    """Answers with two chunks per search, named after the Item the filter asks for ('any' when none)."""
    def __init__(self):
        self.calls = []

    def search(self, query, k=5, filters=NO_FILTERS):
        self.calls.append((k, filters))
        return [hit(f"{filters.item or 'any'}-{n}") for n in range(2)]


FY = Filters.of(ticker="AAPL", fiscal_year=2024, fiscal_period="FY")


def test_a_figure_question_gets_the_statement_items_chunks_first_interleaved():
    base = ByItem()
    hits = StatementFirstIndex(base).search("What was Apple's net income in fiscal 2024?", k=4, filters=FY)
    assert [h.chunk_id for h in hits] == ["Item 8-0", "any-0", "Item 8-1", "any-1"]
    assert base.calls == [(4, FY), (4, Filters.of(ticker="AAPL", fiscal_year=2024, fiscal_period="FY",
                                                  item="Item 8"))]


def test_a_quarter_uses_the_10q_statements():
    q2 = Filters.of(ticker="AAPL", fiscal_year=2025, fiscal_period="Q2")
    assert StatementFirstIndex(ByItem()).search("Apple's revenue in Q2?", 2, q2)[0].chunk_id == "Part I Item 1-0"


@pytest.mark.parametrize("query, filters", [
    ("What drove Apple's revenue growth in fiscal 2024?", FY),            # an explanation, not a figure
    ("What was Apple's net income?", Filters.of(ticker="AAPL")),           # no period: the form is unknown
    ("What was Apple's net income?", Filters.of(ticker="AAPL", fiscal_period="FY", item="Item 7")),   # Item given
])
def test_otherwise_the_search_is_unchanged(query, filters):
    base = ByItem()
    assert [h.chunk_id for h in StatementFirstIndex(base).search(query, 2, filters)] == [
        f"{filters.item or 'any'}-0", f"{filters.item or 'any'}-1"]
    assert len(base.calls) == 1


@pytest.mark.parametrize("question, figure", [
    ("What was Costco's net income in fiscal 2023?", True),
    ("By what percentage did Goldman Sachs' net earnings change?", True),
    ("What was NVIDIA's gross margin in fiscal 2023, in percent?", True),
    ("What drove the change in Microsoft's revenue in fiscal 2023?", False),
    ("What export-control risks does NVIDIA describe in its fiscal 2025 10-K?", False),
    ("What does UnitedHealth say about the Change Healthcare cyberattack?", False),
    ("When does Microsoft's fiscal year end?", False),
])
def test_figure_questions(question, figure):
    assert asks_for_figure(question) is figure


def test_statement_items():
    assert (statement_item("FY"), statement_item("Q3"), statement_item(None)) == ("Item 8", "Part I Item 1", None)
