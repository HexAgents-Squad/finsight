"""retrieval/entities.py (W1-05): companies, fiscal years and periods read from a question, and the filters."""
import pytest

from retrieval.common import Filters
from retrieval.entities import Entities, companies, extract, from_turns, periods, years


@pytest.mark.parametrize("text, tickers", [
    ("What was Apple's revenue?", ("AAPL",)),
    ("Compare JPMorgan Chase with Bank of America.", ("JPM", "BAC")),          # the longest name wins
    ("How did J.P. Morgan and Goldman Sachs' net income compare?", ("JPM", "GS")),
    ("Which had more cash, P&G or J&J?", ("PG", "JNJ")),
    ("What did Facebook report?", ("META",)),                                  # a former name in the aliases
    ("Revenue of Wal-Mart and Coca-Cola", ("WMT", "KO")),
    ("What was C's net income?", ("C",)),                                      # a ticker in capitals
    ("Explain the change in MSFT and NVDA margins", ("MSFT", "NVDA")),
    ("Did Citi and Citigroup report the same?", ("C",)),                       # one company, named twice
])
def test_companies_by_name_alias_or_ticker_in_order(text, tickers):
    assert companies(text) == tickers


@pytest.mark.parametrize("text", [
    "What was the company's net income? c. 2024",                              # the letter c is no ticker
    "Item 1C cybersecurity disclosures",                                        # not a ticker inside a word
    "a meta-analysis of the costs", "Applesauce sales",                          # no partial-word aliases
    "What was Tesla's revenue?",                                                # not in config/companies.yaml
])
def test_no_company_from_letters_parts_of_words_or_other_companies(text):
    assert companies(text) == ()


@pytest.mark.parametrize("text, found", [
    ("in fiscal 2023, in percent?", (2023,)), ("FY2024 results", (2024,)), ("FY24 and FY25", (2024, 2025)),
    ("from fiscal 2023 to fiscal 2025", (2023, 2025)), ("in 2024.", (2024,)),
    ("revenue of $2,024 million", ()), ("grew 2024.5 percent", ()), ("12024 units", ()),
])
def test_fiscal_years(text, found):
    assert years(text) == found


@pytest.mark.parametrize("text, found", [
    ("in the second quarter of fiscal 2025", ("Q2",)), ("Q3 2025", ("Q3",)), ("2025Q1 revenue", ("Q1",)),
    ("the six months ended March 2025", ("Q2",)), ("nine months", ("Q3",)),
    ("in the first half of fiscal 2025", ("Q2",)), ("the first  nine months", ("Q3",)),
    ("NVIDIA revenue 1Q25", ("Q1",)), ("Apple Q2FY25 revenue", ("Q2",)), ("3Q2025", ("Q3",)),
    ("the three months ended March 29, 2025", ("Q1", "Q2", "Q3")),     # which quarter depends on the fiscal year
    ("the fourth quarter of fiscal 2024", ("FY",)),                    # Q4 is in the annual report
    ("in its fiscal 2024 10-K", ("FY",)), ("in its latest 10-Q", ("Q1", "Q2", "Q3")),
    ("in fiscal 2024", ()),                                            # no period named: FY when filtering
    ("Q1 and Q2 of fiscal 2025", ("Q1", "Q2")),
])
def test_fiscal_periods(text, found):
    assert periods(text) == found


def test_latest_reads_as_the_newest_indexed_year_or_quarter():
    assert extract("What was Microsoft's revenue in its latest fiscal year?") == Entities(("MSFT",), (2025,), ())
    assert extract("What was Apple's revenue in the most recent quarter?") == Entities(("AAPL",), (2025,), ("Q3",))


def test_a_follow_up_keeps_what_it_does_not_name():
    turns = ["What was Meta's revenue in the second quarter of fiscal 2025?", "And in the third quarter?"]
    assert from_turns(turns) == Entities(("META",), (2025,), ("Q3",))
    turns = ["What were UnitedHealth's total revenues in fiscal 2023?", "And its net earnings that year?"]
    assert from_turns(turns) == Entities(("UNH",), (2023,), ())
    assert from_turns(["What about Apple?"]) == Entities(("AAPL",), (), ())
    assert from_turns([]) == Entities()


def test_filters_per_company_year_and_period():
    e = Entities(("AAPL", "MSFT"), (2024,), ())
    assert e.filters() == [Filters.of(ticker="AAPL", fiscal_year=2024), Filters.of(ticker="MSFT", fiscal_year=2024)]
    assert e.filters(period=True) == [Filters.of(ticker="AAPL", fiscal_year=2024, fiscal_period="FY"),
                                      Filters.of(ticker="MSFT", fiscal_year=2024, fiscal_period="FY")]
    q = Entities(("META",), (2025,), ("Q2",))
    assert q.filters(period=True) == [Filters.of(ticker="META", fiscal_year=2025, fiscal_period="Q2")]


def test_a_period_the_corpus_does_not_index_keeps_company_and_year():
    # only fiscal-2025 10-Qs are indexed: fiscal 2024 Q2 searches fiscal 2024 (the 10-K) instead of nothing
    e = Entities(("AAPL",), (2024, 2025), ("Q2",))
    assert e.filters(period=True) == [Filters.of(ticker="AAPL", fiscal_year=2024),
                                      Filters.of(ticker="AAPL", fiscal_year=2025, fiscal_period="Q2")]


def test_no_company_and_no_year_means_no_filter():
    assert Entities().filters() == [] and Entities().filters(period=True) == []
    assert Entities((), (2024,), ()).filters() == [Filters.of(fiscal_year=2024)]
    assert Entities(("KO",), (), ()).filters(period=True) == [Filters.of(ticker="KO", fiscal_period="FY")]


def test_a_longer_name_wins_over_another_companys_shorter_one(tmp_path, monkeypatch):
    """Names are tried longest first, so "Bank of America" is not read as a company called "Bank" (no two
    companies in config/companies.yaml share such a prefix today; this keeps the rule for when they do)."""
    import retrieval.entities as entities
    config = tmp_path / "companies.yaml"
    config.write_text("companies:\n  - {ticker: BK, name: Bank}\n  - {ticker: BAC, name: Bank of America}\n",
                      encoding="utf-8")
    monkeypatch.setattr(entities, "COMPANIES_FILE", config)
    entities._names.cache_clear()
    try:
        assert companies("Bank of America's net income") == ("BAC",)
        assert companies("Bank's net income") == ("BK",)
    finally:
        entities._names.cache_clear()


@pytest.mark.parametrize("text, found", [("NVIDIA revenue 1Q25", (2025,)), ("Apple Q2FY25 revenue", (2025,)),
                                         ("3Q2025", (2025,))])
def test_years_in_quarter_tokens(text, found):
    assert years(text) == found


def test_a_share_class_is_not_a_ticker():
    assert companies("Alphabet Class C shares repurchased in fiscal 2024") == ("GOOGL",)
    assert companies("Series C preferred stock of C") == ("C",)


def test_a_follow_up_naming_only_a_year_asks_about_that_fiscal_year():
    read = from_turns(["What was Apple's revenue in Q3 2025?", "What about fiscal 2024?"])
    assert (read.tickers, read.years, read.periods) == (("AAPL",), (2024,), ())
    assert read.filters(period=True) == [Filters.of(ticker="AAPL", fiscal_year=2024, fiscal_period="FY")]


def test_a_year_only_turn_in_the_middle_ends_an_earlier_quarter():
    read = from_turns(["What was Apple's revenue in Q2 fiscal 2025?", "And in fiscal 2025 as a whole?",
                       "And Microsoft's?"])
    assert (read.tickers, read.years, read.periods) == (("MSFT",), (2025,), ())
    assert from_turns(["Apple's revenue in Q2 fiscal 2025?", "And Microsoft's?"]).periods == ("Q2",)


@pytest.mark.parametrize("text, found", [
    ("Did Apple chase growth in 2024?", ("AAPL",)), ("What meta trends did Pfizer note?", ("PFE",)),
    ("What did Chase report?", ("JPM",)), ("META's revenue", ("META",)),
    ("what did bank of america report?", ("BAC",)),               # a name of several words: any case
])
def test_a_one_word_name_in_lower_case_is_an_ordinary_word(text, found):
    assert companies(text) == found


def test_entities_are_tuples_whatever_sequence_is_given():
    restored = Entities(["AAPL", "MSFT"], [2024], [])           # as a checkpoint restores them
    assert restored == Entities(("AAPL", "MSFT"), (2024,), ()) and hash(restored)
