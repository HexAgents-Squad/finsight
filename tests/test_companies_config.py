"""Contract tests for config/companies.yaml (the corpus scope every work package depends on)."""
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

COMPANIES_FILE = Path(__file__).resolve().parents[1] / "config" / "companies.yaml"

# The agreed corpus scope (spec §3, ADR-002). Changing any of these is a contract change: update the spec too.
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_COMPANY_COUNT = 20
SECTOR_COUNTS = {"tech": 6, "financials": 5, "healthcare": 5, "consumer": 4}   # ADR-002
TENK_FISCAL_YEARS = [2023, 2024, 2025]
TENQ_FISCAL_YEARS = [2025]
TENQ_FISCAL_PERIODS = ["Q1", "Q2", "Q3"]
TICKER_PATTERN = re.compile(r"[A-Z]{1,5}")
CIK_PATTERN = re.compile(r"\d{10}")                     # SEC CIK, zero-padded to 10 digits


def alias_key(alias: str) -> str:
    """Case- and whitespace-insensitive form, so "Apple", " apple " and "APPLE" collide."""
    return " ".join(alias.split()).casefold()


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(COMPANIES_FILE.read_text(encoding="utf-8"))


def test_schema_version_and_forms(config):
    assert config["schema_version"] == EXPECTED_SCHEMA_VERSION
    assert config["forms"]["10-K"]["fiscal_years"] == TENK_FISCAL_YEARS
    assert config["forms"]["10-Q"]["fiscal_years"] == TENQ_FISCAL_YEARS
    assert config["forms"]["10-Q"]["fiscal_periods"] == TENQ_FISCAL_PERIODS


def test_twenty_unique_companies_with_agreed_sector_split(config):
    companies = config["companies"]
    tickers = [c["ticker"] for c in companies]
    assert len(companies) == EXPECTED_COMPANY_COUNT
    assert len(set(tickers)) == EXPECTED_COMPANY_COUNT, f"duplicate tickers: {[t for t, n in Counter(tickers).items() if n > 1]}"
    assert Counter(c["sector"] for c in companies) == SECTOR_COUNTS


def test_every_company_has_valid_fields(config):
    for c in config["companies"]:
        assert TICKER_PATTERN.fullmatch(c["ticker"]), c
        assert CIK_PATTERN.fullmatch(c["cik"]), f"{c['ticker']}: CIK must be 10 digits, zero-padded"
        assert c["name"].strip(), c
        assert 1 <= c["fiscal_year_end_month"] <= 12, c
        assert c["fiscal_year_end"].strip(), c
        aliases = c["aliases"]
        assert isinstance(aliases, list) and aliases, f"{c['ticker']}: aliases must be a non-empty list"
        assert all(isinstance(a, str) and a.strip() for a in aliases), (
            f"{c['ticker']}: aliases must be non-empty strings, got {aliases}")


def test_aliases_are_unambiguous(config):
    owner: dict[str, str] = {}
    for c in config["companies"]:
        for alias in [*c["aliases"], c["ticker"]]:
            key = alias_key(alias)
            assert owner.setdefault(key, c["ticker"]) == c["ticker"], (
                f"alias '{alias}' used by both {owner[key]} and {c['ticker']}")
