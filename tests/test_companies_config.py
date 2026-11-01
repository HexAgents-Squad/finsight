"""Contract tests for config/companies.yaml (the corpus scope every work package depends on)."""
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

COMPANIES_FILE = Path(__file__).resolve().parents[1] / "config" / "companies.yaml"
SECTORS = {"tech", "financials", "healthcare", "consumer"}


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(COMPANIES_FILE.read_text(encoding="utf-8"))


def test_schema_version_and_forms(config):
    assert config["schema_version"] == 2
    assert config["forms"]["10-K"]["fiscal_years"] == [2023, 2024, 2025]
    assert config["forms"]["10-Q"]["fiscal_years"] == [2025]
    assert config["forms"]["10-Q"]["fiscal_periods"] == ["Q1", "Q2", "Q3"]


def test_twenty_unique_companies_in_four_sectors(config):
    companies = config["companies"]
    tickers = [c["ticker"] for c in companies]
    assert len(companies) == 20
    assert len(set(tickers)) == 20, f"duplicate tickers: {[t for t, n in Counter(tickers).items() if n > 1]}"
    assert {c["sector"] for c in companies} == SECTORS


def test_every_company_has_valid_fields(config):
    for c in config["companies"]:
        assert re.fullmatch(r"[A-Z]{1,5}", c["ticker"]), c
        assert re.fullmatch(r"\d{10}", c["cik"]), f"{c['ticker']}: CIK must be 10 digits, zero-padded"
        assert c["name"].strip(), c
        assert 1 <= c["fiscal_year_end_month"] <= 12, c
        assert c["fiscal_year_end"].strip(), c
        assert c["aliases"], f"{c['ticker']}: needs at least one alias"


def test_aliases_are_unambiguous(config):
    owner: dict[str, str] = {}
    for c in config["companies"]:
        for alias in [*c["aliases"], c["ticker"]]:
            key = alias.lower()
            assert owner.setdefault(key, c["ticker"]) == c["ticker"], (
                f"alias '{alias}' used by both {owner[key]} and {c['ticker']}")
