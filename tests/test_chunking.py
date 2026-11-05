"""retrieval/chunking.py: the three chunking variants, context headers and deterministic IDs (offline)."""
import json

import pytest

from ingest import SAMPLE_DIR
from retrieval.chunking import (
    C1_OVERLAP,
    C1_SIZE,
    PACK_OVERLAP,
    PACK_SIZE,
    TABLE_MAX_CHARS,
    VARIANTS,
    chunk_corpus,
    chunk_document,
    fixed_windows,
    header,
    is_table,
    pack,
    split_table,
    units,
)


def doc(text, item="Item 7", period="FY", **kw):
    base = {"doc_id": f"AAPL_10-K_2024_{item.replace(' ', '')}", "ticker": "AAPL", "form": "10-K",
            "fiscal_year": 2024, "fiscal_period": period, "item": item, "item_title": "Some long title",
            "period_end": "2024-09-28", "source_url": "https://www.sec.gov/x.htm", "text": text}
    return {**base, **kw}


STATEMENT = ("CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions)\n\nYears ended\n"
             "September 28, 2024 | September 30, 2023\nNet sales:\nProducts | 294,866 | 298,085\n"
             "Total net sales | 391,035 | 383,285")


def test_header_names_the_filing_and_section():
    assert header(doc("x")) == "[AAPL 10-K FY2024 · Item 7 · MD&A]"
    assert header(doc("x", item="Part I Item 2", period="Q2", form="10-Q", fiscal_year=2025)) == \
        "[AAPL 10-Q FY2025 Q2 · Part I Item 2 · MD&A]"
    assert header(doc("x", item="Item 9A")) == "[AAPL 10-K FY2024 · Item 9A · Some long title]"


def test_fixed_windows_have_the_size_and_overlap():
    text = "".join(chr(65 + i % 26) for i in range(2_000))
    windows = fixed_windows(text, C1_SIZE, C1_OVERLAP)
    assert all(len(w) == C1_SIZE for w in windows[:-1])
    assert windows[1][:C1_OVERLAP] == windows[0][-C1_OVERLAP:]
    assert "".join(w[: C1_SIZE - C1_OVERLAP] for w in windows[:-1]) + windows[-1] == text


def test_long_paragraphs_are_broken_into_units_that_fit():
    sentence = "Revenue grew because of strong demand. "
    text = "Short intro.\n\n" + sentence * 100
    parts = units(text)
    assert parts[0] == "Short intro." and all(len(u) <= PACK_SIZE for u in parts)


def test_packing_respects_the_size_and_overlaps_at_a_word_boundary():
    words = [f"para{i} " + "word " * 40 for i in range(30)]
    chunks = pack(words, PACK_SIZE, PACK_OVERLAP)
    assert all(len(c) <= PACK_SIZE + PACK_OVERLAP + 1 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail = nxt.split("\n")[0]
        assert tail and prev.endswith(tail) and len(tail) <= PACK_OVERLAP and not tail.startswith(" ")


@pytest.mark.parametrize("variant", VARIANTS)
def test_chunks_never_cross_a_section_and_ids_are_deterministic(variant):
    docs = [doc("First section. " * 200), doc("Second section. " * 200, item="Item 1A")]
    chunks = chunk_corpus(docs, variant)
    assert {c.doc_id for c in chunks} == {d["doc_id"] for d in docs}
    assert all(("First" in c.text) != ("Second" in c.text) for c in chunks)
    assert chunks == chunk_corpus(docs, variant)
    assert chunks[0].chunk_id == f"{docs[0]['doc_id']}_{variant}_0000"
    assert chunks[0].index_text.startswith(chunks[0].header + "\n")


def test_table_detection():
    assert is_table("Years ended\n2024 | 2023\nNet sales:\nProducts | 1 | 2")
    assert not is_table("Revenue | grew\nThis paragraph is prose with one pipe.\nAnd more prose.")


def test_c3_keeps_a_table_whole_with_its_caption():
    text = "Intro paragraph about results.\n\n" + STATEMENT + "\n\nClosing prose."
    chunks = chunk_document(doc(text, item="Item 8"), "C3")
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].text.startswith("CONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions)\nYears ended")
    assert "Total net sales | 391,035" in tables[0].text
    assert [c.text for c in chunks if c.kind == "text"] == ["Intro paragraph about results.", "Closing prose."]


def test_a_table_over_the_cap_is_split_by_rows_repeating_its_header():
    rows = "\n".join(f"Line item {i} | {i:,} | {i + 1:,}" for i in range(400))
    table = "Years ended\n2024 | 2023\n" + rows
    parts = split_table(table, "BALANCE SHEET")
    assert len(parts) > 1 and all(len(p) <= TABLE_MAX_CHARS for p in parts)
    assert all(p.startswith("BALANCE SHEET\nYears ended\n2024 | 2023\n") for p in parts)
    body = [ln for p in parts for ln in p.split("\n")[3:]]
    assert body == rows.split("\n")                       # every row kept once, in order


def test_unknown_variant_is_an_error():
    with pytest.raises(ValueError, match="unknown chunking variant"):
        chunk_document(doc("x"), "C9")


@pytest.mark.parametrize("variant", VARIANTS)
def test_sample_corpus(variant):
    docs = [json.loads(line) for line in (SAMPLE_DIR / "sections_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    chunks = chunk_corpus(docs, variant)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) and {c.doc_id for c in chunks} == {d["doc_id"] for d in docs}
    assert all(c.text.strip() for c in chunks)
