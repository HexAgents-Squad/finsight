"""scripts/spot_check_sections.py (W1-10): a reproducible WP2 spot-check sample and its worksheet, offline."""
import json

import pytest

import scripts.spot_check_sections as sc

PROSE = "The Company reported higher net sales in every geographic segment during the fiscal year. "


def doc(ticker, form, item, n):
    text = f"{item}. Title\n" + PROSE * n
    period = "FY" if form == "10-K" else "Q1"
    return {"doc_id": f"{ticker}_{form}_2024_{item.replace(' ', '')}", "ticker": ticker, "form": form,
            "fiscal_year": 2024, "fiscal_period": period, "item": item, "item_title": "Title",
            "source_location": "main document", "source_url": f"https://www.sec.gov/{ticker}.htm",
            "n_chars": len(text), "text": text}


def corpus():
    docs = []
    for t in [f"T{i:02d}" for i in range(20)]:
        docs += [doc(t, "10-K", "Item 7", 400), doc(t, "10-K", "Item 14", 1), doc(t, "10-Q", "Part I Item 2", 200)]
    return docs


def test_the_sample_is_reproducible_and_spread_over_companies_and_forms():
    docs = corpus()
    picked = sc.sample(docs)
    assert picked == sc.sample(docs) and picked != sc.sample(docs, seed=11)
    assert len({d["ticker"] for d in picked}) == sc.N_10K + sc.N_10Q
    assert [d["form"] for d in picked] == ["10-K"] * sc.N_10K + ["10-Q"] * sc.N_10Q


def test_longer_sections_are_drawn_more_often():
    docs = corpus()
    items = [d["item"] for seed in range(30) for d in sc.sample(docs, seed) if d["form"] == "10-K"]
    assert items.count("Item 7") > 10 * items.count("Item 14")          # 400 sentences against 1


def test_the_worksheet_lists_each_section_with_its_link_text_and_a_results_row(tmp_path):
    src = tmp_path / "sections.jsonl"
    src.write_text("".join(json.dumps(d) + "\n" for d in corpus()), encoding="utf-8")
    out = tmp_path / "spot.md"
    assert sc.main(["--sections", str(src), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    picked = sc.sample(corpus())
    for d in picked:
        assert f"`{d['doc_id']}`" in text and d["source_url"] in text
    assert text.count("Sentences to find in the SEC page:") == 10
    assert text.count("| `") == 10 and "**Checked by:**" in text
    assert sc.main(["--sections", str(src), "--out", str(out)]) == 1          # never overwrites a worksheet
    assert sc.main(["--sections", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "x.md")]) == 2


def test_middle_sentences_are_prose_from_the_middle_in_order():
    text = "Head.\n" + "".join(f"Sentence number {i} says the company sold more products this year. " for i in range(40))
    found = sc.middle_sentences(text, seed=3)
    numbers = [int(s.split()[2]) for s in found]
    assert len(found) == sc.SENTENCES and numbers == sorted(numbers) and 8 <= min(numbers) and max(numbers) <= 31
    assert sc.middle_sentences("Revenue | 1,234 | 5,678\n" * 50, seed=1) == []     # table rows are no sentences


@pytest.mark.parametrize("sentence", [
    "The commercial allowance for loan and lease losses increased $153 million during 2025 to $4.8 billion.",
    "Special Resolution Regime confers substantially the same powers on the Bank of England, as the U.K. "
    "resolution authority, and substantially the same requirements on U.K. financial institutions.",
    "These include reviews by the Federal Trade Commission, U.S. Congressional committees and the SEC.",
    "The principal broker-dealer subsidiaries are BofA Securities, Inc. and Merrill Lynch, Pierce, Fenner & Smith.",
])
def test_a_decimal_or_an_abbreviation_does_not_end_a_sentence(sentence):
    # the first worksheet cut these at "$4.", "U." and "U." (W1-10 review of the screenshots)
    text = "Intro line.\n" + ("Filler words make this paragraph long enough to sit in the middle of it. " * 8 + "\n") * 2
    text = text + sentence + " " + text
    assert sentence in [m.group() for m in sc.SENTENCE.finditer(text)]


def test_a_sentence_starts_at_a_sentence_start():
    # not mid-sentence at a capitalised name ("... investee in Latin America and a loss of $7 million ...")
    # the search starts where the middle half begins, which can be inside a sentence (Coca-Cola, sentence 3)
    text = ("The earlier paragraph of this note ends with this sentence. During the period the company recorded a charge related to an equity method "
            "investee in Latin America and a loss of $7 million related to post-closing adjustments in Vietnam.")
    found = [m.group() for m in sc.SENTENCE.finditer(text, text.index("investee"))]
    assert found == []
    assert [m.group() for m in sc.SENTENCE.finditer(text)][-1].startswith("During the period")
