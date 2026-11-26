"""scripts/final_report.py (W3-12, ADR-047): the 12-page report is traceable, follows the plan's outline, and its PDF is
current. Offline and without a browser: the PDF is built by hand with Chrome; these tests run --check's logic."""
from __future__ import annotations

import json
import re

import pytest

import scripts.final_report as fr

REPORT = fr.SOURCE.read_text(encoding="utf-8")
OUTLINE = ["Problem, users and scope", "System architecture", "Data", "Evaluation design", "Prompting baseline",
           "Retrieval experiments", "Agentic layer", "Compensating for a small model", "End-to-end results",
           "Responsible AI", "Limitations and future work"]       # FinSight_Project_Plan.txt §13, the report outline
MEMBERS = ["Alok Srivastava", "Anirudh M K", "Chandra Shekhar Pandey", "Amresh Kumar", "Giridharan Selvaraj"]


def test_the_committed_report_passes_every_check():
    assert fr.check() == []                  # traceable, paths exist, the PDF built from this text, <= 12 pages


def test_the_report_follows_the_outline_and_names_an_owner_and_sources_for_each_section():
    body = [s for s in fr.sections(fr.body_only(REPORT)) if s.title != "front matter"]
    assert len(body) == len(OUTLINE)
    for n, (s, part) in enumerate(zip(body, OUTLINE, strict=True), start=1):
        assert s.title.startswith(f"{n}. ") and part in s.title, (n, s.title)
    for s in body:
        assert re.search(r"^\*Owners? \(reviews?\): .+\*$", s.text, re.M), s.title
        assert s.sources, s.title
    for name in MEMBERS:
        assert name in REPORT


def test_the_build_record_is_within_the_page_limit():
    record = json.loads(fr.RECORD.read_text(encoding="utf-8"))
    assert record["pages_total"] <= fr.MAX_PAGES                      # the whole PDF, appendix included
    assert record["pages_body"] < record["pages_total"]
    assert record["pages_total"] == fr.pdf_pages(fr.PDF.read_bytes())


def _report(tmp_path, body: str, source: str = "") -> tuple[str, object]:
    (tmp_path / "res.md").write_text(source, encoding="utf-8")
    return body, tmp_path


def test_a_number_its_sources_do_not_hold_is_reported(tmp_path):
    md, root = _report(tmp_path, "## 1. A\n\nRight 42/48 and 1,563 facts, 0.774 recall.\n\n*Sources:* `res.md`\n",
                       "| x | 42/48 | 1563 | 0.774 |")
    assert fr.untraced(md, root) == []
    md2 = md.replace("42/48", "43/48").replace("0.774", "0.775")
    assert fr.untraced(md2, root) == ["1. A: 0.775 is in none of its sources", "1. A: 43/48 is in none of its sources"]


def test_a_count_with_a_separator_is_a_claim_even_in_the_range_of_years(tmp_path):
    md, root = _report(tmp_path, "## 1. A\n\nUp to 2,000 documents, in 2024.\n\n*Sources:* `res.md`\n", "none")
    assert fr.untraced(md, root) == ["1. A: 2000 is in none of its sources"]


def test_a_fraction_is_one_claim_and_names_years_and_small_counts_are_not_claims(tmp_path):
    md, root = _report(tmp_path, "## 1. A\n\nIn 2024, W3-03 and ADR-039 ran 3 checks on Qwen3-8B; 5/12 right.\n\n"
                                 "*Sources:* `res.md`\n", "5 and 12 appear apart")
    assert fr.untraced(md, root) == ["1. A: 5/12 is in none of its sources"]
    (root / "res.md").write_text("summary 5/12", encoding="utf-8")
    assert fr.untraced(md, root) == []


def test_missing_sources_and_unsourced_numbers_are_reported(tmp_path):
    md = "## 1. A\n\nIt took 8 min 34 s.\n\n## 2. B\n\nSee 1,260 documents.\n\n*Sources:* `nope.md`\n"
    assert fr.untraced(md, tmp_path) == ["1. A: states numbers (34 ...) but cites no source",
                                         "2. B: cites nope.md, which does not exist",
                                         "2. B: 1260 is in none of its sources"]


def test_code_spans_comments_and_the_owner_line_hold_no_claims(tmp_path):
    md, root = _report(tmp_path, "## 1. A\n\n*Owner (reviews): someone with 99 tickets.*\n\nRun `--port 8501` "
                                 "<!-- 777 -->and [the PR](https://x/pull/71).\n\n*Sources:* `res.md`\n")
    assert fr.untraced(md, root) == []


def test_the_rubric_checklist_names_only_existing_paths(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x", encoding="utf-8")
    text = "| x | `docs/a.md`, `docs/`, `make up`, `docs/missing.md` |"
    assert fr.checklist_problems(text, tmp_path) == ["rubric checklist: docs/missing.md does not exist"]
    assert fr.checklist_problems(fr.CHECKLIST.read_text(encoding="utf-8")) == []


def test_markdown_renders_tables_nested_lists_figures_and_escapes_text():
    html = fr.to_html("## T\n\nA < B & **bold** `x<y`\n\n| Check | n |\n|---|---|\n| a | 1 |\n\n- one\n  - two\n- three\n\n"
                      "![Figure 1. Arch.](figures/architecture.svg)\n\n*Sources:* `a.md`\n")
    assert "<p>A &lt; B &amp; <strong>bold</strong> <code>x&lt;y</code></p>" in html
    assert ('<table><thead><tr><th>Check</th><th class="num">n</th></tr></thead><tbody>\n'
            '<tr><td>a</td><td class="num">1</td></tr>') in html               # figures right-aligned, words left
    assert html.count("<ul>") == 2 and html.count("</ul>") == 2
    assert 'src="data:image/svg+xml;base64,' in html and "<figcaption>Figure 1. Arch.</figcaption>" in html
    assert '<p class="meta"><em>Sources:</em> <code>a.md</code></p>' in html


def test_the_html_is_deterministic_and_the_appendix_is_cut_for_the_page_count():
    assert fr.html_sha256(REPORT) == fr.html_sha256(REPORT)
    assert fr.APPENDIX in REPORT and fr.APPENDIX not in fr.body_only(REPORT)
    heading = "<h2>Appendix A."
    assert heading not in fr.page(fr.body_only(REPORT)) and heading in fr.page(REPORT)


def test_an_edited_report_makes_the_pdf_stale(monkeypatch, tmp_path):
    source = tmp_path / "final_report.md"
    source.write_text(REPORT.replace("Summary.", "Summary (edited)."), encoding="utf-8")
    monkeypatch.setattr(fr, "SOURCE", source)
    assert any("the PDF is stale" in p for p in fr.check())


def test_pdf_pages_counts_page_objects_not_the_page_tree():
    fake = b"<< /Type /Pages /Count 2 >> << /Type /Page /Parent 1 0 R >> << /Type/Page >>"
    assert fr.pdf_pages(fake) == 2


@pytest.mark.parametrize("token, expected", [("1,563", True), ("12", False), ("13", True), ("2025", False),
                                             ("2,000", True), ("0.5", True), ("3.50", True)])
def test_which_numbers_are_checked(token, expected):
    assert fr.checked(token) is expected
