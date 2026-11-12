"""The hand-written interpretation of eval/results/model_comparison.md (W1-15) and ADR-019 against the generated
tables and the score files: a re-scored run that changes a number, or an example quoted under the wrong label,
makes these tests fail instead of leaving a stale sentence for the mentor."""
import json
import re

import eval.score_answers as sa

REPORT = sa.RESULTS_DIR / sa.REPORT_NAME
DECISIONS = sa.RESULTS_DIR.parents[1] / "docs" / "decisions.md"
FRACTION = re.compile(r"(?<![\w/.])(\d+)/(\d+)(?![\w/])")
EXAMPLE_REF = re.compile(r"\b(M[123]) ([A-Z]{1,2}\d{2})\b")
RUNS = {"M1": "M1_qwen3-8b_rag_think-off_b0-v1", "M2": "M2_llama3.2-3b_rag_think-off_b0-v1",
        "M3": "M3_qwen3-8b_closed_think-off_b0-v1"}
# Figures cited from outside the generated tables, each with where it comes from (checked below where possible)
OUTSIDE_TABLES = {
    "69/73": "context_pack_gold.meta.json: evidence_in_pack / with_gold_sources",
    "20/32": "table 7: the three runs' graders' exact agreement added up",
    "18/46": "table 8: M1's numeric questions with all gold figures in the passages, of all with a dollar figure",
    "30/102": "table 5: 102 questions minus M2's 72 valid B0 replies",
    "163/177": "W1-14 merge output: labels given by both graders, and how many agree (PR #23; not in the repo)",
}


def parts():
    text = REPORT.read_text(encoding="utf-8")
    start, end = text.index(sa.START), text.index(sa.END)
    return text[start:end], text[end:]


def adr_019() -> str:
    return next(line for line in DECISIONS.read_text(encoding="utf-8").splitlines() if line.startswith("| ADR-019 |"))


def table_rows(block: str, heading: str) -> dict[str, list[str]]:
    """Rows of the markdown table under `heading`: first cell -> the other cells."""
    section = block.split(heading, 1)[1].split("\n###", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and not set(line) <= set("|-") and len(cells) > 1:
            rows[cells[0]] = cells[1:]
    return rows


def scores() -> dict[str, dict[str, dict]]:
    return {run: {s["id"]: s for s in (json.loads(x) for x in (sa.SCORES_DIR / f"{rid}.jsonl").read_text(
        encoding="utf-8").splitlines())} for run, rid in RUNS.items()}


def test_every_fraction_cited_is_in_the_tables_or_has_a_named_source():
    generated, interpretation = parts()
    in_tables = {f"{a}/{b}" for a, b in FRACTION.findall(generated)}
    for name, text in (("interpretation", interpretation), ("ADR-019", adr_019())):
        cited = {f"{a}/{b}" for a, b in FRACTION.findall(text)}
        assert cited, f"{name} cites no figures"
        assert cited - in_tables <= set(OUTSIDE_TABLES), (name, sorted(cited - in_tables - set(OUTSIDE_TABLES)))


def test_the_figures_from_outside_the_tables_still_hold():
    generated, _ = parts()
    meta = json.loads((sa.RESULTS_DIR / "context_pack_gold.meta.json").read_text(encoding="utf-8"))
    assert f"{meta['evidence_in_pack']}/{meta['with_gold_sources']}" == "69/73"
    agree = table_rows(generated, "### 7.")["graders' exact agreement (independent grades)"]
    assert "/".join(str(sum(int(c.split("/")[i]) for c in agree)) for i in (0, 1)) == "20/32"
    rows = table_rows(generated, "### 8.")
    m1 = [rows[k][0].split(" =")[0] for k in ("all in the passages", "not all in the passages")]
    with_figures = sum(int(x.split("/")[1]) for x in m1)
    assert f"{m1[0].split('/')[1]}/{with_figures}" == "18/46"
    valid = table_rows(generated, "### 5.")["valid B0 JSON"][1]
    assert f"{102 - int(valid.split('/')[0])}/102" == "30/102"


def test_the_overall_graders_kappa_quoted_is_recomputed_from_the_score_files():
    pairs = [tuple(s["grader_grades"]) for run in scores().values() for s in run.values() if s.get("grader_grades")]
    assert len(pairs) == 32 and f"{sa.kappa(pairs):.2f}" == "0.43"
    assert "kappa 0.43" in parts()[1] and "kappa 0.43" in adr_019()


def test_the_taxonomy_table_repeats_table_6_and_its_examples_carry_that_label():
    generated, interpretation = parts()
    counts = table_rows(generated, "### 6.")
    quoted = table_rows(interpretation, "### Failure taxonomy")
    quoted.pop("Label")
    assert set(quoted) == set(counts) - {"Label"}
    by_run = scores()
    for label, cells in quoted.items():
        assert cells[:3] == counts[label], label
        refs = EXAMPLE_REF.findall(cells[3])
        assert refs, f"{label}: no example"
        for run, qid in refs:
            assert by_run[run][qid]["failure_label"] == label, (label, run, qid)


def test_retrieval_misses_quoted_are_misses():
    by_run = scores()
    for qid in ("A09", "A15", "B01", "B05", "C04", "A12", "C01", "F03"):
        assert by_run["M1"][qid]["gold_figures_in_pack"] is False, qid
