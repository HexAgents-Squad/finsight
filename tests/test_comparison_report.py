"""The hand-written interpretation of eval/results/model_comparison.md (W1-15) against its generated tables: a
re-scored run that changes a number makes these tests fail instead of leaving a stale sentence for the mentor."""
import json
import re

import eval.score_answers as sa

REPORT = sa.RESULTS_DIR / sa.REPORT_NAME
FRACTION = re.compile(r"(?<![\w/.])(\d+)/(\d+)(?![\w/])")
# Figures the interpretation cites from outside the generated tables, each with the file it comes from
OUTSIDE_TABLES = {
    "69/73": "context_pack_gold.meta.json: evidence_in_pack / with_gold_sources",
    "20/32": "table 7: the three runs' graders' exact agreement added up",
    "163/177": "W1-14 merge output: labels given by both graders, and how many agree (PR #23)",
    "30/102": "table 5: 102 questions minus 72 valid B0 replies",
}


def parts():
    text = REPORT.read_text(encoding="utf-8")
    start, end = text.index(sa.START), text.index(sa.END)
    return text[start:end], text[end:]


def table_rows(block: str, heading: str) -> dict[str, list[str]]:
    """Rows of the markdown table under `heading`: first cell -> the other cells."""
    section = block.split(heading, 1)[1].split("\n###", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and not set(line) <= set("|-") and len(cells) > 1:
            rows[cells[0]] = cells[1:]
    return rows


def test_every_fraction_in_the_interpretation_is_in_the_tables_or_has_a_named_source():
    generated, interpretation = parts()
    in_tables = {f"{a}/{b}" for a, b in FRACTION.findall(generated)}
    cited = {f"{a}/{b}" for a, b in FRACTION.findall(interpretation)}
    assert cited, "the interpretation cites no figures"
    assert cited - in_tables <= set(OUTSIDE_TABLES), sorted(cited - in_tables - set(OUTSIDE_TABLES))


def test_the_figures_from_outside_the_tables_still_hold():
    generated, _ = parts()
    meta = json.loads((sa.RESULTS_DIR / "context_pack_gold.meta.json").read_text(encoding="utf-8"))
    assert f"{meta['evidence_in_pack']}/{meta['with_gold_sources']}" == "69/73"
    agree = table_rows(generated, "### 7.")["graders' exact agreement (independent grades)"]
    done, total = (sum(int(c.split("/")[i]) for c in agree) for i in (0, 1))
    assert f"{done}/{total}" == "20/32"
    valid = table_rows(generated, "### 5.")["valid B0 JSON"][1]               # M2
    assert f"{102 - int(valid.split('/')[0])}/102" == "30/102"


def test_the_taxonomy_table_repeats_table_6():
    generated, interpretation = parts()
    counts = table_rows(generated, "### 6.")
    quoted = table_rows(interpretation, "### Failure taxonomy")
    quoted.pop("Label")
    assert set(quoted) == set(counts) - {"Label"}
    for label, cells in quoted.items():
        assert cells[:3] == counts[label], label
