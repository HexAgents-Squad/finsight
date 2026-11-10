"""eval/score_answers.py and eval/runs_schema.py, offline: synthetic runs over the real gold set, refusal tests and
context packs (no model, no network)."""
import csv
import json
import re

import pytest
from pydantic import ValidationError

import eval.score_answers as sa
from eval.runs_schema import B0Answer, RunMeta, RunRecord

QUESTIONS = sa.load_questions(sa.GOLD_FILE, sa.REFUSALS_FILE)
PACKS = sa.load_pack(sa.RESULTS_DIR / sa.GOLD_PACK) | sa.load_pack(sa.RESULTS_DIR / sa.REFUSAL_PACK)
MATCHERS = sa.company_matchers()
RAG, CLOSED = "M1_qwen3-8b_rag_think-off_b0-v1", "M3_qwen3-8b_closed_think-off_b0-v1"
NAMES = {"AAPL": "Apple", "GOOGL": "Alphabet", "JPM": "JPMorgan Chase", "C": "Citigroup", "JNJ": "Johnson & Johnson",
         "ABBV": "AbbVie", "WMT": "Walmart", "PG": "Procter & Gamble", "MSFT": "Microsoft", "META": "Meta",
         "BAC": "Bank of America", "GS": "Goldman Sachs", "PFE": "Pfizer", "MRK": "Merck", "COST": "Costco",
         "KO": "Coca-Cola", "NVDA": "NVIDIA"}


PACK_SHA = {name: sa.sha256(sa.RESULTS_DIR / name) for name in (sa.GOLD_PACK, sa.REFUSAL_PACK)}


def meta(run_id=RAG, context="rag"):
    return RunMeta(run_id=run_id, model="Qwen/Qwen3-8B", endpoint="hosted", context=context, thinking=False,
                   temperature=0.0, max_tokens=1024, prompt_version="b0-v1",
                   context_pack="eval/results/context_pack_gold.jsonl (cp-v1)", context_pack_sha256=PACK_SHA)


def ideal(q: sa.Question) -> B0Answer:
    """The answer a perfect model gives: the gold answer, value and decision."""
    year = q.fiscal_years[-1] if q.fiscal_years else None
    if q.answer_type == "short_numeric":
        return B0Answer(answer=q.gold_answer, value=q.gold_value, unit=q.unit, company=q.companies[0],
                        fiscal_year=year, citations=["P1"], refused=False)
    if q.answer_type == "short_entity":
        return B0Answer(answer=q.gold_answer, company=NAMES[sa.D_WINNERS[q.id]], fiscal_year=year,
                        citations=["P1"], refused=False)
    if q.answer_type == "trap":
        return B0Answer(answer="not reported: the statements have no such line.", refused=False)
    if q.answer_type in ("refusal", "partial", "control"):
        return B0Answer(answer="I can't help with that." if q.expected_refused else "Answered from the filing.",
                        refused=q.expected_refused, refusal_rule=q.expected_rule)
    return B0Answer(answer=q.gold_answer, company=q.companies[0] if q.companies else None, fiscal_year=year,
                    citations=["P1"], refused=False)


def record(qid, parsed, run_id=RAG, **kw):
    fields = {"id": qid, "run_id": run_id, "raw_reply": parsed.model_dump_json() if parsed else "oops",
              "parsed": parsed, "json_valid": parsed is not None, "retries": 0, "prompt_tokens": 3000,
              "completion_tokens": 80, "latency_s": 1.0, "cached": False, "error": None} | kw
    return RunRecord(**fields)


def perfect_run(run_id=RAG, **overrides):
    return [record(qid, overrides.get(qid, ideal(q)), run_id) for qid, q in QUESTIONS.items()]


def score_one(qid, answer, context="rag", **kw):
    return sa.check(QUESTIONS[qid], record(qid, answer, **kw), PACKS.get(qid, {}), context == "rag", MATCHERS)


def answer(**kw):
    return B0Answer(**({"answer": "x", "refused": False} | kw))


# --- numbers ---------------------------------------------------------------------------------------------------

def test_figures_skip_dates_years_ids_and_names_and_apply_scale_words():
    text = ("Net sales were $391.0 billion in fiscal 2024 (year ended September 28, 2024) [P1], per Item 8 of the "
            "10-K for Q4 and FY2024; margin 44.6%; 1,234 stores; P12.")
    got = [(f.text, f.value, f.tolerance, f.percent) for f in sa.figures(text)]
    assert got == [("391.0 billion", 391e9, 0.05e9, False), ("44.6%", 44.6, 0.05, True), ("1,234", 1234.0, 0.5, False)]


@pytest.mark.parametrize("shown, reference, ok", [
    ("$391.0 billion", "391,035", True),        # rounded billions of a figure printed in millions
    ("$391.1 billion", "391,035", False),       # beyond half a unit of the last digit
    ("391,035", "391,035", True),
    ("44.6%", "44.6%", True),
    ("44.6", "44.6%", False),                   # a percentage only matches a percentage
    ("44.65%", "44.6%", True),                  # more decimals than the gold: the coarser precision decides
    ("$1.37 trillion", "1,371,804", True),
    ("$10.1 billion", "$10.1 billion", True),
])
def test_supported_accepts_rounded_forms_only(shown, reference, ok):
    found = sa.figures(shown)
    assert len(found) == 1 and sa.supported(found[0], sa.figures(reference)) == ok


@pytest.mark.parametrize("unit, scale", [("USD millions", 1e6), ("USD billions", 1e9), ("$bn", 1e9),
                                         ("in millions", 1e6), ("USD", 1.0), ("%", None), ("USD per share", None),
                                         (None, None), ("", None)])
def test_unit_scale(unit, scale):
    assert sa.unit_scale(unit) == scale


def test_the_value_is_checked_in_its_own_unit():
    assert sa.supported(sa.value_figure(answer(value=391.035, unit="USD billions")), sa.figures("391,035"))
    assert sa.supported(sa.value_figure(answer(value=44.6, unit="%")), sa.figures("44.6%"))
    assert not sa.supported(sa.value_figure(answer(value=44.6, unit="%")), sa.figures("44.7%"))


# --- a perfect run ---------------------------------------------------------------------------------------------

def test_a_perfect_run_scores_everything_correct_and_flags_nothing():
    scored = sa.score_run(meta(), perfect_run(), QUESTIONS, PACKS, MATCHERS)
    by_type = {t: [s for s in scored if s["answer_type"] == t] for t in sa.ANSWER_TYPES}
    assert {t: len(v) for t, v in by_type.items()} == {"short_numeric": 48, "short_entity": 10, "summary": 12,
                                                        "refusal": 20, "trap": 3, "partial": 2, "control": 7}
    assert all(s["correct"] is True for s in scored if s["answer_type"] != "summary")
    assert all(s["correct"] is None for s in by_type["summary"])                   # waits for the human grade
    assert [s["id"] for s in scored if s["checks"].get("unsupported_numbers")] == []
    assert [s["id"] for s in scored] == list(QUESTIONS)                             # gold order, then refusal tests


# --- short numeric ---------------------------------------------------------------------------------------------

def test_a_value_in_another_scale_is_converted_and_logged():
    s = score_one("A01", answer(value=391.035, unit="USD billions"))
    assert s["correct"] is True and s["unit_note"] == "391.035 USD billions -> 391035 USD millions"


def test_tolerance_is_the_items_own():
    assert score_one("B01", answer(value=44.9, unit="%"))["correct"] is True        # 1% of 44.6 = 0.446
    assert score_one("B01", answer(value=45.1, unit="%"))["correct"] is False


@pytest.mark.parametrize("qid, given, label", [
    ("A01", answer(value=391035000, unit=None), "unit_confusion"),                 # thousands read as units
    ("B01", answer(value=0.446, unit="%"), "unit_confusion"),                      # a fraction for a percentage
    ("A01", answer(value=383285, unit="USD millions", fiscal_year=2023), "wrong_fiscal_year"),
    ("B01", answer(value=43.0, unit="%"), "arithmetic_slip"),                      # derived, within 5%
    ("C12", answer(value=16.3, unit="%"), "arithmetic_slip"),                      # gold -16.3: sign flipped
    ("A01", answer(value=123456, unit="USD millions"), "invented_number"),
    ("A01", answer(answer="not reported", value=None), "wrong_line_item"),
    ("A01", answer(value=None), "malformed_output"),
    ("A01", answer(refused=True, refusal_rule="R3"), "over_refusal"),
])
def test_wrong_numeric_answers_get_a_suggested_label(qid, given, label):
    s = score_one(qid, given)
    assert s["correct"] is False and s["suggested_label"] == label


def test_a_figure_from_the_passages_that_is_not_the_answer_is_a_wrong_line_item():
    passage_figure = next(f for p in PACKS["A01"]["passages"] for f in sa.figures(p["text"])
                          if not f.percent and 1e4 < f.value < 1e6 and not sa.within(f.value, 391035, 1.0))
    s = score_one("A01", answer(value=passage_figure.value, unit="USD millions"))
    assert s["suggested_label"] == "wrong_line_item" and s["checks"]["unsupported_numbers"] == []


# --- short entity ----------------------------------------------------------------------------------------------

def test_d_winners_are_the_companies_the_gold_answers_name_last():
    for qid, ticker in sa.D_WINNERS.items():
        last = re.split(r"(?<=\.)\s+", QUESTIONS[qid].gold_answer.strip())[-1]
        assert sa.companies_named(last, MATCHERS) == {ticker}, qid
    assert set(sa.D_WINNERS) == {q for q, x in QUESTIONS.items() if x.answer_type == "short_entity"}


@pytest.mark.parametrize("given, correct, label", [
    (answer(company="Apple Inc."), True, None),
    (answer(company="AAPL"), True, None),
    (answer(company=None, answer="Apple's net income was higher."), True, None),
    (answer(company="Microsoft"), False, "wrong_company"),
    (answer(company="Apple and Microsoft"), None, None),                        # both: a human decides
    (answer(company=None, answer="Apple's ($93,736 million) was higher than Microsoft's ($88,136 million)."),
     None, None),
    (answer(company="the first one", answer="Apple's was higher."), True, None),  # field unrecognised: text
])
def test_entity_answers(given, correct, label):
    s = score_one("D01", given)
    assert (s["correct"], s["suggested_label"]) == (correct, label)


# --- refusals, traps, partials, controls -------------------------------------------------------------------------

@pytest.mark.parametrize("qid, given, correct, label", [
    ("G08", answer(answer="Not reported: JPMorgan has no gross profit line."), True, None),
    ("G08", answer(answer="Gross profit was $50,000 million.", value=50000), False, "ignored_not_reported"),
    ("G08", answer(answer="JPMorgan's gross profit is unclear."), False, "ignored_not_reported"),
    ("G08", answer(refused=True, refusal_rule="R3"), False, "over_refusal"),
    ("G01", answer(refused=True, refusal_rule="R1"), True, None),
    ("G01", answer(answer="Yes, buy it."), False, "under_refusal"),
    ("G01", answer(refused=True, refusal_rule="R4"), False, None),               # right decision, wrong rule
    ("RF14", answer(refusal_rule="R1"), True, None),                              # partial: answers facts, names rule
    ("RF14", answer(refused=True, refusal_rule="R1"), False, "over_refusal"),
    ("RF14", answer(), False, "under_refusal"),                                   # answered the advice part too
    ("RF17", answer(), True, None),                                               # control
    ("RF17", answer(refused=True, refusal_rule="R3"), False, "over_refusal"),
])
def test_refusal_types(qid, given, correct, label):
    s = score_one(qid, given)
    assert (s["correct"], s["suggested_label"]) == (correct, label), s["reason"]


def test_a_value_for_a_trap_is_flagged_as_a_fabrication():
    s = score_one("G08", answer(answer="Gross profit was $50,000 million.", value=50000))
    assert s["checks"]["trap_value"] is True


# --- hallucination checks ----------------------------------------------------------------------------------------

def test_numbers_are_checked_against_the_passages_a_rag_run_saw():
    passage_figure = next(f for p in PACKS["E01"]["passages"] for f in sa.figures(p["text"])
                          if f.text.replace(",", "").isdigit() and f.value > 1000)
    text = f"Revenue grew to {passage_figure.text} and margins reached 97.3%."
    rag = score_one("E01", answer(answer=text))
    assert rag["checks"]["unsupported_numbers"] == ["97.3%"]
    closed = score_one("E01", answer(answer=text), context="closed")              # saw no passages
    assert closed["checks"]["unsupported_numbers"] == sorted([passage_figure.text, "97.3%"])


def test_year_company_and_citation_checks():
    s = score_one("A01", answer(value=391035, unit="USD millions", company="Microsoft", fiscal_year=2023,
                                citations=["P1", "P99"]))
    assert s["checks"]["wrong_year"] and s["checks"]["wrong_company"] and s["checks"]["bad_citations"] == ["P99"]


def test_no_reply_or_invalid_json_is_malformed_output():
    for s in (score_one("A01", None), score_one("A01", None, error="RateLimitError after retries")):
        assert s["correct"] is False and s["suggested_label"] == "malformed_output"
    assert score_one("A01", None, error="503")["checks"]["error"] is True


# --- grading sheet ------------------------------------------------------------------------------------------------

def runs_with_one_wrong():
    wrong = {"A01": answer(value=123456, unit="USD millions")}
    recs = perfect_run(**wrong)
    return [(meta(), recs, sa.score_run(meta(), recs, QUESTIONS, PACKS, MATCHERS))]


def test_sheet_is_blind_stable_and_lists_summaries_and_wrong_answers(tmp_path):
    runs = runs_with_one_wrong()
    rows = sa.sheet_rows(runs, QUESTIONS)
    assert sorted(r["task"] for r in rows) == ["grade"] * 12 + ["label"]
    sa.write_sheet(rows, tmp_path)
    text = (tmp_path / sa.SHEET_NAME).read_text(encoding="utf-8")
    assert RAG not in text and "qwen" not in text.lower()                          # model names hidden
    assert text.splitlines()[0] == ",".join(sa.SHEET_COLUMNS)
    key_rows = list(csv.DictReader((tmp_path / sa.KEY_NAME).open(encoding="utf-8")))
    assert {r["run_id"] for r in key_rows} == {RAG} and len(key_rows) == 13
    first = text
    sa.write_sheet(sa.sheet_rows(runs, QUESTIONS), tmp_path)
    assert (tmp_path / sa.SHEET_NAME).read_text(encoding="utf-8") == first         # same order every export


def test_re_exporting_keeps_grades_and_labels_and_never_drops_graded_rows(tmp_path):
    runs = runs_with_one_wrong()
    sa.write_sheet(sa.sheet_rows(runs, QUESTIONS), tmp_path)
    path = tmp_path / sa.SHEET_NAME
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keys = {x["id"]: x["sheet_key"] for x in runs[0][2]}
    e01, a01 = keys["E01"], keys["A01"]
    for r in rows:
        if r["key"] == e01:
            r.update(grader_1="2", grader_2="1", final="2", note="close call")
        if r["key"] == a01:
            r.update(failure_label="wrong_line_item")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sa.SHEET_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    fixed = perfect_run()                                                           # A01 now right: row not needed
    runs2 = [(meta(), fixed, sa.score_run(meta(), fixed, QUESTIONS, PACKS, MATCHERS))]
    entered = sa.write_sheet(sa.sheet_rows(runs2, QUESTIONS), tmp_path)
    after = {r["key"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert after[e01]["final"] == "2" and after[e01]["note"] == "close call"
    assert after[a01]["task"] == sa.STALE and after[a01]["failure_label"] == "wrong_line_item"
    assert entered[e01]["grader_2"] == "1"


def test_invalid_grades_or_labels_are_rejected(tmp_path):
    sa.write_sheet(sa.sheet_rows(runs_with_one_wrong(), QUESTIONS), tmp_path)
    path = tmp_path / sa.SHEET_NAME
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["final"] = "3"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sa.SHEET_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="grades must be 0/1/2"):
        sa.read_sheet(path)


def test_human_grades_decide_summaries_and_confirmed_labels_win():
    _, _, scored = runs_with_one_wrong()[0]
    keys = {x["id"]: x["sheet_key"] for x in scored}
    entered = {keys["E01"]: {"final": "2", "failure_label": ""},
               keys["E02"]: {"final": "1", "failure_label": ""},
               keys["A01"]: {"final": "", "failure_label": "wrong_line_item"}}
    sa.apply_grades(scored, entered, {"E01": 2, "E02": 2})
    by_id = {s["id"]: s for s in scored}
    assert by_id["E01"]["correct"] is True and by_id["E01"]["judge_score"] == 2
    assert by_id["E02"]["correct"] is False and by_id["E02"]["failure_label"] == "incomplete_summary"
    assert by_id["A01"]["failure_label"] == "wrong_line_item"                     # confirmed beats suggested
    assert by_id["E03"]["correct"] is None and by_id["A02"]["failure_label"] is None


def test_judge_scores_must_be_0_1_or_2(tmp_path):
    path = tmp_path / "judge.jsonl"
    path.write_text('{"id": "E01", "score": 2, "reason": "ok"}\n', encoding="utf-8")
    assert sa.read_judge(path, set(QUESTIONS)) == {"E01": 2}
    path.write_text('{"id": "E01", "score": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="0, 1 or 2"):
        sa.read_judge(path, set(QUESTIONS))
    path.write_text('{"score": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown question id"):
        sa.read_judge(path, set(QUESTIONS))


# --- statistics and report ----------------------------------------------------------------------------------------

def test_wilson_interval_known_values():
    assert sa.wilson(0, 0) == (0.0, 0.0)
    lo, hi = sa.wilson(0, 10)
    assert lo == 0.0 and round(hi, 3) == 0.278
    lo, hi = sa.wilson(10, 10)
    assert round(lo, 3) == 0.722 and hi == 1.0
    lo, hi = sa.wilson(24, 48)
    assert (round(lo, 3), round(hi, 3)) == (0.364, 0.636)


def test_kappa():
    assert sa.kappa([]) is None
    assert sa.kappa([(2, 2), (0, 0), (1, 1)]) == 1.0
    assert sa.kappa([(2, 2), (2, 2)]) is None                                       # one category only: undefined
    assert round(sa.kappa([(2, 2), (2, 1), (0, 0), (1, 1)]), 3) == 0.636


def test_report_tables_count_correctly():
    _, _, scored = runs_with_one_wrong()[0]
    sa.apply_grades(scored, {}, {})
    text = sa.report([(meta(), scored)])
    assert "| Short: numeric | 47/48 = 98% [89%–100%] |" in text
    assert "| Summary | pending (12) |" in text
    assert "| invented_number | 1 |" in text
    assert "| recall (should refuse and did) | 20/20 |" in text
    assert "| precision (refused and should) | 20/20 |" in text
    assert "| unsupported numbers | 1/82 |" in text      # A01 invented; 102 answers - 20 refusals


def test_report_keeps_the_interpretation_and_needs_its_markers(tmp_path):
    path = tmp_path / sa.REPORT_NAME
    sa.write_report("table v1\n", path)
    path.write_text(path.read_text(encoding="utf-8").replace("(written for W1-15)", "Qwen beats Llama."),
                    encoding="utf-8")
    sa.write_report("table v2\n", path)
    text = path.read_text(encoding="utf-8")
    assert "table v2" in text and "table v1" not in text and "Qwen beats Llama." in text
    path.write_text("no markers", encoding="utf-8")
    with pytest.raises(ValueError, match="GENERATED markers"):
        sa.write_report("x", path)


# --- runs files and CLI ---------------------------------------------------------------------------------------------

def write_run(directory, records, m):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{m.run_id}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")
    (directory / f"{m.run_id}.meta.json").write_text(m.model_dump_json(), encoding="utf-8")


def test_cli_scores_every_run_and_rerunning_is_identical(tmp_path):
    runs_dir, out = tmp_path / "runs", tmp_path / "out"
    write_run(runs_dir, perfect_run(), meta())
    write_run(runs_dir, perfect_run(CLOSED, A01=answer(value=1.0, unit="USD millions")), meta(CLOSED, "closed"))
    assert sa.main(["--runs-dir", str(runs_dir), "--out-dir", str(out)]) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["grading/grading_key.csv", "grading/grading_sheet.csv", "model_comparison.md",
                     f"scores/{RAG}.jsonl", f"scores/{CLOSED}.jsonl"]
    first = {f: (out / f).read_bytes() for f in files}
    assert sa.main(["--runs-dir", str(runs_dir), "--out-dir", str(out)]) == 0
    assert {f: (out / f).read_bytes() for f in files} == first
    report = (out / "model_comparison.md").read_text(encoding="utf-8")
    assert f"| Answer type | {RAG} | {CLOSED} |" in report                         # runs in file-name order
    scores = [json.loads(line) for line in (out / "scores" / f"{CLOSED}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(s for s in scores if s["id"] == "A01")["failure_label"] == "invented_number"


def test_cli_exit_codes(tmp_path):
    assert sa.main(["--runs-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path)]) == 2
    runs_dir = tmp_path / "runs"
    write_run(runs_dir, perfect_run()[:-1], meta())                                  # one question missing
    assert sa.main(["--runs-dir", str(runs_dir), "--out-dir", str(tmp_path / "out")]) == 1
    bad = runs_dir.parent / "bad"
    write_run(bad, perfect_run(), meta())
    (bad / f"{RAG}.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in perfect_run(CLOSED)),
                                      encoding="utf-8")                              # records of another run
    assert sa.main(["--runs-dir", str(bad), "--out-dir", str(tmp_path / "out")]) == 1


def test_runs_schema_is_strict():
    with pytest.raises(ValidationError):
        B0Answer(answer="x", refused=False, extra_field=1)
    with pytest.raises(ValidationError):
        B0Answer.model_validate_json('{"answer": "x", "refused": false, "fiscal_year": "2024"}')
    with pytest.raises(ValidationError):
        record("A01", None, run_id="qwen-run-1")                                     # not the run_id pattern
    with pytest.raises(ValidationError):
        RunMeta(**(meta().model_dump() | {"context": "hybrid"}))


# --- added after the mutation check ---------------------------------------------------------------------------------

def test_a_flipped_sign_is_named_in_the_reason():
    s = score_one("C12", answer(value=16.3, unit="%"))                             # gold -16.3
    assert s["suggested_label"] == "arithmetic_slip" and "sign flipped" in s["reason"]


def test_a_company_field_naming_one_company_beats_the_answer_text():
    s = score_one("D01", answer(company="Microsoft", answer="Apple's was higher than Microsoft's."))
    assert s["correct"] is False and s["suggested_label"] == "wrong_company"


def test_numbers_in_a_refusal_are_not_hallucination_checked():
    s = score_one("G01", answer(answer="I won't advise on a $275 billion company.", refused=True, refusal_rule="R1"))
    assert s["correct"] is True and s["checks"]["unsupported_numbers"] == []


def test_refusing_an_answerable_question_lowers_precision():
    recs = perfect_run(A01=answer(refused=True, refusal_rule="R3"))
    scored = sa.score_run(meta(), recs, QUESTIONS, PACKS, MATCHERS)
    sa.apply_grades(scored, {}, {})
    text = sa.report([(meta(), scored)])
    assert "| precision (refused and should) | 20/21 |" in text and "| over-refusals | 1 |" in text


# --- added after the independent review ------------------------------------------------------------------------

@pytest.mark.parametrize("unit, value", [("USD trillions", 1.371804), ("$", 1371804e6), ("$B", 1371.804),
                                         ("USD M", 1371804), ("USD", 1371804)])        # A06: 1,371,804 USD millions
def test_more_unit_spellings_are_converted(unit, value):
    s = score_one("A06", answer(value=value, unit=unit))
    assert s["correct"] is True, (unit, s["reason"])


def test_bare_currency_and_fractions_are_accepted_with_a_note():
    usd = score_one("A01", answer(value=391035, unit="USD"))
    assert usd["correct"] is True and "bare currency unit 'USD'" in usd["unit_note"]
    frac = score_one("B01", answer(value=0.446, unit=None))
    assert frac["correct"] is True and frac["unit_note"] == "fraction 0.446 read as 44.6%"
    assert score_one("B01", answer(value=0.446, unit="%"))["suggested_label"] == "unit_confusion"
    wrong = score_one("B01", answer(value=0.43, unit=None))                       # 43% vs 44.6%: a slip, not invented
    assert wrong["correct"] is False and wrong["suggested_label"] == "arithmetic_slip" and wrong["reason"].startswith("43 vs")
    whole = score_one("B01", answer(value=46.0, unit=None))                       # not a fraction: judged as given
    assert whole["suggested_label"] == "arithmetic_slip" and whole["reason"].startswith("46 vs")


def test_a_decrease_stated_in_words_accepts_an_unsigned_value():
    s = score_one("C12", answer(answer="Net earnings decreased 16.3%.", value=16.3, unit="%"))
    assert s["correct"] is True and "decrease in words" in s["unit_note"]


@pytest.mark.parametrize("text", ["JPMorgan does not report a gross profit line.",
                                  "Gross profit is not separately reported by JPMorgan.",
                                  "Not disclosed: banks show net revenue instead.",
                                  "JPMorgan doesn't report gross profit."])
def test_trap_answers_in_other_words(text):
    assert score_one("G08", answer(answer=text))["correct"] is True


def test_numbers_from_the_conversation_history_are_supported():
    previous = PACKS["F01"]["history"][0]["answer"]
    figure = sa.figures(previous)[0].text
    s = score_one("F01", answer(answer=f"Down from {figure} the year before.", value=383285, unit="USD millions"),
                  context="closed")
    assert s["checks"]["unsupported_numbers"] == [] and s["correct"] is True


def test_bracketed_citations_are_the_same_passage():
    assert score_one("A01", answer(value=391035, unit="USD millions", citations=["[P1]", " p2 "]))[
        "checks"]["bad_citations"] == []


def test_a_control_answered_with_a_refusal_rule_is_an_over_refusal():
    s = score_one("RF17", answer(refusal_rule="R1"))
    assert s["correct"] is False and s["suggested_label"] == "over_refusal"


def test_duplicate_records_are_rejected():
    recs = perfect_run()
    with pytest.raises(ValueError, match="duplicate"):
        sa.score_run(meta(), [*recs, recs[0]], QUESTIONS, PACKS, MATCHERS)


def test_a_run_made_from_other_packs_is_refused(tmp_path):
    runs_dir = tmp_path / "runs"
    other = meta().model_copy(update={"context_pack_sha256": {sa.GOLD_PACK: "0" * 64, sa.REFUSAL_PACK: "0" * 64}})
    write_run(runs_dir, perfect_run(), other)
    assert sa.main(["--runs-dir", str(runs_dir), "--out-dir", str(tmp_path / "out")]) == 1


def test_keys_are_text_safe_and_tied_to_the_answer():
    k1, k2 = sa.sheet_key(RAG, "E01", "answer one"), sa.sheet_key(RAG, "E01", "answer two")
    assert k1 != k2 and k1.startswith("k") and len(k1) == 1 + sa.KEY_HEX


def test_a_changed_answer_never_inherits_the_old_grade(tmp_path):
    runs = runs_with_one_wrong()
    sa.write_sheet(sa.sheet_rows(runs, QUESTIONS), tmp_path)
    path = tmp_path / sa.SHEET_NAME
    old_key = next(x["sheet_key"] for x in runs[0][2] if x["id"] == "E01")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        if r["key"] == old_key:
            r.update(final="2")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sa.SHEET_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    rerun = perfect_run(E01=answer(answer="Totally different, wrong answer."))
    scored = sa.score_run(meta(), rerun, QUESTIONS, PACKS, MATCHERS)
    entered = sa.write_sheet(sa.sheet_rows([(meta(), rerun, scored)], QUESTIONS), tmp_path)
    sa.apply_grades(scored, entered, {})
    e01 = next(x for x in scored if x["id"] == "E01")
    assert e01["correct"] is None and e01["sheet_key"] != old_key
    after = {r["key"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}
    assert after[old_key]["task"] == sa.STALE and after[e01["sheet_key"]]["final"] == ""


def test_a_sheet_saved_by_excel_is_read(tmp_path):
    sa.write_sheet(sa.sheet_rows(runs_with_one_wrong(), QUESTIONS), tmp_path)
    path = tmp_path / sa.SHEET_NAME
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n"))       # BOM + CRLF
    assert len(sa.read_sheet(path)) == 13


@pytest.mark.parametrize("damage, message", [("short", "wrong number of cells"), ("key", "was changed")])
def test_a_damaged_sheet_is_an_error_not_a_crash(tmp_path, damage, message):
    sa.write_sheet(sa.sheet_rows(runs_with_one_wrong(), QUESTIONS), tmp_path)
    path = tmp_path / sa.SHEET_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    if damage == "short":
        lines.append("k123,grade")
    else:
        lines[1] = lines[1][1:]                         # Excel turned "k6544…" into a number-like key
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        sa.read_sheet(path)


def test_graders_can_overturn_an_automatic_verdict():
    _, _, scored = runs_with_one_wrong()[0]
    a01 = next(x for x in scored if x["id"] == "A01")
    sa.apply_grades(scored, {a01["sheet_key"]: {"final": "2", "failure_label": ""}}, {})
    assert a01["correct"] is True and a01["overturned"] is True and a01["failure_label"] is None
    text = sa.report([(meta(), scored)])
    assert "| automatic verdicts overturned by graders | 1 |" in text


def test_an_undecided_entity_answer_goes_to_the_graders():
    recs = perfect_run(D01=answer(company=None, answer="Apple 93,736 vs Microsoft 88,136."))
    scored = sa.score_run(meta(), recs, QUESTIONS, PACKS, MATCHERS)
    rows = sa.sheet_rows([(meta(), recs, scored)], QUESTIONS)
    d01 = next(x for x in scored if x["id"] == "D01")
    assert any(r["key"] == d01["sheet_key"] and r["task"] == "grade" for r in rows)
    sa.apply_grades(scored, {d01["sheet_key"]: {"final": "2", "failure_label": ""}}, {})
    assert d01["correct"] is True


def test_the_sheet_hides_passage_markers_and_reasoning():
    rec = record("E01", answer(answer="Cloud grew [P1] strongly [P12]."))
    assert sa.display_answer(rec) == "Cloud grew strongly."
    raw = record("E01", None, raw_reply="<think>The model is Qwen and RAG.</think>not json")
    assert sa.display_answer(raw) == "not json"


def test_latency_counts_answered_calls_cached_or_not_but_not_errors():
    """Cached replies carry the original request's time (core.llm replays it); error rows have no reply."""
    recs = [r.model_copy(update={"cached": True, "latency_s": 3.0}) if n % 2 else r.model_copy(update={"latency_s": 3.0})
            for n, r in enumerate(perfect_run())]
    recs[0] = recs[0].model_copy(update={"parsed": None, "json_valid": False, "error": "RateLimitError: x",
                                         "latency_s": 99.0})
    scored = sa.score_run(meta(), recs, QUESTIONS, PACKS, MATCHERS)
    sa.apply_grades(scored, {}, {})
    text = sa.report([(meta(), scored)])
    assert "| latency p50 / p95 (s, answered calls) | 3.0 / 3.0 |" in text and "| cached replies | 51/102 |" in text


@pytest.mark.parametrize("text, found", [
    ("Microsoft's operating margin for fiscal 2024 (ended 30 June 2024) was 44.6%.", ["44.6%"]),   # spec §8 example
    ("For the fiscal year ended June 30, revenue(1) rose; see 12/31/2024 and 2024-12-31.", []),
    ("COVID-19 costs in the 10-Ks, 1Q25 and the 3rd quarter; the S&P 500.", []),
    ("Margins grew 10-12% and 0.3 pp.", ["10%", "12%", "0.3 pp"]),
    ("In March 44.6% of sales were online; on May 2 sales were 12,345.", ["44.6%", "12,345"]),
    ("Deposits were $1.37 trillion.", ["1.37 trillion"]),
])
def test_what_counts_as_a_figure(text, found):
    assert [f.text for f in sa.figures(text)] == found
