# Model comparison (W1-08 scorer)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| Short: numeric | 16/48 = 33% [22%–47%] | 6/48 = 12% [6%–25%] | 0/48 = 0% [0%–7%] |
| Short: entity | 5/10 = 50% [24%–76%] | 4/10 = 40% [17%–69%] | 5/10 = 50% [24%–76%] |
| Summary | 8/12 = 67% [39%–86%] | 5/12 = 42% [19%–68%] | 0/12 = 0% [0%–24%] |
| Refusal & not reported | 21/32 = 66% [48%–80%] | 7/32 = 22% [11%–39%] | 21/32 = 66% [48%–80%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| fact | 8/15 = 53% [30%–75%] | 5/15 = 33% [15%–58%] | 0/15 = 0% [0%–20%] |
| ratio | 7/15 = 47% [25%–70%] | 1/15 = 7% [1%–30%] | 0/15 = 0% [0%–20%] |
| period_comparison | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] |
| multi_turn | 1/6 = 17% [3%–56%] | 0/6 = 0% [0%–39%] | 0/6 = 0% [0%–39%] |
| multi_company | 5/10 = 50% [24%–76%] | 4/10 = 40% [17%–69%] | 5/10 = 50% [24%–76%] |
| qualitative | 8/12 = 67% [39%–86%] | 5/12 = 42% [19%–68%] | 0/12 = 0% [0%–24%] |
| refuse | 5/10 = 50% [24%–76%] | 1/10 = 10% [2%–40%] | 8/10 = 80% [49%–94%] |
| refusal tests: refusal | 9/13 = 69% [42%–87%] | 1/13 = 8% [1%–33%] | 7/13 = 54% [29%–77%] |
| refusal tests: partial | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] |
| refusal tests: control | 7/7 = 100% [65%–100%] | 5/7 = 71% [36%–92%] | 6/7 = 86% [49%–97%] |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| unsupported numbers | 20/74 | 15/66 | 45/70 |
| wrong fiscal year | 0/74 | 0/66 | 0/70 |
| wrong company | 0/74 | 0/66 | 0/70 |
| cited a passage it was not given (closed-book: any citation) | 0/74 | 0/66 | 0/70 |
| value given for a 'not reported' trap | 1/3 | 0/3 | 1/3 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| precision (refused and should) | 18/28 | 5/8 | 16/32 |
| recall (should refuse and did) | 18/20 | 5/20 | 16/20 |
| right rule among correct refusals | 14/18 | 1/5 | 13/16 |
| over-refusals | 10 | 5 | 20 |
| under-refusals | 3 | 3 | 6 |

### 5. Output validity, tokens, latency and human corrections

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| valid B0 JSON | 102/102 | 72/102 | 102/102 |
| JSON retries | 0 | 36 | 0 |
| no reply (errors) | 0 | 0 | 0 |
| prompt tokens (mean) | 3448 | 4253 | 699 |
| completion tokens (mean) | 88 | 82 | 76 |
| latency p50 / p95 (s, answered calls) | 2.4 / 3.6 | 42.5 / 66.3 | 1.8 / 2.3 |
| cached replies | 102/102 | 102/102 | 102/102 |
| automatic verdicts overturned by graders | 0 | 0 | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| unit_confusion | 0 | 3 | 0 |
| arithmetic_slip | 10 | 4 | 2 |
| invented_number | 8 | 14 | 28 |
| wrong_line_item | 7 | 8 | 3 |
| ignored_not_reported | 1 | 0 | 1 |
| wrong_company | 5 | 3 | 5 |
| over_refusal | 10 | 5 | 20 |
| under_refusal | 3 | 3 | 6 |
| malformed_output | 1 | 31 | 0 |
| incomplete_summary | 4 | 6 | 8 |
| wrong_refusal_rule | 3 | 3 | 3 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| graders' exact agreement (independent grades) | 7/12 | 7/12 | 6/8 |
| graders' Cohen's kappa | 0.17 | 0.39 | 0.54 |
| final grade vs judge: exact agreement | 12/12 | 9/11 | 4/8 |
| final grade vs judge: Cohen's kappa | 1.00 | 0.70 | 0.20 |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit)

| Gold dollar figures | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| all in the passages | 12/18 = 67% [44%–84%] | 4/18 = 22% [9%–45%] | – (closed-book) |
| not all in the passages | 3/28 = 11% [4%–27%] | 1/28 = 4% [1%–18%] | – (closed-book) |
<!-- GENERATED:END -->

## Interpretation

Model comparison v1 (W1-15, 13 Nov 2026). Every figure below is a cell of the generated tables above unless its
source is named; `tests/test_comparison_report.py` fails if a cited figure or example drifts from the data.

### Setup
Three runs answered the same 102 questions (80 gold-v1 items + 22 refusal tests): **M1** Qwen3-8B on the team
endpoint with retrieved passages, **M2** Llama-3.2-3B (local Ollama, 8k context) with the same passages, **M3**
Qwen3-8B closed-book. All three used the same frozen zero-shot prompt B0 (b0-v1), temperature 0, one retry on
invalid JSON and no tools: the baseline the Week 2 agent must beat. The passages are the W1-12 context pack (C3
hybrid retrieval with the gold item's company and year as filters, 4,000 tokens per question). Short answers are
scored automatically (W1-08); 32 answers (31 summaries and one "which company" answer the scorer could not decide)
were graded 0/1/2 by two team members independently and settled together (W1-14), with Qwen3-8B as an LLM judge
for comparison.

### Findings
1. **Retrieval often misses the figure, and most wrong numeric answers coincide with a miss.** Even with the
   correct company and year as filters, every gold dollar figure was in the passages (digit for digit) for only 18
   of the 46 numeric questions that have one (table 8). Qwen answered 12/18 of those correctly and 3/28 of the rest;
   Llama 4/18 and 1/28; 25 of Qwen's 32 wrong numeric answers are in the second group (M1 score file). The split is
   not a controlled experiment: the second group also holds most of the computed questions (21 of its 28 are ratio,
   period or follow-up questions), and a rounded figure counts as absent, which is how 3 of those 28 were still
   answered (e.g. A08 from "$62.6 billion"). The context pack's own measure (a passage from the gold-source section
   for 69 of 73 questions, `context_pack_gold.meta.json`) counts the right section, not the right line, so it
   overstated retrieval. Some failures follow directly:
   Qwen's "not in the passages" refusals of A09, A15, B01, B05 and C04 were correct about the passages (none holds
   the gold figures, though the scorer counts them as over-refusals), and A12 gave Coca-Cola's Q2 revenue
   ($12,535 million) for a Q3 question because only the Q2 table was retrieved.
2. **With passages Qwen is far better than closed-book.** Short numeric 16/48 and summaries 8/12, against 0/48 and
   0/12 closed-book. Closed-book answers also contain more numbers found in no evidence: 45/70 against 20/74
   (table 3; for runs with passages the evidence includes up to 4,000 tokens of passages, so their numbers are
   more likely to be "supported" by chance: the direction is clear, the size of the gap less so).
   invented_number is the closed-book run's most common failure (28), e.g. Apple's fiscal 2024 net sales as
   "$394.3 billion" (gold $391,035 million).
3. **Summaries are easier than exact numbers.** Qwen with passages is right on 8/12 summaries but 16/48 short
   numeric answers; Llama on 5/12 and 6/48. Computed figures are hardest: ratio 7/15 and **period comparison 0/12
   for all three runs**, including the three period questions whose inputs were all retrieved (C02, C08, C10), so
   calculation fails on its own too. Some period answers lacked an input (C01: "12%" NVIDIA revenue growth for a
   gold 114.2%; the fiscal 2025 revenue was not retrieved, fiscal 2024 only as "$60.9 billion"), and three are
   whole-number roundings of the right value that the tolerance rejects (C06 9 for 8.6 and C08 8 for 8.1 at 1%; C10
   0 for 0.3 at 5%). These three are among M1's 10 arithmetic_slip labels, its joint most common label with
   over_refusal, so real calculation errors are fewer than the label count.
4. **Qwen's refusals follow the passages.** Of M1's 10 over_refusal labels, the 5 answerable questions above lacked
   the gold figures in the pack, F03 too, two are "not reported" traps (G08, G10) where it said the figure is not
   provided but set refused=true, and two are partial-refusal tests (RF14, RF15). Its refusals of out-of-scope
   questions are good: precision 18/28, recall 18/20, the right rule in 14/18 (table 4). Closed-book it
   over-refuses 20 times, 10 of them as R3 (outside the corpus); most of them are fiscal-2025 figures it calls "not
   reported (yet)", pointing to its training cutoff rather than the prompt (M3 score file).
5. **The smaller model fails on format first.** Llama-3.2-3B returned valid B0 JSON for 72/102 questions after a
   retry: of the 30 invalid replies, 23 left the answer or the citations null, 4 wrote numbers as strings, 2 used
   thousands separators (repaired, still counted invalid) and 1 gave the answer as a number (M2 runs file). It refused only 5/20 of the questions
   that must be refused, naming the right rule for 1/5. It is closer to Qwen on entity comparisons (4/10 against
   5/10) and summaries (5/12 against 8/12) than on numbers (6/48 against 16/48). Its latency (42.5 s median on a
   laptop CPU) is not comparable with the endpoint's 2.4 s.
6. **Wrong answers stay inside the question's frame.** No answer put a company or fiscal year outside the question in
   its fields (0/74, 0/66, 0/70), but the wrong quarter (A12) and the wrong choice between the two companies asked
   about (D05: Pfizer named, although J&J's R&D was $17,232 million against Pfizer's $10,822 million) do occur.
7. **The LLM judge is lenient, and the humans disagree too.** Final human grades vs the judge: 12/12 (M1), 9/11
   (M2), 4/8 (M3); in all 6 disagreements the judge gave the higher grade (score files). The closed-book cell is
   small (8 answers), so this suggests rather than shows that the judge is weakest without evidence. The two graders
   agreed on 20/32 independent grades (Cohen's kappa 0.43 over all 32; per run 0.17-0.54) and on 163/177 failure
   labels both gave (W1-14 merge), so "partly correct" needs a sharper rubric. The judge can screen; humans decide.

### Failure taxonomy (table 6) with examples (`failure_examples.md`)
| Label | M1 | M2 | M3 | Example |
|---|---|---|---|---|
| invented_number | 8 | 14 | 28 | M3 A01: Apple's fiscal 2024 net sales "$394.3 billion" (gold $391,035 million) |
| over_refusal | 10 | 5 | 20 | M3 E01 refuses to summarise drivers the 10-K explains; M1 C04 (figures not retrieved, see finding 1) |
| arithmetic_slip | 10 | 4 | 2 | M1 C08: "8%" for Costco's 8.1% growth (whole-number rounding) |
| malformed_output | 1 | 31 | 0 | M2 C02: `"value": "24.1"` (a string), and 24.1% is not the gold 59.5% |
| wrong_line_item | 7 | 8 | 3 | M1 A12: the Q2 figure for a Q3 question (only the Q2 table retrieved) |
| wrong_company | 5 | 3 | 5 | M1 D05: Pfizer named; J&J $17,232 million vs Pfizer $10,822 million R&D |
| incomplete_summary | 4 | 6 | 8 | M1 E12: names FX, pricing and volume for P&G but not the +1% pricing / -1% FX behind the flat sales (graded 1) |
| under_refusal | 3 | 3 | 6 | M3 RF04: Tesla revenue "$92.4 billion", a company outside the corpus (R3) |
| wrong_refusal_rule | 3 | 3 | 3 | M1 RF10: refused as R1 where the test expects R6 |
| unit_confusion | 0 | 3 | 0 | M2 A08: Pfizer revenue as 62,600,000,000 "USD billions" (the figure in dollars, labelled billions) |
| ignored_not_reported | 1 | 0 | 1 | M3 G09: Amazon total liabilities "$135,550 million", a line Amazon does not report |

### Caveats
- Filters: the passages were retrieved with the gold item's company and year, an upper bound; question-derived
  filters (W1-05) will lower accuracy, on top of the chunk-level misses in finding 1.
- Small samples: 8-48 answers per cell (Wilson intervals in the tables); differences inside an interval are not
  established, e.g. entity 5/10 against 4/10, or the judge's 4/8.
- One frozen, untuned prompt. Review found two prompt properties that cost points and are kept for honesty: R2's
  wording also covers RF10, and R3 says "20 companies" without naming them (docs/week1_plan.txt §3).
- Closed-book "not reported" also means "I don't know", so M3 earns trap credit by not knowing.
- The judge is Qwen grading Qwen and Llama, with a rubric that has no "grounded" criterion.
- Scoring: a reported dollar figure must match at the precision printed; a computed percentage within the item's
  tolerance (so "8" for 8.1 is wrong); the refused flag is trusted (M1 RF12 set refused=true but wrote the poem);
  over_refusal also covers refusals that were right about the passages (finding 1).
- No verdict was overturned by the graders; the grading sheet records every grade and label.

### Next steps
- **Retrieval first (W1-05 and the Week 2 agent):** statement figures through `get_financial_fact` (XBRL, no
  retrieval needed), and retrieval aimed at statement tables for the period asked (quarter, not only year), then
  question-derived filters; re-measure table 8. Table 8 shows where the failures sit, not a measured effect: the
  re-measurement will show how much better retrieval actually helps.
- **Calculation through tools:** every growth rate and ratio through `calculate` (exact, half-up), so period
  comparisons are computed from retrieved or XBRL inputs, and the agent says it cannot compute when inputs are
  missing instead of guessing (C01).
- **Prompting baselines (W1-09) on the dev set:** S1 few-shot structured output (format and refusal rules), S2
  chain-of-thought with self-consistency (arithmetic), S3 decomposition (period and multi-company questions).
- **Critic checks (W1-07):** recompute numeric inputs, check the period (quarter and year) against the question and
  the cited passage, and when the passages lack a figure, fetch it with a tool before refusing.
- **Model:** keep Qwen3-8B as the agent's model (Llama-3.2-3B fails the output contract on 30/102 questions).
  The judge rubric gets a "grounded" criterion before the judge is used beyond screening.

### Traceability
Runs: `eval/results/model_runs/<run_id>.jsonl` with `.meta.json` (model, prompt and pack sha256, git commit);
scores: `eval/results/scores/` (per question, including `gold_figures_in_pack`); grades and labels:
`eval/results/grading/grading_sheet.csv` (key -> run in `grading_key.csv`); judge: `grading/judge_<run_id>.jsonl`.
Regenerate the tables with `python eval/score_answers.py` (no network; replies are cached, so a re-run of the models
is free and identical). Decision record: ADR-019 in `docs/decisions.md`.
