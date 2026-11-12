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
<!-- GENERATED:END -->

## Interpretation

Model comparison v1 (W1-15, 13 Nov 2026). Every figure below is a cell of the generated tables above, unless its
source is named; `tests/test_comparison_report.py` fails if a cited fraction disappears from them.

### Setup
Three runs answered the same 102 questions (80 gold-v1 items + 22 refusal tests): **M1** Qwen3-8B on the team
endpoint with retrieved passages, **M2** Llama-3.2-3B (local Ollama, 8k context) with the same passages, **M3**
Qwen3-8B closed-book. All three used the same frozen zero-shot prompt B0 (b0-v1), temperature 0, one retry on
invalid JSON and no tools: this is the baseline the Week 2 agent must beat. The passages are the W1-12 context
pack (C3 hybrid retrieval, oracle company/year filters, 4,000 tokens per question; a passage from the gold source
was in the pack for 69 of the 73 questions that have gold sources, `context_pack_gold.meta.json`). Short answers
are scored automatically (W1-08); the 32 summary-style answers were graded 0/1/2 by two team members
independently and settled together (W1-14), with Qwen3-8B as an LLM judge for comparison.

### Findings
1. **Retrieval is the largest lever.** With passages Qwen gets 16/48 short numeric answers and 8/12 summaries
   right; closed-book it gets 0/48 and 0/12. Closed-book answers also invent: 45/70 answers contain a number found
   in no evidence, against 20/74 with passages (table 3), and invented_number is its most common failure (28).
2. **Exact numbers are much harder than summaries.** Qwen with passages is right on 8/12 summaries but 16/48 short
   numeric answers; Llama on 5/12 and 6/48. Within short answers, figures copied from a statement do best (fact
   8/15) and computed ones worst: ratio 7/15 and **period comparison 0/12 for all three models**. Some misses are
   large (M1 C01: NVIDIA revenue growth 12% for a gold 114.2%), some are whole-number roundings of the right
   value (C06 9 for 8.6, C08 8 for 8.1, C10 0 for 0.3) that the 1% tolerance rejects. arithmetic_slip is M1's
   most common label (10).
3. **Qwen refuses answerable questions.** M1 has 10 over-refusals (table 4); 7 of them cite R4 ("not in the
   passages"), and for the 5 answerable ones among them (A09, A15, B01, B05, C04) a passage from the gold-source
   section was in the pack (M1 runs file and context pack). Example C04: it would not compute JPMorgan's net
   income change although both figures were given. Closed-book it over-refuses 20 times, 10 of them as R3 (out
   of scope). Its refusals are otherwise good: precision 18/28 and recall 18/20, with the right rule in 14/18.
4. **The smaller model fails on format first.** Llama-3.2-3B returned valid B0 JSON for 72/102 questions even
   after a retry (malformed_output 31: null answers and citations, numbers as quoted strings), refused only 5/20
   of the questions that must be refused, and named the right rule for 1/5. It is closer to Qwen on entity
   comparisons (4/10 against 5/10) and summaries (5/12 against 8/12) than on numbers (6/48 against 16/48), so a
   3B model would need a stricter output step and tools for figures. Its latency (42.5 s median on a laptop CPU) is not
   comparable with the endpoint's 2.4 s.
5. **Wrong periods and line items look plausible.** M1 A12 gave Coca-Cola's Q2 revenue ($12,535 million, a real
   10-Q figure) for a Q3 question; it is 0.6% from the gold $12,455 million, so only an exact-figure rule catches
   it (wrong_line_item: M1 7, M2 8). No answer in any run named the wrong fiscal year or company in its fields
   (0/74, 0/66, 0/70): errors are in the figure, not the frame.
6. **The LLM judge agrees with the humans only where the answer had evidence.** Final human grades vs the judge:
   12/12 (M1) and 9/11 (M2), but 4/8 (M3). The two graders agreed with each other on 20/32 independent grades
   (Cohen's kappa 0.43 over all 32, W1-14 merge; per run 0.17-0.54) and on 163/177 failure labels both gave
   (W1-14 merge), so "partly correct" needs a sharper rubric. The judge can screen grounded answers; humans
   still decide.

### Failure taxonomy (table 6) with examples (`failure_examples.md`)
| Label | M1 | M2 | M3 | Example |
|---|---|---|---|---|
| invented_number | 8 | 14 | 28 | M3 G09: "$135,550 million" total liabilities, a line Amazon does not report |
| over_refusal | 10 | 5 | 20 | M1 C04: "not reported in the provided passages" with both figures in them; M3 E01 refuses a summary |
| arithmetic_slip | 10 | 4 | 2 | M1 C01: 12% growth for a gold 114.2% |
| malformed_output | 1 | 31 | 0 | M2 C02: `"value": "24.1"` (a string), and 24.1% is not the gold 59.5% |
| wrong_line_item | 7 | 8 | 3 | M1 A12: the Q2 figure for a Q3 question |
| wrong_company | 5 | 3 | 5 | M1 D05: Pfizer for J&J ($17,232 million vs $10,822 million R&D) |
| incomplete_summary | 4 | 6 | 8 | M1 E12: names FX, pricing and volume for P&G but not the +1% pricing / -1% FX behind the flat sales (graded 1) |
| under_refusal | 3 | 3 | 6 | answers a question that must be refused (e.g. advice, forecasts) |
| wrong_refusal_rule | 3 | 3 | 3 | refuses correctly with the wrong rule (RF10/RF11 as R1 instead of R6) |
| unit_confusion | 0 | 3 | 0 | the right figure in the wrong scale |
| ignored_not_reported | 1 | 0 | 1 | a value for a figure the filing does not report (trap questions) |

### Caveats
- Oracle filters: the passages were retrieved with the gold item's company and year, so these are upper bounds;
  question-derived filters (W1-05) will lower them.
- Small samples: 10-48 questions per cell (Wilson intervals in the tables); differences inside an interval are not
  established, e.g. entity 5/10 vs 4/10.
- One frozen prompt, not tuned. Review found two prompt properties that cost points and are kept for honesty: R2's
  wording also covers RF10, and R3 says "20 companies" without naming them (docs/week1_plan.txt §3).
- Closed-book "not reported" also means "I don't know", so M3 earns trap credit by not knowing.
- The judge is Qwen grading Qwen and Llama, with a rubric that has no "grounded" criterion.
- Scoring: a reported dollar figure must match at the precision printed; a computed percentage within 1% (so
  "8" for 8.1 is wrong); the refused flag is trusted (M1 RF12 set refused=true but wrote the poem).
- No verdict was overturned by the graders; the grading sheet records every grade and label.

### Next steps
- **Tools (Week 2 agent):** every growth rate and ratio through `calculate` (exact, half-up), statement figures
  through `get_financial_fact` with the fiscal period checked: aimed at period_comparison 0/12, arithmetic_slip
  and the A12-type wrong period.
- **Prompting baselines (W1-09) on the dev set:** S1 few-shot structured output (format and refusal rules), S2
  chain-of-thought with self-consistency (arithmetic), S3 decomposition (period and multi-company questions).
- **Critic checks (W1-07):** recompute numeric inputs, check the period and fiscal year against the passage, and
  flag an R4 refusal when the passages contain the requested line.
- **Retrieval (W1-05):** question-derived filters instead of oracle ones; re-measure M1.
- **Model:** keep Qwen3-8B as the agent's model (Llama-3.2-3B fails the output contract on 30/102 questions).
  Judge prompts get a "grounded" criterion before the judge is used beyond screening.

### Traceability
Runs: `eval/results/model_runs/<run_id>.jsonl` with `.meta.json` (model, prompt and pack sha256, git commit);
scores: `eval/results/scores/`; grades and labels: `eval/results/grading/grading_sheet.csv` (key ->
run in `grading_key.csv`); judge: `grading/judge_<run_id>.jsonl`. Regenerate the tables with
`python eval/score_answers.py` (no network; replies are cached, so a re-run of the models is free and identical).
Decision record: ADR-019 in `docs/decisions.md`.
