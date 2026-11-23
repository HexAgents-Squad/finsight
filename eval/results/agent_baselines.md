# The agent (A1) next to B0 and S1 / S2 / S3 on Qwen3-8B (W2-11)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| Short: numeric | 16/48 = 33% [22%–47%] | 17/48 = 35% [23%–50%] | 20/48 = 42% [29%–56%] | 21/48 = 44% [31%–58%] | 42/48 = 88% [75%–94%] | 44/48 = 92% [80%–97%] |
| Short: entity | 5/10 = 50% [24%–76%] | 5/10 = 50% [24%–76%] | 6/10 = 60% [31%–83%] | 8/10 = 80% [49%–94%] | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] |
| Summary | 8/12 = 67% [39%–86%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] | 3/12 = 25% [9%–53%] | 5/12 = 42% [19%–68%] | 9/12 = 75% [47%–91%] |
| Refusal & not reported | 21/32 = 66% [48%–80%] | 31/32 = 97% [84%–99%] | 22/32 = 69% [51%–82%] | 19/32 = 59% [42%–74%] | 29/32 = 91% [76%–97%] | 32/32 = 100% [89%–100%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| fact | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 14/15 = 93% [70%–99%] | 15/15 = 100% [80%–100%] |
| ratio | 7/15 = 47% [25%–70%] | 8/15 = 53% [30%–75%] | 6/15 = 40% [20%–64%] | 8/15 = 53% [30%–75%] | 11/15 = 73% [48%–89%] | 13/15 = 87% [62%–96%] |
| period_comparison | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] | 11/12 = 92% [65%–99%] | 11/12 = 92% [65%–99%] |
| multi_turn | 1/6 = 17% [3%–56%] | 1/6 = 17% [3%–56%] | 2/6 = 33% [10%–70%] | 2/6 = 33% [10%–70%] | 6/6 = 100% [61%–100%] | 5/6 = 83% [44%–97%] |
| multi_company | 5/10 = 50% [24%–76%] | 5/10 = 50% [24%–76%] | 6/10 = 60% [31%–83%] | 8/10 = 80% [49%–94%] | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] |
| qualitative | 8/12 = 67% [39%–86%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] | 3/12 = 25% [9%–53%] | 5/12 = 42% [19%–68%] | 9/12 = 75% [47%–91%] |
| refuse | 5/10 = 50% [24%–76%] | 10/10 = 100% [72%–100%] | 8/10 = 80% [49%–94%] | 7/10 = 70% [40%–89%] | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] |
| refusal tests: refusal | 9/13 = 69% [42%–87%] | 13/13 = 100% [77%–100%] | 7/13 = 54% [29%–77%] | 5/13 = 38% [18%–64%] | 11/13 = 85% [58%–96%] | 13/13 = 100% [77%–100%] |
| refusal tests: partial | 0/2 = 0% [0%–66%] | 1/2 = 50% [9%–91%] | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] | 1/2 = 50% [9%–91%] | 2/2 = 100% [34%–100%] |
| refusal tests: control | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| unsupported numbers | 20/74 | 15/72 | 4/85 | 19/88 | 0/81 | 0/82 |
| unsupported numbers, no calculation result counted as evidence (B0's rule; the same count for a pack run) | 20/74 | 15/72 | 4/85 | 19/88 | 4/81 | 3/82 |
| wrong fiscal year | 0/74 | 0/72 | 0/85 | 0/88 | 0/81 | 0/82 |
| wrong company | 0/74 | 0/72 | 0/85 | 0/88 | 0/81 | 0/82 |
| cited a passage it was not given (closed-book: any citation) | 0/74 | 0/72 | 0/85 | 0/88 | 0/81 | 0/82 |
| value given for a 'not reported' trap | 1/3 | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| precision (refused and should) | 18/28 | 20/30 | 14/17 | 13/14 | 20/21 | 20/20 |
| recall (should refuse and did) | 18/20 | 20/20 | 14/20 | 13/20 | 20/20 | 20/20 |
| right rule among correct refusals | 14/18 | 20/20 | 12/14 | 10/13 | 18/20 | 20/20 |
| over-refusals | 10 | 5 | 9 | 4 | 2 | 0 |
| under-refusals | 3 | 0 | 3 | 4 | 0 | 0 |

### 5. Output validity, tokens, latency and human corrections

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| valid B0 JSON | 102/102 | 102/102 | 102/102 | 102/102 | 102/102 | 102/102 |
| JSON retries | 0 | 0 | 0 | 0 | 0 | 0 |
| no reply (errors) | 0 | 0 | 0 | 0 | 0 | 0 |
| prompt tokens (mean) | 3448 | 4465 | 18912 | 7073 | 2781 | 3197 |
| completion tokens (mean) | 88 | 81 | 972 | 324 | 219 | 223 |
| latency p50 / p95 (s, answered calls) | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 | 5.0 / 12.4 | 5.5 / 11.2 |
| cached replies | 102/102 | 0/102 | 101/102 | 0/102 | 0/102 | 0/102 |
| automatic verdicts overturned by graders | 0 | 4 | 0 | 1 | 0 | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| wrong_fiscal_year | 0 | 0 | 0 | 1 | 0 | 0 |
| unit_confusion | 0 | 1 | 2 | 2 | 0 | 0 |
| arithmetic_slip | 10 | 11 | 5 | 6 | 1 | 0 |
| invented_number | 8 | 8 | 2 | 8 | 0 | 0 |
| wrong_line_item | 7 | 7 | 14 | 7 | 4 | 3 |
| ignored_not_reported | 1 | 0 | 0 | 1 | 0 | 0 |
| wrong_company | 5 | 5 | 2 | 2 | 0 | 0 |
| over_refusal | 10 | 5 | 9 | 4 | 2 | 0 |
| under_refusal | 3 | 0 | 3 | 4 | 0 | 0 |
| malformed_output | 1 | 0 | 0 | 0 | 0 | 1 |
| incomplete_summary | 4 | 8 | 8 | 9 | 7 | 3 |
| wrong_refusal_rule | 3 | 0 | 6 | 7 | 2 | 0 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| graders' exact agreement (independent grades) | 7/12 | 11/12 | 14/14 | 10/12 | 21/22 | 13/13 |
| graders' Cohen's kappa | 0.17 | 0.83 | 1.00 | 0.67 | 0.91 | 1.00 |
| final grade vs judge: exact agreement | 12/12 | 7/12 | 5/12 | 5/12 | 6/12 | – |
| final grade vs judge: Cohen's kappa | 1.00 | 0.29 | 0.12 | 0.10 | 0.29 | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit) (an agent run: in what its own tools returned)

| Gold dollar figures | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| all in the passages | 12/18 = 67% [44%–84%] | 13/18 = 72% [49%–88%] | 17/18 = 94% [74%–99%] | 17/18 = 94% [74%–99%] | 40/40 = 100% [91%–100%] | 42/44 = 95% [85%–99%] |
| not all in the passages | 3/28 = 11% [4%–27%] | 3/28 = 11% [4%–27%] | 1/28 = 4% [1%–18%] | 2/28 = 7% [2%–23%] | 0/6 = 0% [0%–39%] | 0/2 = 0% [0%–66%] |

### 9. Cost per question and self-consistency

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|---|
| model calls per question (mean, JSON retries included) | 1.0 | 1.0 | 5.0 | 2.0 | 1.8 | 1.9 |
| tokens per question (mean, prompt + completion) | 3536 | 4546 | 19883 | 7397 | 3000 | 3419 |
| seconds per question p50 / p95 | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 | 5.0 / 12.4 | 5.5 / 11.2 |
| S2: all valid samples agreed | – | – | 62/102 | – | – | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – | – | 33/102 | – | – | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – | – | 7/102 | – | – | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | 8 / 4 / 0 | 8 / 4 / 0 | 8 / 3 / 1 | 9 / 3 / 0 | 3 / 9 / 0 | – |

### 10. Paired with M1_qwen3-8b_rag_think-off_b0-v1 (B0): questions right only with the run / only with B0, two-sided sign test

| Answers | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| Short: numeric | 3 / 2 (p = 1.00, n = 48) | 6 / 2 (p = 0.29, n = 48) | 7 / 2 (p = 0.18, n = 48) | 29 / 3 (p < 0.01, n = 48) | 29 / 1 (p < 0.01, n = 48) |
| Short: entity | 1 / 1 (p = 1.00, n = 10) | 2 / 1 (p = 1.00, n = 10) | 3 / 0 (p = 0.25, n = 10) | 5 / 0 (p = 0.06, n = 10) | 5 / 0 (p = 0.06, n = 10) |
| Summary | 0 / 4 (p = 0.12, n = 12) | 0 / 5 (p = 0.06, n = 12) | 0 / 5 (p = 0.06, n = 12) | 2 / 5 (p = 0.45, n = 12) | 3 / 2 (p = 1.00, n = 12) |
| Refusal & not reported | 10 / 0 (p < 0.01, n = 32) | 4 / 3 (p = 1.00, n = 32) | 2 / 4 (p = 0.69, n = 32) | 8 / 0 (p = 0.01, n = 32) | 11 / 0 (p < 0.01, n = 32) |
| All decided answers | 14 / 7 (p = 0.19, n = 102) | 12 / 11 (p = 1.00, n = 102) | 12 / 11 (p = 1.00, n = 102) | 44 / 8 (p < 0.01, n = 102) | 48 / 3 (p < 0.01, n = 102) |

### 11. The agent's own checks and cost

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|
| verified by its critic | 102/102 | 98/102 |
| correct among the verified (decided) | 86/102 | 95/98 |
| correct among the not verified (decided) | – | 0/4 |
| tool calls per question (mean); questions with none | 1.6; 21 | 1.7; 20 |
| tool calls that returned an error | 4/166 | 5/170 |
| stopped by a limit (time, steps, tokens) | 0 | 0 |
| live requests / cache hits (chat replies) | 187 / 0 | 194 / 0 |
| short numeric answers with several values and no calculation (no single value to score) | 0 | 1 |

### 13. Against A1_qwen3-8b_agent_think-off_a-v1 (A1): correct answers (run vs A1), right only with the run / only with A1 (sign test), and whether the type is worse; a version is final only if no type is worse

| Answer type | A5_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| Short: numeric | 44 vs 42 of 48; 3 / 1 (p = 0.62); not worse |
| Short: entity | 10 vs 10 of 10; 0 / 0 (p = 1.00); not worse |
| Summary | 9 vs 5 of 12; 4 / 0 (p = 0.12); not worse |
| Refusal & not reported | 32 vs 29 of 32; 3 / 0 (p = 0.25); not worse |
| No answer type worse than A1 | yes |
<!-- GENERATED:END -->

## Interpretation

W2-11, 21 Nov 2026 (ADR-034). A1 is the agent graph (planner, executor, composer, critic; ADR-024 to ADR-031) on
the same 80 gold questions and 22 refusal tests as B0 and S1-S3. It was run once by `eval/run_agent.py` on the
shared endpoint (Qwen3-8B, thinking off, temperature 0): 187 live requests, no errors, no question stopped by a
limit. After the PR reviews fixed how runs are recorded and scored, the runs file was re-derived from the reply cache
(`--replay-of`, 0 requests). Every answer and tool call is identical; the timings and cost above are the live run's,
and the meta's `replay` says when and with which code. The live run's meta names main (c9a2d40), because the
runner it used was committed right after it (fb1e63c); the replay names the commit of the code that wrote this
file.

Its prompts are the planner and composer prompts tuned on the development set and pinned by sha256 before this
run; the gold set was not read by any of them. Unlike B0 and S1-S3 it had no context pack. It chose its own tools
(XBRL facts, the calculator, the filing search) and saw only each question and, for a follow-up, the earlier
questions, never an earlier answer. Same scorer, same tolerances, paired on the same questions.

**All 102 answers are decided** (the W2-12 grading round graded the 10 multi-company answers and the 12
summaries, and confirmed the failure labels).
- Short numeric: 42/48 against B0's 16/48 (29 right only with A1, 3 only with B0; sign test p ≈ 3×10⁻⁶).
  - By category: fact 14/15, ratio 11/15, period comparison 11/12 (B0 0/12, the best strategy 4/12), follow-ups 6/6.
- Multi-company: 10/10 against B0's 5/10 (5 / 0, p ≈ 0.06).
- Refusal and not reported: 29/32 against 21/32 (8 / 0, p ≈ 0.008).
  - Recall 20/20, precision 20/21, the right rule on 18 of 20, no under-refusal, the three "not reported" traps right.
- Summaries: 5/12 against B0's 8/12 (2 / 5, p ≈ 0.45), the one answer type where A1 is not ahead (see below).
- All answers: 44 / 8 against B0 (p ≈ 4×10⁻⁷, n = 102).

**Why.** Table 8 is consistent with the gain coming from what the agent fetches. On 40 of the 46 numeric questions
with gold dollar figures, its own tools returned every gold figure (an XBRL fact is in the gold answers' millions),
and it got all 40 right; on the other 6 it got none. For A1 this split is close to circular: fetching the right line
item both puts the gold figure in its evidence and makes the answer right. So it shows where its errors lie, not by
itself why it beats B0. For B0 the variable is the fixed pack: 28 of the 46 packs lacked a gold figure, and B0 got 3
of those right. The bottleneck is now which figure the agent fetches, not how it reads or computes it.

**Grounding, and how it is measured.** Table 3 counts unsupported numbers under two rules for the agent:
- **The agent rule (first row):** its evidence is its passages and fetched figures (facts, period rows), plus a
  calculator result only when every input of that calculation is a fetched figure at its own precision. On these
  terms no non-refused answer has an unsupported number (0/81), and every citation names a source a tool returned.
- **B0's rule (second row, the like-for-like comparison):** no calculation result counts as evidence, so a derived
  number is supported only by the gold answer, as for B0. Here A1 has 4 of 81 (B04, B14, C08: wrong derived values;
  D05: a difference) against B0's 20 of 74.

Either way the check asks whether a figure is grounded, not whether it is the right line item: A06, B04, B14, B15
and C08 below are grounded figures of the wrong metric. The scorer suggests a wrong line item (fetched inputs) or
an invented number (otherwise) for a wrong calculated value, since the calculator's arithmetic is exact; the
graders' labels (table 6) agree on A06, B04, B14 and B15, and label C08 an arithmetic slip and B08 an
over-refusal. B0's labels are unchanged.

**The 9 wrong decided answers** (plus 7 summaries graded below 2, under Summaries). Six come from one gap: the
facts tool has no metric for the line item asked.
- **Substituted and named as the line item asked** (four): the planner fetched a nearby metric, and the answer
  presents the result as the one asked.
  - B04: operating cash flow (−42,012) for noninterest expense, giving "noninterest expense … 23.7%".
  - B15: total assets and liabilities for current ones, giving "current ratio … 1.69".
  - B14: company-wide operating income for the AWS segment's.
  - C08: total revenue, which includes membership fees, for net sales (8.2% against 8.1%).
- **Substituted openly** (one): A06 was asked for total deposits and answered with total assets, naming them.
- **Missed** (one): B08 answered that UnitedHealth's medical cost ratio is not reported separately; the filing
  reports it (85.5%).

The other three are refusals:
- RF14: a partial refusal refused whole.
- RF10 and RF13: the right decision under the wrong rule (R2 for R6, R5 for R7).

**The critic.** It verified every answer: 86 of the 102 are right, so it passed 16 wrong ones (the 9 above and
7 summaries graded 0 or 1), and it never sent a draft back. The composer does fix contract problems before the critic sees a draft (node
notes on E01, E05, E06 and RF20), but that is not why the 9 got through: they are metric misses its checks do not
see (nor does it judge a summary's substance). Its metric check, `check_metric_matches_question` (agent/critic.py), compares the metrics the question names
(`named_metrics`) with those the draft fetched:
- it finds no named metric in A06 (deposits), B14 (a segment) or B15 (current assets and liabilities), so it checks
  nothing there;
- for a question that names an operation (a ratio, a percentage), it passes as soon as one named metric was
  fetched. B04 and B14 fetched revenue as the denominator, so they pass whatever their numerator was;
- it reads C08's "net sales" as revenue, which is the metric fetched;
- its list of line items that must come from a search rather than an XBRL fact (`SEARCH_ONLY`) has three entries,
  all stock-based compensation, and it rejects every fact row of such a draft, a legitimate denominator included.

**Summaries.** The graders give A1's 12 summaries 5 twos (E04, E08, E09, E11, E12), 3 ones (E02, E06, E10) and 4
zeros (E01, E03, E05, E07): 5/12 against B0's 8/12. The four zeros answer a neighbouring question from the passages
the agent found, e.g. E01 explains operating income rather than the revenue growth asked about. The two graders
agree on 21 of A1's 22 graded rows (kappa 0.91). The LLM judge gave 3 twos and 9 ones and agrees with the final
grades on 6 of 12 (kappa 0.29): higher on 4, lower on 2. A1's summaries are longer and rest on passages it found
itself, rather than on the gold answer's section; the judge's reasons are of two kinds, additions (factors the
reference does not mention) and substance (drivers the reference names, missed). The summary is A1's weakest
answer type, and the agent's search for the section that answers "what drove ..." is the place to improve it.

The handout showed the graders what the agent retrieved, with each figure's company, year and unit. That section
told them one of the answers searched for itself, so the round was not fully blind to which rows are the agent's
(GRADING.md says so).

**Cost.** 1.8 model calls and 3.0k tokens per question (B0: 1 call, 3.5k), 5.0 s median and 12.4 s at p95 (the
limit is 60 s). The facts tool was called 102 times, the calculator 34, the search 30; 4 of 166 calls returned an
error (`not_reported`, on G08-G10 and RF21), which the agent handled. The request counts are chat requests; the 30
searches also embedded their queries through the search tool's own embedder, not counted here.

**Limits of this comparison.**
- One run at temperature 0.
- B0 and S1-S3 were given oracle-filtered packs (W1-12), and A1 found its own sources, so the comparison is between
  systems, not prompts.
- Part of the refusal gain is the planner prompt, not the architecture: it has worked refusal examples (W2-02,
  tuned on the development set) shaped like the refusal tests' kinds, for example a fact followed by "should I
  buy", while B0's zero-shot prompt has none. S1, whose few-shot examples include a refusal, reached 31/32 once
  the graders overturned four of its automatic verdicts (W2-12).
- Grounded is not correct: see the check above.

### Week 3 ablations (plan)

Each ablation turns off one component on the A1 configuration, unchanged otherwise:
- same questions, scorer, tolerances and seed;
- scored by `eval/score_answers.py` and paired with A1 by the sign test.

They need a switch in `eval/run_agent.py` (`--ablation`, run ids A2-A4) with offline tests. Run them one at a time
on the shared key, through the reply cache: a call whose prompt is unchanged replays for free, so each costs only
the calls it changes.

| Run | Switched off | How | What it measures | Expected from A1 | New requests (est.) |
|---|---|---|---|---|---|
| A2 | the calculator | `calculate` removed from the executor's tools and the planner's tool list | whether the agent can still answer a derived question: the composer drops any figure no evidence prints (node notes on B03, B04, B06, B13), so it cannot compute one itself | most of the 26 ratio and period-comparison answers that used the calculator (of 27) lose their value ("no value"), unless the composer's guard is relaxed too, which would be a second change and its own run | ~140 (every planner call, plus the composer on the 34 calculations) |
| A3 | the critic | a new path in the graph: the composer's first draft becomes the answer, with no check (verified stays false; today a passing verdict sets it, a failing one sends the draft back once) | what the critic changes | no answer changes: it rejected no A1 draft | ~0 (planner and composer calls replay) |
| A4 | statements first | `search_filings` without `StatementFirstIndex` (plain hybrid, ADR-017) | what the Item 8 preference adds | changes only on the 30 questions that searched, mostly summaries and traps | ~30 (composers after a search) |

A3's expected null result is itself the finding: on these questions the critic corrects nothing. Week 3 should fix
the existing metric check in three ways, each of which matters:
- **Vocabulary:** teach `named_metrics` deposits, noninterest expense, current assets and liabilities, segment
  figures, and net sales as distinct from total revenue. On its own this catches A06, B15 and C08, not B04 or B14.
- **The overlap rule:** for an operation, every fetched metric the draft uses must be named by the question or be
  the denominator the operation implies, not just one of them. This is what catches B04 and B14, whose revenue
  denominator now lets any numerator through.
- **`SEARCH_ONLY`:** reject only the substituted metric, not every fact row, so a searched numerator over a revenue
  fact (a correct draft) passes, and add the line items no XBRL metric covers.

The facts tool gaps behind those answers are a tools fix. Order:
1. The ablations, on the frozen A1 configuration, so each switch is measured against the same baseline.
2. The critic and metric changes, as a new version (A5) tuned on the development set first, each with offline tests
   on these nine answers' shapes.

### W3-08: version A5 on gold (ADR-037)

**One gold run, pinned.** `A5_qwen3-8b_agent_think-off_a-v1` is main after W3-03 to W3-07, run once at commit
e731597 after the development set v2 go / no-go (go): 194 live requests, no error, nothing changed after it. The
failure classes behind its fixes were found on A1's gold run (above); every fix is a class-level change tuned on the
development set v2, never on gold.

**Decision: A5 is final.** Section 13: no answer type is worse than A1. Graded by Chandra Shekhar and Amresh (13 of
13 grades agreeing, kappa 1.00); 70 of the 102 answers are word for word A1's and keep A1's grades.
- Short numeric 44/48 (A1 42/48): A06, B04, B15 now right (the facts tool's new metrics, W3-04); F02 now wrong, the
  follow-up "What was it in fiscal 2025?" after Microsoft's fiscal 2024 operating margin, answered with the operating
  income.
- Short entity 10/10 (A1 10/10).
- Summaries 9/12 graded 2 and 19 of 24 points (A1 5/12 and 13; B0 8/12 and 20). Better: E02, E03, E05, E06, E07.
  Worse: E10 (Walmart's revenue growth), from 1 to 0: the answer gives membership income. E01 (Google Cloud) stays 0.
- Refusal and not reported 32/32 (A1 29/32): RF10, RF13, RF14 now right (W3-06).
- 95/102 in all (A1 86, B0 50). Against the plan's targets: refusal and summaries met, entity kept, numeric 44 of
  the 46 aimed for. Still wrong: B08, B14, C08 (line items the facts tool has no metric for).

**Cost:** latency p50 5.54 s, p95 11.2 s and 3,419 tokens a question (A1: 4.97 s, 12.5 s and 3,000). A5 makes 1.90 LLM
calls and 1.67 tool calls a question (A1 1.83 and 1.63), so the optimization target is not met on gold.

**The critic:** it verified 98 of 102 answers (A1's verified all). 85 of the 87 decided verified answers are right,
and the two decided answers it left unverified are both wrong.
