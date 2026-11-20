# The agent (A1) next to B0 and S1 / S2 / S3 on Qwen3-8B (W2-11)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| Short: numeric | 16/48 = 33% [22%–47%] | 17/48 = 35% [23%–50%] | 20/48 = 42% [29%–56%] | 21/48 = 44% [31%–58%] | 42/48 = 88% [75%–94%] |
| Short: entity | 5/10 = 50% [24%–76%] | 5/10 = 50% [24%–76%] | 6/8 = 75% [41%–93%], 2 pending | 8/10 = 80% [49%–94%] | pending (10) |
| Summary | 8/12 = 67% [39%–86%] | pending (12) | pending (12) | pending (12) | pending (12) |
| Refusal & not reported | 21/32 = 66% [48%–80%] | 27/32 = 84% [68%–93%] | 22/32 = 69% [51%–82%] | 18/32 = 56% [39%–72%] | 29/32 = 91% [76%–97%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| fact | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 14/15 = 93% [70%–99%] |
| ratio | 7/15 = 47% [25%–70%] | 8/15 = 53% [30%–75%] | 6/15 = 40% [20%–64%] | 8/15 = 53% [30%–75%] | 11/15 = 73% [48%–89%] |
| period_comparison | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] | 11/12 = 92% [65%–99%] |
| multi_turn | 1/6 = 17% [3%–56%] | 1/6 = 17% [3%–56%] | 2/6 = 33% [10%–70%] | 2/6 = 33% [10%–70%] | 6/6 = 100% [61%–100%] |
| multi_company | 5/10 = 50% [24%–76%] | 5/10 = 50% [24%–76%] | 6/8 = 75% [41%–93%], 2 pending | 8/10 = 80% [49%–94%] | pending (10) |
| qualitative | 8/12 = 67% [39%–86%] | pending (12) | pending (12) | pending (12) | pending (12) |
| refuse | 5/10 = 50% [24%–76%] | 7/10 = 70% [40%–89%] | 8/10 = 80% [49%–94%] | 6/10 = 60% [31%–83%] | 10/10 = 100% [72%–100%] |
| refusal tests: refusal | 9/13 = 69% [42%–87%] | 13/13 = 100% [77%–100%] | 7/13 = 54% [29%–77%] | 5/13 = 38% [18%–64%] | 11/13 = 85% [58%–96%] |
| refusal tests: partial | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] | 1/2 = 50% [9%–91%] |
| refusal tests: control | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| unsupported numbers | 20/74 | 15/72 | 4/85 | 19/88 | 0/81 |
| wrong fiscal year | 0/74 | 0/72 | 0/85 | 0/88 | 0/81 |
| wrong company | 0/74 | 0/72 | 0/85 | 0/88 | 0/81 |
| cited a passage it was not given (closed-book: any citation) | 0/74 | 0/72 | 0/85 | 0/88 | 0/81 |
| value given for a 'not reported' trap | 1/3 | 0/3 | 0/3 | 1/3 | 0/3 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| precision (refused and should) | 18/28 | 20/30 | 14/17 | 13/14 | 20/21 |
| recall (should refuse and did) | 18/20 | 20/20 | 14/20 | 13/20 | 20/20 |
| right rule among correct refusals | 14/18 | 20/20 | 12/14 | 10/13 | 18/20 |
| over-refusals | 10 | 10 | 3 | 1 | 1 |
| under-refusals | 3 | 0 | 7 | 8 | 0 |

### 5. Output validity, tokens, latency and human corrections

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| valid B0 JSON | 102/102 | 102/102 | 102/102 | 102/102 | 102/102 |
| JSON retries | 0 | 0 | 0 | 0 | 0 |
| no reply (errors) | 0 | 0 | 0 | 0 | 0 |
| prompt tokens (mean) | 3448 | 4465 | 18912 | 7073 | 2781 |
| completion tokens (mean) | 88 | 81 | 972 | 324 | 219 |
| latency p50 / p95 (s, answered calls) | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 | 5.0 / 12.4 |
| cached replies | 102/102 | 0/102 | 101/102 | 0/102 | 0/102 |
| automatic verdicts overturned by graders | 0 | 0 | 0 | 0 | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| unit_confusion | 0 | 0 | 2 | 2 | 0 |
| arithmetic_slip | 10 | 10 | 5 | 7 | 5 |
| invented_number | 8 | 11 | 2 | 9 | 0 |
| wrong_line_item | 7 | 5 | 14 | 7 | 1 |
| ignored_not_reported | 1 | 0 | 0 | 2 | 0 |
| wrong_company | 5 | 5 | 2 | 2 | 0 |
| over_refusal | 10 | 10 | 3 | 1 | 1 |
| under_refusal | 3 | 0 | 7 | 8 | 0 |
| malformed_output | 1 | 0 | 3 | 2 | 0 |
| incomplete_summary | 4 | 0 | 0 | 0 | 0 |
| wrong_refusal_rule | 3 | 0 | 2 | 3 | 2 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| graders' exact agreement (independent grades) | 7/12 | – | – | – | – |
| graders' Cohen's kappa | 0.17 | – | – | – | – |
| final grade vs judge: exact agreement | 12/12 | – | – | – | – |
| final grade vs judge: Cohen's kappa | 1.00 | – | – | – | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit) (an agent run: in what its own tools returned)

| Gold dollar figures | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| all in the passages | 12/18 = 67% [44%–84%] | 13/18 = 72% [49%–88%] | 17/18 = 94% [74%–99%] | 17/18 = 94% [74%–99%] | 40/40 = 100% [91%–100%] |
| not all in the passages | 3/28 = 11% [4%–27%] | 3/28 = 11% [4%–27%] | 1/28 = 4% [1%–18%] | 2/28 = 7% [2%–23%] | 0/6 = 0% [0%–39%] |

### 9. Cost per question and self-consistency

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|---|
| model calls per question (mean, JSON retries included) | 1.0 | 1.0 | 5.0 | 2.0 | 1.8 |
| tokens per question (mean, prompt + completion) | 3536 | 4546 | 19883 | 7397 | 3000 |
| seconds per question p50 / p95 | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 | 5.0 / 12.4 |
| S2: all valid samples agreed | – | – | 62/102 | – | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – | – | 33/102 | – | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – | – | 7/102 | – | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | 8 / 4 / 0 | 8 / 4 / 0 | 8 / 3 / 1 | 9 / 3 / 0 | 3 / 9 / 0 |

### 10. Paired with M1_qwen3-8b_rag_think-off_b0-v1 (B0): questions right only with the run / only with B0, two-sided sign test

| Answers | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|---|---|---|
| Short: numeric | 3 / 2 (p = 1.00, n = 48) | 6 / 2 (p = 0.29, n = 48) | 7 / 2 (p = 0.18, n = 48) | 29 / 3 (p < 0.01, n = 48) |
| Short: entity | 1 / 1 (p = 1.00, n = 10) | 2 / 0 (p = 0.50, n = 8) | 3 / 0 (p = 0.25, n = 10) | – |
| Summary | – | – | – | – |
| Refusal & not reported | 6 / 0 (p = 0.03, n = 32) | 4 / 3 (p = 1.00, n = 32) | 1 / 4 (p = 0.38, n = 32) | 8 / 0 (p = 0.01, n = 32) |
| All decided answers | 10 / 3 (p = 0.09, n = 90) | 12 / 5 (p = 0.14, n = 88) | 11 / 6 (p = 0.33, n = 90) | 37 / 3 (p < 0.01, n = 80) |

### 11. The agent's own checks and cost

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| verified by its critic | 102/102 |
| correct among the verified (decided) | 71/80 |
| correct among the not verified (decided) | – |
| tool calls per question (mean); questions with none | 1.6; 21 |
| tool calls that returned an error | 4/166 |
| stopped by a limit (time, steps, tokens) | 0 |
| live requests / cache hits (chat replies) | 187 / 0 |
| short numeric answers with several values and no calculation (no single value to score) | 0 |
<!-- GENERATED:END -->

## Interpretation

W2-11, 21 Nov 2026 (ADR-034). A1 is the agent graph (planner, executor, composer, critic; ADR-024 to ADR-031) on
the same 80 gold questions and 22 refusal tests as B0 and S1-S3, run once by `eval/run_agent.py` on the shared
endpoint (Qwen3-8B, thinking off, temperature 0): 187 live requests, no cache hits, no errors, no question stopped
by a limit. Its prompts are the planner and composer prompts tuned on the development set and pinned by sha256
before this run; the gold set was not read by any of them. Unlike B0 and S1-S3 it had no context pack: it chose its
own tools (XBRL facts, the calculator, the filing search) and saw only each question and, for a follow-up, the
earlier questions, never an earlier answer. Same scorer, same tolerances, paired on the same questions.

**Decided answers.**
- Short numeric: 42/48 against B0's 16/48 (29 right only with A1, 3 only with B0; sign test p ≈ 3×10⁻⁶).
  - By category: fact 14/15, ratio 11/15, period comparison 11/12 (B0 0/12, the best strategy 4/12), follow-ups 6/6.
- Refusal and not reported: 29/32 against 21/32 (8 / 0, p ≈ 0.008).
  - Recall 20/20, precision 20/21, the right rule on 18 of 20, no under-refusal, the three "not reported" traps right.
- All decided answers: 37 / 3 against B0 (p ≈ 2×10⁻⁸, n = 80).
- Pending: the 10 multi-company answers name both companies, so a grader decides each (the scorer's rule). So do
  the 12 summaries.

**Why.** Table 8 says where the gain comes from. On 40 of the 46 numeric questions with gold dollar figures, the
agent's own tools returned every gold figure, and it got all 40 right. On the other 6 it got none. Its arithmetic
runs in the calculator, and its figures come from XBRL facts: no unsupported number in 81 answers, and every
citation names a source a tool returned. B0's figures had to be read off pack passages, and 28 of these 46 packs
lacked a gold figure. The bottleneck is now which figure the agent fetches, not how it reads or computes it.

**The 9 wrong answers.** Six of them are one failure, silent metric substitution: the facts tool has no metric for
the line item asked, and the planner fetched a nearby one, which the answer then names as the one asked.
- A06: total assets for total deposits.
- B04: operating cash flow (−42,012) for noninterest expense, giving "23.7%".
- B15: total assets and liabilities for current assets and liabilities.
- B14: company-wide operating income for the AWS segment's.
- C08: total revenue, with membership fees, for net sales (8.2% against 8.1%).
- B08: answered "not reported separately" but still carried a value.

The other three are refusals:
- RF14: a partial refusal refused whole.
- RF10 and RF13: the right decision under the wrong rule (R2 for R6, R5 for R7).

The critic verified every answer: 71 of the 80 verified decided answers are right, so it passed 9 wrong ones. It
never sent a draft back, because the composer already fixes contract problems itself (node notes on E01 and E05).
No current check compares the metric a value came from with the line item the answer names.

**Summaries.** The LLM judge gives A1's 12 summaries 3 twos, 9 ones and no zero, against 8 or 9 twos for every
other run. Its reasons are mostly "adds factors the reference does not mention" or "misses drivers the reference
names". A1's summaries are longer, and rest on passages it found itself rather than the gold answer's section. The
judge is a second opinion that marks additions down; the graders decide in the `--pending` round. The handout
shows them what the agent retrieved.

**Cost.** 1.8 model calls and 3.0k tokens per question (B0: 1 call, 3.5k), 5.0 s median and 12.4 s at p95 (the
limit is 60 s). The facts tool was called 102 times, the calculator 34, the search 30; 4 of 166 calls returned an
error, which the agent handled.

**Limits of this comparison.**
- One run at temperature 0.
- B0 and S1-S3 were given oracle-filtered packs (W1-12), and A1 found its own sources, so the comparison is between
  systems, not prompts.
- The hallucination check asks whether a figure is grounded in what the tools returned, not whether it is the
  right line item, which is exactly how A06, B04 and B15 pass it.

### Week 3 ablations (plan)

Each ablation turns off one component on the A1 configuration, unchanged otherwise:
- same questions, scorer, tolerances and seed;
- scored by `eval/score_answers.py` and paired with A1 by the sign test.

They need a switch in `eval/run_agent.py` (`--ablation`, run ids A2-A4) with offline tests. Run them one at a time
on the shared key, through the reply cache: a call whose prompt is unchanged replays for free, so each costs only
the calls it changes.

| Run | Switched off | How | What it measures | Expected from A1 | New requests (est.) |
|---|---|---|---|---|---|
| A2 | the calculator | `calculate` removed from the executor's tools and the planner's tool list; the composer computes from the fetched values | what the tool adds over in-prompt arithmetic (S2's best was 4/12 period comparisons) | ratio 11/15 and period comparison 11/12 fall towards S2/S3 levels; facts unchanged | ~140 (every planner call, plus the composer on the 34 calculations) |
| A3 | the critic | the critic passes every draft (verified = false) | what the critic catches | no answer changes: it rejected no A1 draft; only the verified flag | ~0 (every call replays) |
| A4 | statements first | `search_filings` without `StatementFirstIndex` (plain hybrid, ADR-017) | what the Item 8 preference adds | changes only on the 30 questions that searched, mostly summaries and traps | ~30 (composers after a search) |

A3's expected null result is itself the finding: on these questions the critic adds no correction. So Week 3
should strengthen it before ablating it, by checking that each value's metric is the line item the answer names,
which would have flagged A06, B04, B14, B15 and C08. The metric gaps behind those (deposits, current assets and
liabilities, segment figures, net sales separate from total revenue) are a tools fix. Order:
1. The ablations, on the frozen A1 configuration, so each switch is measured against the same baseline.
2. The critic check and the metric additions, as a new version (A5) on the development set first.
