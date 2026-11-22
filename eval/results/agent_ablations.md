# Ablations A2 / A3 / A4: the agent with one component off, paired with A1 (W3-01)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| Short: numeric | 42/48 = 88% [75%–94%] | 19/48 = 40% [27%–54%] | 42/48 = 88% [75%–94%] | 42/48 = 88% [75%–94%] |
| Short: entity | 10/10 = 100% [72%–100%] | 6/6 = 100% [61%–100%], 4 pending | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] |
| Summary | 5/12 = 42% [19%–68%] | 5/12 = 42% [19%–68%] | 5/12 = 42% [19%–68%] | 3/6 = 50% [19%–81%], 6 pending |
| Refusal & not reported | 29/32 = 91% [76%–97%] | 29/32 = 91% [76%–97%] | 29/32 = 91% [76%–97%] | 29/32 = 91% [76%–97%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| fact | 14/15 = 93% [70%–99%] | 14/15 = 93% [70%–99%] | 14/15 = 93% [70%–99%] | 14/15 = 93% [70%–99%] |
| ratio | 11/15 = 73% [48%–89%] | 0/15 = 0% [0%–20%] | 11/15 = 73% [48%–89%] | 11/15 = 73% [48%–89%] |
| period_comparison | 11/12 = 92% [65%–99%] | 0/12 = 0% [0%–24%] | 11/12 = 92% [65%–99%] | 11/12 = 92% [65%–99%] |
| multi_turn | 6/6 = 100% [61%–100%] | 5/6 = 83% [44%–97%] | 6/6 = 100% [61%–100%] | 6/6 = 100% [61%–100%] |
| multi_company | 10/10 = 100% [72%–100%] | 6/6 = 100% [61%–100%], 4 pending | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] |
| qualitative | 5/12 = 42% [19%–68%] | 5/12 = 42% [19%–68%] | 5/12 = 42% [19%–68%] | 3/6 = 50% [19%–81%], 6 pending |
| refuse | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] |
| refusal tests: refusal | 11/13 = 85% [58%–96%] | 11/13 = 85% [58%–96%] | 11/13 = 85% [58%–96%] | 11/13 = 85% [58%–96%] |
| refusal tests: partial | 1/2 = 50% [9%–91%] | 1/2 = 50% [9%–91%] | 1/2 = 50% [9%–91%] | 1/2 = 50% [9%–91%] |
| refusal tests: control | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| unsupported numbers | 0/81 | 0/81 | 0/81 | 0/81 |
| unsupported numbers, no calculation result counted as evidence (B0's rule; the same count for a pack run) | 4/81 | 0/81 | 4/81 | 4/81 |
| wrong fiscal year | 0/81 | 0/81 | 0/81 | 0/81 |
| wrong company | 0/81 | 0/81 | 0/81 | 0/81 |
| cited a passage it was not given (closed-book: any citation) | 0/81 | 0/81 | 0/81 | 0/81 |
| value given for a 'not reported' trap | 0/3 | 0/3 | 0/3 | 0/3 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| precision (refused and should) | 20/21 | 20/21 | 20/21 | 20/21 |
| recall (should refuse and did) | 20/20 | 20/20 | 20/20 | 20/20 |
| right rule among correct refusals | 18/20 | 18/20 | 18/20 | 18/20 |
| over-refusals | 2 | 2 | 2 | 2 |
| under-refusals | 0 | 0 | 0 | 0 |

### 5. Output validity, tokens, latency and human corrections

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| valid B0 JSON | 102/102 | 102/102 | 102/102 | 102/102 |
| JSON retries | 0 | 0 | 0 | 0 |
| no reply (errors) | 0 | 0 | 0 | 0 |
| prompt tokens (mean) | 2781 | 3396 | 2781 | 2805 |
| completion tokens (mean) | 219 | 257 | 219 | 219 |
| latency p50 / p95 (s, answered calls) | 5.0 / 12.4 | 0.5 / 8.0 | 0.1 / 1.5 | 0.1 / 4.1 |
| cached replies | 0/102 | 68/102 | 102/102 | 88/102 |
| automatic verdicts overturned by graders | 0 | 0 | 0 | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| arithmetic_slip | 1 | 0 | 1 | 1 |
| wrong_line_item | 4 | 2 | 4 | 4 |
| over_refusal | 2 | 2 | 2 | 2 |
| malformed_output | 0 | 26 | 0 | 0 |
| incomplete_summary | 7 | 7 | 7 | 3 |
| wrong_refusal_rule | 2 | 2 | 2 | 2 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| graders' exact agreement (independent grades) | 21/22 | 17/18 | 21/22 | 15/16 |
| graders' Cohen's kappa | 0.91 | 0.90 | 0.91 | 0.81 |
| final grade vs judge: exact agreement | 6/12 | – | – | – |
| final grade vs judge: Cohen's kappa | 0.29 | – | – | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit) (an agent run: in what its own tools returned)

| Gold dollar figures | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| all in the passages | 40/40 = 100% [91%–100%] | 18/40 = 45% [31%–60%] | 40/40 = 100% [91%–100%] | 40/40 = 100% [91%–100%] |
| not all in the passages | 0/6 = 0% [0%–39%] | 0/6 = 0% [0%–39%] | 0/6 = 0% [0%–39%] | 0/6 = 0% [0%–39%] |

### 9. Cost per question and self-consistency

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| model calls per question (mean, JSON retries included) | 1.8 | 2.3 | 1.8 | 1.9 |
| tokens per question (mean, prompt + completion) | 3000 | 3654 | 3000 | 3024 |
| seconds per question p50 / p95 | 5.0 / 12.4 | 0.5 / 8.0 | 0.1 / 1.5 | 0.1 / 4.1 |
| S2: all valid samples agreed | – | – | – | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – | – | – | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – | – | – | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | 3 / 9 / 0 | – | – | – |

### 10. Paired with A1_qwen3-8b_agent_think-off_a-v1 (A1): questions right only with the ablation / only with A1, two-sided sign test

| Answers | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|
| Short: numeric | 0 / 23 (p < 0.01, n = 48) | 0 / 0 (p = 1.00, n = 48) | 0 / 0 (p = 1.00, n = 48) |
| Short: entity | 0 / 0 (p = 1.00, n = 6) | 0 / 0 (p = 1.00, n = 10) | 0 / 0 (p = 1.00, n = 10) |
| Summary | 0 / 0 (p = 1.00, n = 12) | 0 / 0 (p = 1.00, n = 12) | 0 / 0 (p = 1.00, n = 6) |
| Refusal & not reported | 0 / 0 (p = 1.00, n = 32) | 0 / 0 (p = 1.00, n = 32) | 0 / 0 (p = 1.00, n = 32) |
| All decided answers | 0 / 23 (p < 0.01, n = 98) | 0 / 0 (p = 1.00, n = 102) | 0 / 0 (p = 1.00, n = 96) |

### 11. The agent's own checks and cost

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|---|
| verified by its critic | 102/102 | 77/102 | 0/102 | 101/102 |
| correct among the verified (decided) | 86/102 | 59/73 | – | 84/96 |
| correct among the not verified (decided) | – | 0/25 | 86/102 | – |
| tool calls per question (mean); questions with none | 1.6; 21 | 1.3; 21 | 1.6; 21 | 1.6; 21 |
| tool calls that returned an error | 4/166 | 4/132 | 4/166 | 4/166 |
| stopped by a limit (time, steps, tokens) | 0 | 0 | 0 | 0 |
| live requests / cache hits (chat replies) | 187 / 0 | 85 / 153 | 0 / 187 | 18 / 171 |
| short numeric answers with several values and no calculation (no single value to score) | 0 | 27 | 0 | 0 |

### 12. What each ablation changed from A1_qwen3-8b_agent_think-off_a-v1

| Measure | A2_qwen3-8b_agent_think-off_a-v1-no-calculator | A3_qwen3-8b_agent_think-off_a-v1-no-critic | A4_qwen3-8b_agent_think-off_a-v1-no-statements-first |
|---|---|---|---|
| component switched off | no-calculator | no-critic | no-statements-first |
| answers that differ from the reference's (text) | 31/102 | 0/102 | 7/102 |
| of those: right both times / right only now / right only before / wrong both times / undecided | 0 / 0 / 23 / 4 / 4 | 0 / 0 / 0 / 0 / 0 | 1 / 0 / 0 / 0 / 6 |
| changed answers by type (numeric, entity, summary, refusal and not reported) | 27, 4, 0, 0 | 0, 0, 0, 0 | 1, 0, 6, 0 |
<!-- GENERATED:END -->

## Interpretation

**What was run (W3-01, ADR-036).** Each ablation turns one component off on the frozen A1 configuration: the same
102 questions (80 gold, 22 refusal tests), prompts, limits, retrieval settings and scorer. A2 has no calculator (the
executor never runs the plan's calculation), A3 no critic (the composer's draft is the answer, unverified), A4 no
statements first (search on plain hybrid retrieval). Run on 22 Nov 2026, one at a time on the shared key through the
reply cache, so only the calls a switch changed went live: A3 0, A4 18, A2 85 (103 in all). Before them, A1 was
replayed from the cache on this branch's code: 0 live requests and every record byte for byte the committed one, so
the switches leave A1 unchanged. Nothing here was tuned on these results.

**A2, the calculator, is the component that matters most.** Short numeric answers fall from 42/48 to 19/48: ratios
from 11/15 to 0/15, period comparisons from 11/12 to 0/12 and one follow-up (F02, an operating margin), while facts
stay at 14/15. Question by question A2 is right on none where A1 is wrong and wrong on 23 where A1 is right (sign
test p ≈ 2×10⁻⁷). The composer does not compute: in 27 answers it states the fetched figures without the asked
ratio or growth ("Alphabet's rd expense was $49,326 million ... revenue was $350,018 million" for an R&D share), and
the scorer finds no single value to score. It invents nothing either: unsupported numbers are 0/81 under both rules,
and the 4 that A1 has under B0's rule (where no calculation result counts as evidence) are its calculated values.
This is the design working as intended: the model never does arithmetic (ADR-025), so without the calculator a
derived figure is missing, not wrong. It costs more, too: 2.3 model calls and 3,654 tokens per question against
1.8 and 3,000, because the critic sends the drafts back (below).

**A3, the critic, changes no answer on these questions.** All 102 answers are word for word A1's, at no request:
A1's critic passed every first draft (ADR-034), so removing it removes only the verified flag. That flag was on 16
wrong answers in A1 (86 of 102 verified answers right); in A3 nothing is verified and the same 86 are right. This
is the expected null result, and it is the finding behind W3-03: the critic's checks are not the problem in kind
(in A2 its operation check flagged 25 of the 27 drafts that left out the asked calculation, sent each back once and
left them unverified; in A4 it flagged E01's summary for having no citation), but none of them tests what A1 gets
wrong: a substituted line item and the refusal rules. The two A2 drafts it passed are that class again: B02 gives
NVIDIA's gross profit as its "gross margin" ($97,858 million) and F02 answers a follow-up on Microsoft's operating
margin with its operating income, both verified.

**A4, statements first, changes little in the answers.** Seven answers differ from A1's: one numeric (B12, still
right) and six summaries, which wait for the graders; no refusal changed. Numeric answers do not move because the
agent takes figures from the facts tool, not from search, so the statements-first ordering (ADR-017's recall gain
for figure questions) only acts on the 30 questions that searched, mostly summaries and traps. One summary (E01)
lost its citation and was left unverified.

**Not comparable, and pending.**
- Latency: A2-A4 replayed most calls from the cache, which takes no time (A3's median 0.1 s), so their seconds are not
  a cost; compare model calls and tokens.
- Pending human grades: A4's six changed summaries and A2's four two-company answers (D01, D02, D05, D08), which now
  state both companies' figures and no winner the scorer can read, plus labels for A2's 27 wrong answers (the
  suggested `malformed_output` on 26 of them only means "no single value"; the graders give the real label). The
  round is prepared for Chandra Shekhar and Amresh; the scorer fills the cells when their grades are merged.
- One run each at temperature 0. A2 measures the calculator under the composer's guard (it may not state a figure
  no evidence prints); letting the composer compute would be a second change and its own run.
