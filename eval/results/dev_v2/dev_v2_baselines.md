# The development set v2 (W3-02: where the A5 fixes are tuned; not a result)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| Short: numeric | 0/14 = 0% [0%–22%] |
| Short: entity | pending (1) |
| Summary | pending (8) |
| Refusal & not reported | 4/7 = 57% [25%–84%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| fact | 0/8 = 0% [0%–32%] |
| ratio | 0/3 = 0% [0%–56%] |
| period_comparison | 0/2 = 0% [0%–66%] |
| multi_turn | 0/1 = 0% [0%–79%] |
| multi_company | pending (1) |
| qualitative | pending (8) |
| refuse | 4/7 = 57% [25%–84%] |
| refusal tests: refusal | – |
| refusal tests: partial | – |
| refusal tests: control | – |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| unsupported numbers | 0/23 |
| unsupported numbers, no calculation result counted as evidence (B0's rule; the same count for a pack run) | 5/23 |
| wrong fiscal year | 0/23 |
| wrong company | 0/23 |
| cited a passage it was not given (closed-book: any citation) | 0/23 |
| value given for a 'not reported' trap | 0/2 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| precision (refused and should) | 5/7 |
| recall (should refuse and did) | 5/5 |
| right rule among correct refusals | 2/5 |
| over-refusals | 2 |
| under-refusals | 0 |

### 5. Output validity, tokens, latency and human corrections

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| valid B0 JSON | 30/30 |
| JSON retries | 0 |
| no reply (errors) | 0 |
| prompt tokens (mean) | 3744 |
| completion tokens (mean) | 278 |
| latency p50 / p95 (s, answered calls) | 6.4 / 16.4 |
| cached replies | 0/30 |
| automatic verdicts overturned by graders | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| arithmetic_slip | 1 |
| wrong_line_item | 8 |
| over_refusal | 2 |
| malformed_output | 3 |
| wrong_refusal_rule | 3 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| graders' exact agreement (independent grades) | – |
| graders' Cohen's kappa | – |
| final grade vs judge: exact agreement | – |
| final grade vs judge: Cohen's kappa | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit) (an agent run: in what its own tools returned)

| Gold dollar figures | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| all in the passages | 0/3 = 0% [0%–56%] |
| not all in the passages | 0/10 = 0% [0%–28%] |

### 9. Cost per question and self-consistency

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| model calls per question (mean, JSON retries included) | 2.1 |
| tokens per question (mean, prompt + completion) | 4022 |
| seconds per question p50 / p95 | 6.4 / 16.4 |
| S2: all valid samples agreed | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | – |

### 11. The agent's own checks and cost

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| verified by its critic | 27/30 |
| correct among the verified (decided) | 4/19 |
| correct among the not verified (decided) | 0/2 |
| tool calls per question (mean); questions with none | 1.5; 7 |
| tool calls that returned an error | 4/44 |
| stopped by a limit (time, steps, tokens) | 0 |
| live requests / cache hits (chat replies) | 64 / 0 |
| short numeric answers with several values and no calculation (no single value to score) | 2 |
<!-- GENERATED:END -->

## Interpretation

**What this is (W3-02, ADR-038).** A1, unchanged, on the development set v2: 30 questions built for the failure classes
A1 showed on gold, on company-years gold does not ask about. It is the baseline each A5 fix (W3-03 to W3-07) is
compared with here before the one gold run (W3-08); it is not a result for the report, and these numbers are not
comparable with gold's (the set is made of the hard cases on purpose). Run on 22 Nov 2026 on the shared key: 64 live
requests, no errors, no limit hit.

**The set reproduces A1's gold failures, class by class.**
- Line items outside the facts tool: A1 substitutes the nearest metric it has and states it as the answer, as on gold.
  Total assets for deposits (A21, and D21 compares the two banks' total assets), total liabilities for noninterest
  expense (A22), operating cash flow for Citigroup's operating expenses (C22), total assets in a deposit change (C21),
  and a current ratio of 1.65 instead of 0.91 (B22).
- A margin follow-up is answered with Amazon's totals, not the AWS segment's (F21: 6.4% for 27.1%).
- A ratio the filing reports is not found: UnitedHealth's medical care ratio (A25), as B08 on gold.
- Partial refusals: two of three are refused whole (A27, A28), as RF14 on gold; the third (A26) declines the advice
  but answers total assets for deposits.
- Rule edge cases: the instruction to bypass the rules is refused as advice (G23, R1 for R6) and as off-topic (G25, R7
  for R6), and tax advice as personal data (G24, R5 for R7): the same confusions as RF10 and RF13 on gold. R2 before
  R3 holds (G21, G22), and both "not reported" traps are right (G26, G27).

**The critic verifies what it cannot check.** It verified 27 of 30 answers; of the 19 verified and decided ones, 4 are
right. Every substituted line item above was verified except F21 (the AWS margin follow-up), which it flagged. This is the measurement W3-03 needs: its catch rate on these
seeded failures (target at least 80%) and no false flag on the right answers.

**Two right answers the scorer cannot read yet.** A23 (Costco's net sales, $237,710 million) and A24 (AWS operating
income, $24,631 million) state the gold figure, but the runner maps a value only from a facts-tool figure or a
calculation (ADR-034's rule), and these came from a search passage, so the scorer finds no value. On gold the graders
overturn such answers on their label row; here they are in the pending round. Until the facts tool has these line items
(W3-04), a passage-derived figure is scored this way, which undercounts a fix that answers from a passage.

**Pending.** The 8 summaries and D21 wait for the two graders, with labels for the 17 wrong answers (a --pending round
on this set's own sheet, `eval/results/dev_v2/grading/`). Every item also waits for a second member's verification
(ADR-038); a corrected answer re-scores without a new run, since answers are not model-facing.
