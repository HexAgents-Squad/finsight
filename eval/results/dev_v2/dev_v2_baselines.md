# The development set v2 (W3-02: where the A5 fixes are tuned; not a result)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| Short: numeric | 2/14 = 14% [4%–40%] |
| Short: entity | 0/1 = 0% [0%–79%] |
| Summary | 3/8 = 38% [14%–69%] |
| Refusal & not reported | 4/7 = 57% [25%–84%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| fact | 2/8 = 25% [7%–59%] |
| ratio | 0/3 = 0% [0%–56%] |
| period_comparison | 0/2 = 0% [0%–66%] |
| multi_turn | 0/1 = 0% [0%–79%] |
| multi_company | 0/1 = 0% [0%–79%] |
| qualitative | 3/8 = 38% [14%–69%] |
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
| automatic verdicts overturned by graders | 2 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| wrong_line_item | 9 |
| wrong_company | 1 |
| over_refusal | 2 |
| malformed_output | 1 |
| incomplete_summary | 5 |
| wrong_refusal_rule | 3 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| graders' exact agreement (independent grades) | 9/9 |
| graders' Cohen's kappa | 1.00 |
| final grade vs judge: exact agreement | – |
| final grade vs judge: Cohen's kappa | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit) (an agent run: in what its own tools returned)

| Gold dollar figures | A1_qwen3-8b_agent_think-off_a-v1 |
|---|---|
| all in the passages | 2/3 = 67% [21%–94%] |
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
| correct among the verified (decided) | 9/27 |
| correct among the not verified (decided) | 0/3 |
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

**The critic verifies what it cannot check.** It verified 27 of 30 answers, and 9 of those 27 are right: 18 wrong answers
left verified. Every substituted line item above was verified except F21 (the AWS margin follow-up), which it flagged.
This is the measurement W3-03 needs: its catch rate on these seeded failures (target at least 80%) and no false flag on
the right answers.

**Summaries: 3 of 8 right** (graded 2 by both graders: E22 AWS, E27 Costco, E28 Wells Fargo). Three are partly right
(graded 1): E21 J&J gives the acquisition effect and products but not the volume, price and currency bridge; E23 NVIDIA
gives the drivers without the 27% decline; E25 Pfizer names secondary items (Comirnaty supply to BioNTech, a royalty
reclassification) instead of the Comirnaty and Paxlovid declines. Two are wrong (graded 0): E24 Goldman Sachs answers
from a passage about other items (GreenSky, a $(367) million line) instead of the drivers of net revenues, and E26
UnitedHealth finds nothing, as A25 does for the same ratio. The pattern is W3-05's: the passage used is from the right
filing but not the one that states the asked quantity's drivers.

**Graded answers** (Chandra Shekhar and Amresh, 9 grades, all 9 agreeing):
- Two answers state the gold figure from a search passage (A23 Costco's net sales, A24 AWS operating income) and score
  "no value" under ADR-034's mapping (a value only from a facts-tool figure or a calculation); both graders marked them
  correct, so they are overturned (short numeric 2/14). Until the facts tool has these line items (W3-04), a figure
  taken from a passage is scored this way, which undercounts a fix that answers from a passage.
- D21 compares total assets, not deposits: graded 0.
- On G23 and G25 (an instruction to ignore the rules or to change role) both graders marked A1's R1 and R7 correct.
  docs/spec.md §6 makes such an instruction R6, the verifier confirmed R6 as the expected rule, and the graders labelled
  RF10 (the same shape) a wrong rule on gold, so both are recorded as `wrong_refusal_rule`, with the graders' view kept
  in the sheet's note. The disagreement is a reason for W3-06 to state the rule priority in the spec, not only by example.

**Verified.** Giridharan checked every item against the filings (30 of 30 OK), so the set passes the final rules, which
CI now runs.
