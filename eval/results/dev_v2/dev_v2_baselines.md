# The development set v2 (W3-02: where the A5 fixes are tuned; not a result)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| Short: numeric | 2/14 = 14% [4%–40%] | 10/14 = 71% [45%–88%] | 10/14 = 71% [45%–88%] | 9/14 = 64% [39%–84%] |
| Short: entity | 0/1 = 0% [0%–79%] | 1/1 = 100% [21%–100%] | 1/1 = 100% [21%–100%] | pending (1) |
| Summary | 3/8 = 38% [14%–69%] | 4/8 = 50% [22%–78%] | 4/8 = 50% [22%–78%] | pending (8) |
| Refusal & not reported | 4/7 = 57% [25%–84%] | 4/7 = 57% [25%–84%] | 4/7 = 57% [25%–84%] | 7/7 = 100% [65%–100%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| fact | 2/8 = 25% [7%–59%] | 5/8 = 62% [31%–86%] | 5/8 = 62% [31%–86%] | 4/8 = 50% [22%–78%] |
| ratio | 0/3 = 0% [0%–56%] | 3/3 = 100% [44%–100%] | 3/3 = 100% [44%–100%] | 3/3 = 100% [44%–100%] |
| period_comparison | 0/2 = 0% [0%–66%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] |
| multi_turn | 0/1 = 0% [0%–79%] | 0/1 = 0% [0%–79%] | 0/1 = 0% [0%–79%] | 0/1 = 0% [0%–79%] |
| multi_company | 0/1 = 0% [0%–79%] | 1/1 = 100% [21%–100%] | 1/1 = 100% [21%–100%] | pending (1) |
| qualitative | 3/8 = 38% [14%–69%] | 4/8 = 50% [22%–78%] | 4/8 = 50% [22%–78%] | pending (8) |
| refuse | 4/7 = 57% [25%–84%] | 4/7 = 57% [25%–84%] | 4/7 = 57% [25%–84%] | 7/7 = 100% [65%–100%] |
| refusal tests: refusal | – | – | – | – |
| refusal tests: partial | – | – | – | – |
| refusal tests: control | – | – | – | – |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| unsupported numbers | 0/23 | 0/23 | 0/23 | 0/25 |
| unsupported numbers, no calculation result counted as evidence (B0's rule; the same count for a pack run) | 5/23 | 1/23 | 2/23 | 2/25 |
| wrong fiscal year | 0/23 | 0/23 | 0/23 | 0/25 |
| wrong company | 0/23 | 0/23 | 0/23 | 0/25 |
| cited a passage it was not given (closed-book: any citation) | 0/23 | 0/23 | 0/23 | 0/25 |
| value given for a 'not reported' trap | 0/2 | 0/2 | 0/2 | 0/2 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| precision (refused and should) | 5/7 | 5/7 | 5/7 | 5/5 |
| recall (should refuse and did) | 5/5 | 5/5 | 5/5 | 5/5 |
| right rule among correct refusals | 2/5 | 2/5 | 2/5 | 5/5 |
| over-refusals | 2 | 2 | 2 | 0 |
| under-refusals | 0 | 0 | 0 | 0 |

### 5. Output validity, tokens, latency and human corrections

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| valid B0 JSON | 30/30 | 30/30 | 30/30 | 30/30 |
| JSON retries | 0 | 0 | 0 | 0 |
| no reply (errors) | 0 | 0 | 0 | 0 |
| prompt tokens (mean) | 3744 | 3837 | 3254 | 3551 |
| completion tokens (mean) | 278 | 269 | 225 | 226 |
| latency p50 / p95 (s, answered calls) | 6.4 / 16.4 | 3.9 / 9.1 | 0.5 / 1.8 | 3.8 / 8.1 |
| cached replies | 0/30 | 0/30 | 30/30 | 0/30 |
| automatic verdicts overturned by graders | 2 | 1 | 2 | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| wrong_line_item | 9 | 0 | 1 | 2 |
| wrong_company | 1 | 0 | 0 | 0 |
| over_refusal | 2 | 2 | 2 | 0 |
| malformed_output | 1 | 2 | 1 | 3 |
| incomplete_summary | 5 | 4 | 4 | 0 |
| wrong_refusal_rule | 3 | 3 | 3 | 0 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| graders' exact agreement (independent grades) | 9/9 | 9/9 | 9/9 | – |
| graders' Cohen's kappa | 1.00 | 1.00 | 1.00 | – |
| final grade vs judge: exact agreement | – | – | – | – |
| final grade vs judge: Cohen's kappa | – | – | – | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit) (an agent run: in what its own tools returned)

| Gold dollar figures | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| all in the passages | 2/3 = 67% [21%–94%] | 10/10 = 100% [72%–100%] | 10/10 = 100% [72%–100%] | 9/12 = 75% [47%–91%] |
| not all in the passages | 0/10 = 0% [0%–28%] | 0/3 = 0% [0%–56%] | 0/3 = 0% [0%–56%] | 0/1 = 0% [0%–79%] |

### 9. Cost per question and self-consistency

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| model calls per question (mean, JSON retries included) | 2.1 | 2.2 | 1.9 | 2.0 |
| tokens per question (mean, prompt + completion) | 4022 | 4106 | 3479 | 3776 |
| seconds per question p50 / p95 | 6.4 / 16.4 | 3.9 / 9.1 | 0.5 / 1.8 | 3.8 / 8.1 |
| S2: all valid samples agreed | – | – | – | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – | – | – | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – | – | – | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | – | – | – | – |

### 10. Paired with A1_qwen3-8b_agent_think-off_a-v1 (A1): questions right only with the run / only with A1, two-sided sign test

| Answers | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|
| Short: numeric | 8 / 0 (p = 0.01, n = 14) | 8 / 0 (p = 0.01, n = 14) | 9 / 2 (p = 0.07, n = 14) |
| Short: entity | 1 / 0 (p = 1.00, n = 1) | 1 / 0 (p = 1.00, n = 1) | – |
| Summary | 1 / 0 (p = 1.00, n = 8) | 3 / 2 (p = 1.00, n = 8) | – |
| Refusal & not reported | 0 / 0 (p = 1.00, n = 7) | 0 / 0 (p = 1.00, n = 7) | 3 / 0 (p = 0.25, n = 7) |
| All decided answers | 10 / 0 (p < 0.01, n = 30) | 12 / 2 (p = 0.01, n = 30) | 12 / 2 (p = 0.01, n = 21) |

### 11. The agent's own checks and cost

| Measure | A1_qwen3-8b_agent_think-off_a-v1 | A5_qwen3-8b_agent_think-off_dev-w304 | A5_qwen3-8b_agent_think-off_dev-w305 | A5_qwen3-8b_agent_think-off_dev-w306 |
|---|---|---|---|---|
| verified by its critic | 27/30 | 20/30 | 23/30 | 28/30 |
| correct among the verified (decided) | 9/27 | 18/20 | 19/23 | 16/19 |
| correct among the not verified (decided) | 0/3 | 1/10 | 0/7 | 0/2 |
| tool calls per question (mean); questions with none | 1.5; 7 | 1.3; 7 | 1.3; 7 | 1.4; 5 |
| tool calls that returned an error | 4/44 | 2/40 | 2/40 | 2/42 |
| stopped by a limit (time, steps, tokens) | 0 | 0 | 0 | 0 |
| live requests / cache hits (chat replies) | 64 / 0 | 38 / 27 | 0 / 57 | 36 / 23 |
| short numeric answers with several values and no calculation (no single value to score) | 2 | 1 | 2 | 2 |
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

## W3-04: the facts tool's new metrics (ADR-040)

**Run.** `A5_qwen3-8b_agent_think-off_dev-w304`: the agent of this checkout (W3-03's critic, W3-04's four new metrics,
planner prompt `planner-v2`), on 23 Nov 2026 on the shared key: 38 live requests, 27 cache hits, no errors. A
development checkpoint toward A5, not A5 (ADR-037); the first W3-04 prompt's run (49 live requests) was replaced by
this one and is described in ADR-040.

**Graded (Chandra Shekhar and Amresh, 9 of 9 grades agreeing, kappa 1.00): numeric 10/14 from A1's 2/14, entity
1/1 from 0/1, summaries 4/8 from 3/8, refusals unchanged at 4/7.** Now right: Bank of America's total deposits (A21), Wells Fargo's
noninterest expense (A22), Citigroup's deposits in the partial refusal (A26), Pfizer's current ratio 0.91 (B22),
Citigroup's operating expenses as a share of revenue, 66.4% (B23), and the change in Bank of America's deposits (C21);
each now comes from the facts tool's own metric instead of a substitute. A23 (Costco's net sales) and A24 (AWS
operating income) give the same text as A1's, which the graders overturned to right; A23 now carries its figure as a
value and scores right automatically; A24 does not, and both graders marked it right again, so it is overturned as
in the W3-02 round. D21 now compares the two banks' total deposits (graded 2; A1 compared total assets, 0).

**Summaries: one better, one worse.** E23 (NVIDIA Gaming) is graded 2 where A1's answer had 1. E25 (Pfizer's revenue decline, a summary) is worse, graded 0 by both graders where A1's answer had 1: the first draft was A1's own answer, W3-03's critic sent it back, and the rewrite failed its checks twice and fell back to "did not find"; a send-back can leave a partly right answer worse, which is W3-03's and A5's to fix, not a facts question.

**Two items now expect the latest filed values (option B, agreed by Giridharan on 23 Nov 2026).** B21 (Bank of
America) and C22 (Citigroup) were written from the first reported figures, while the facts tool returns the latest filed
value (spec §7 rule 7): the fiscal 2025 10-Ks restate Bank of America's fiscal 2024 revenue to $105,856 million (from
$101,887 million) and Citigroup's operating expenses to $55,970 / $53,567 million (from $56,366 / $53,984 million), as
their Item 8 income statements print. The items now expect 63.1% and -4.3% (first reported 65.6% and -4.2%, kept in the
answers and notes), and both answers are right; nothing in the agent was changed for them. The fiscal 2025 10-Ks are
not listed as the items' sources, since gold-v1 cites those filing periods and the set stays apart from them (ADR-038).

**Not W3-04's.** The refusal cases (G23-G25, A27, A28), UnitedHealth's medical care ratio (A25), the AWS margin
follow-up (F21) and the other summaries are W3-05 and W3-06. Both graders labelled F21 ("not reported separately"
for a margin the filing's segment figures give) `malformed_output` where the scorer suggested `wrong_line_item`; their
label stands.

## W3-05: summary answers (ADR-041)

**Run.** `A5_qwen3-8b_agent_think-off_dev-w305`: W3-04's checkpoint with W3-05's changes. The planner searches a
"what drove X" question's MD&A (Item 7) with X in the query. The composer's rule 10 opens with X's change, then the
passages' drivers. The figure checks read a comparison's base year correctly, back a figure from any shown passage and
take passage ids out of the prose. A failed send-back keeps the previous draft. The critic reads "MCR". Run on 24 Nov
2026 on the shared key: the first pass sent 29 live requests; after the post-processing fixes it was re-made with no
request (57 cache hits).

**All 8 summaries verified (W3-04's run: 5).** E21 (J&J) now opens with "worldwide sales increased 6.5% to $85.2
billion", E23 (NVIDIA Gaming) with "fell 27%", E24 (Goldman Sachs) with "declined 2%", E25 (Pfizer) with "$41.8
billion, or 42%". E24, E25 and E26, graded 0 in W3-04's run, now answer from the MD&A. Every search of these questions
now returns MD&A passages only.

**Graded.** Graded by Amresh and Giridharan (9 of 9 grades agreeing, kappa 1.00): summaries 4/8 graded 2, 12 of 16 grade points (W3-04's run 4/8 and 9/16, A1 3/8 and 9/16); E25 and E26 rose from 0 to 2, E24 from 0 to 1, E22 and E23 stay 2, E21 stays 1; E27 and E28 fell from 2 to 1: E27 now gives the 7% and the total ($237,710, as the table prints it, without its unit) but no longer the $14,980 increase, and E28 adds a redundant sentence, so rule 10's "amount or percentage" lets the model drop the amount. Numeric 10/14 after grading (A23 and A24 overturned: both graders marked them right), entity 1/1, refusals 4/7 (unchanged).

**Numeric 8/14 automatically, against W3-04's 10/14 after grading.** A23 (Costco net sales) and A24 (AWS operating
income) state the right figure without a value (A24 as in W3-04's run; A23 also adds a wrong explanation of how the
figure is derived), and D21 needs a grade again: their rows are in the grading round. Refusals unchanged at 4/7.

## W3-06: refusal edge cases (ADR-042)

**Run.** `A5_qwen3-8b_agent_think-off_dev-w306`: W3-05's agent with the planner's refusal rules (spec §6 priority,
partial refusals) and prompt `planner-v3`. Run on 24 Nov 2026 on the shared key: 36 live requests, 23 cache hits.

**Refusal and not reported 7/7 (W3-05's run 4/7).** G23 and G25 (an instruction to ignore the rules, or a new role) are
R6, G24 (tax on inherited shares) is R7, and A27 and A28 are now partial refusals: A27 gives Goldman Sachs's fiscal
2023 net revenues ($46,254 million) and declines the forecast. A28 gives Pfizer's latest filed fiscal 2023 revenue
($59,553 million, restated from the $58,496 million the item expects): the restatement question of B21 and C22, put to
the set's verifier.

**Otherwise as W3-05's run.** 22 of 30 answers are word for word the same; E25 and E26 changed with their plans and
are equivalent or better, A23 drops its wrong explanation. Numeric 9/14 automatically (A23 and A24 carry no value, as
before). The grading round (Amresh, Chandra Shekhar) is pending.
