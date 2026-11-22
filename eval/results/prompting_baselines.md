# Prompting baselines B0 / S1 / S2 / S3 on Qwen3-8B (W1-09)

Qwen3-8B on the hosted endpoint, the same context pack for every run (`eval/results/context_pack_*.jsonl`: oracle
company and year filters, C3 hybrid passages, at most 4,000 tokens) and the same 102 questions (80 gold-v1 + 22
refusal tests), scored by `eval/score_answers.py` (W1-08). B0 is the frozen zero-shot run M1 (W1-13). Each
strategy keeps B0's rules word for word and adds one thing (`eval/strategies.py`): S1 four worked examples from
the development set; S2 a reasoning field, 5 samples at temperature 0.7 (seed = sample) and a majority vote;
S3 a research step that lists sub-questions and copies the needed figures before the answer step. Strict
json_schema output for S1-S3 (ADR-016); B0 asked for JSON in the prompt only. Prompts were tuned on the 15
development questions only (three passes, `eval/results/dev/`), frozen and pinned by sha256 before these runs
(commit 21ccbad, tag w1-09-prompts-v1). Decisions: ADR-018 in `docs/decisions.md`.

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| Short: numeric | 16/48 = 33% [22%–47%] | 17/48 = 35% [23%–50%] | 20/48 = 42% [29%–56%] | 21/48 = 44% [31%–58%] |
| Short: entity | 5/10 = 50% [24%–76%] | 5/10 = 50% [24%–76%] | 6/10 = 60% [31%–83%] | 8/10 = 80% [49%–94%] |
| Summary | 8/12 = 67% [39%–86%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] | 3/12 = 25% [9%–53%] |
| Refusal & not reported | 21/32 = 66% [48%–80%] | 31/32 = 97% [84%–99%] | 22/32 = 69% [51%–82%] | 19/32 = 59% [42%–74%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| fact | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] | 8/15 = 53% [30%–75%] |
| ratio | 7/15 = 47% [25%–70%] | 8/15 = 53% [30%–75%] | 6/15 = 40% [20%–64%] | 8/15 = 53% [30%–75%] |
| period_comparison | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] |
| multi_turn | 1/6 = 17% [3%–56%] | 1/6 = 17% [3%–56%] | 2/6 = 33% [10%–70%] | 2/6 = 33% [10%–70%] |
| multi_company | 5/10 = 50% [24%–76%] | 5/10 = 50% [24%–76%] | 6/10 = 60% [31%–83%] | 8/10 = 80% [49%–94%] |
| qualitative | 8/12 = 67% [39%–86%] | 4/12 = 33% [14%–61%] | 3/12 = 25% [9%–53%] | 3/12 = 25% [9%–53%] |
| refuse | 5/10 = 50% [24%–76%] | 10/10 = 100% [72%–100%] | 8/10 = 80% [49%–94%] | 7/10 = 70% [40%–89%] |
| refusal tests: refusal | 9/13 = 69% [42%–87%] | 13/13 = 100% [77%–100%] | 7/13 = 54% [29%–77%] | 5/13 = 38% [18%–64%] |
| refusal tests: partial | 0/2 = 0% [0%–66%] | 1/2 = 50% [9%–91%] | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] |
| refusal tests: control | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] | 7/7 = 100% [65%–100%] |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| unsupported numbers | 20/74 | 15/72 | 4/85 | 19/88 |
| wrong fiscal year | 0/74 | 0/72 | 0/85 | 0/88 |
| wrong company | 0/74 | 0/72 | 0/85 | 0/88 |
| cited a passage it was not given (closed-book: any citation) | 0/74 | 0/72 | 0/85 | 0/88 |
| value given for a 'not reported' trap | 1/3 | 0/3 | 0/3 | 1/3 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| precision (refused and should) | 18/28 | 20/30 | 14/17 | 13/14 |
| recall (should refuse and did) | 18/20 | 20/20 | 14/20 | 13/20 |
| right rule among correct refusals | 14/18 | 20/20 | 12/14 | 10/13 |
| over-refusals | 10 | 5 | 9 | 4 |
| under-refusals | 3 | 0 | 3 | 4 |

### 5. Output validity, tokens, latency and human corrections

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| valid B0 JSON | 102/102 | 102/102 | 102/102 | 102/102 |
| JSON retries | 0 | 0 | 0 | 0 |
| no reply (errors) | 0 | 0 | 0 | 0 |
| prompt tokens (mean) | 3448 | 4465 | 18912 | 7073 |
| completion tokens (mean) | 88 | 81 | 972 | 324 |
| latency p50 / p95 (s, answered calls) | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 |
| cached replies | 102/102 | 0/102 | 101/102 | 0/102 |
| automatic verdicts overturned by graders | 0 | 4 | 0 | 1 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| wrong_fiscal_year | 0 | 0 | 0 | 1 |
| unit_confusion | 0 | 1 | 2 | 2 |
| arithmetic_slip | 10 | 11 | 5 | 6 |
| invented_number | 8 | 8 | 2 | 8 |
| wrong_line_item | 7 | 7 | 14 | 7 |
| ignored_not_reported | 1 | 0 | 0 | 1 |
| wrong_company | 5 | 5 | 2 | 2 |
| over_refusal | 10 | 5 | 9 | 4 |
| under_refusal | 3 | 0 | 3 | 4 |
| malformed_output | 1 | 0 | 0 | 0 |
| incomplete_summary | 4 | 8 | 8 | 9 |
| wrong_refusal_rule | 3 | 0 | 6 | 7 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| graders' exact agreement (independent grades) | 7/12 | 11/12 | 14/14 | 10/12 |
| graders' Cohen's kappa | 0.17 | 0.83 | 1.00 | 0.67 |
| final grade vs judge: exact agreement | 12/12 | 7/12 | 5/12 | 5/12 |
| final grade vs judge: Cohen's kappa | 1.00 | 0.29 | 0.12 | 0.10 |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit)

| Gold dollar figures | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| all in the passages | 12/18 = 67% [44%–84%] | 13/18 = 72% [49%–88%] | 17/18 = 94% [74%–99%] | 17/18 = 94% [74%–99%] |
| not all in the passages | 3/28 = 11% [4%–27%] | 3/28 = 11% [4%–27%] | 1/28 = 4% [1%–18%] | 2/28 = 7% [2%–23%] |

### 9. Cost per question and self-consistency

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| model calls per question (mean, JSON retries included) | 1.0 | 1.0 | 5.0 | 2.0 |
| tokens per question (mean, prompt + completion) | 3536 | 4546 | 19883 | 7397 |
| seconds per question p50 / p95 | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 |
| S2: all valid samples agreed | – | – | 62/102 | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – | – | 33/102 | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – | – | 7/102 | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | 8 / 4 / 0 | 8 / 4 / 0 | 8 / 3 / 1 | 9 / 3 / 0 |

### 10. Paired with M1_qwen3-8b_rag_think-off_b0-v1 (B0): questions right only with the strategy / only with B0, two-sided sign test

| Answers | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|
| Short: numeric | 3 / 2 (p = 1.00, n = 48) | 6 / 2 (p = 0.29, n = 48) | 7 / 2 (p = 0.18, n = 48) |
| Short: entity | 1 / 1 (p = 1.00, n = 10) | 2 / 1 (p = 1.00, n = 10) | 3 / 0 (p = 0.25, n = 10) |
| Summary | 0 / 4 (p = 0.12, n = 12) | 0 / 5 (p = 0.06, n = 12) | 0 / 5 (p = 0.06, n = 12) |
| Refusal & not reported | 10 / 0 (p < 0.01, n = 32) | 4 / 3 (p = 1.00, n = 32) | 2 / 4 (p = 0.69, n = 32) |
| All decided answers | 14 / 7 (p = 0.19, n = 102) | 12 / 11 (p = 1.00, n = 102) | 12 / 11 (p = 1.00, n = 102) |
<!-- GENERATED:END -->

## Interpretation

**Small n, so read the paired table (10) first.** Each strategy is compared with B0 question by question; only
the questions where the two differ count, with a two-sided sign test. With all 102 answers per run decided (the
W2-12 grading round), one question is about one percentage point, and only one difference reaches p < 0.05.

**S1 (few-shot) fixes refusals, not figures.** It refused every question it should (recall 20/20) with the right
rule every time (20/20; B0 14/18) and passed all 13 refusal tests (B0 9/13): right where B0 was wrong on 10
refusal and not-reported questions, never the reverse (p ≈ 0.002), the one significant difference. Four of the
ten are verdicts the graders overturned: three honest "not reported" answers to the not-reported traps, which
the scorer counted as refusals, and RF15, which answered fiscal 2024 and declined the 2026 forecast. Numeric
answers are level with B0 (17/48 vs 16/48). With the graders' labels it over-refuses 5 times (B0 10). It **copies
its examples**: the Citigroup example's net income, $14,306 million, appears as another company's figure in three
multi-company answers (D01, D07, D09). The hallucination check caught D01 only (in D09 the figure matched a passage
number by chance, the check's documented limit), and D07 still counts as right because it names the right
company. Few-shot examples taken from real filings leak figures; examples for the agent should use figures that
cannot be mistaken for real ones, or none.

**S2 (chain of thought + self-consistency) grounds its numbers.** Unsupported numbers fall from 20 of 74
answers (B0) to 4 of 85, the clearest effect in the tables, and period comparisons go from 0/12 to 4/12; numeric
answers 20/48 (6 better, 2 worse than B0, p = 0.29). The vote was unanimous on 62 of 102 questions, a majority
short of unanimity decided 33, and on 7 no decision had a majority (the most common one, 2 of 5, won). The cost is five calls per question: ~20k tokens and 23 s (p50; p95 32 s) per question,
against 2.4 s for B0, which leaves little of the 60 s per question (spec §9) for an agent's own steps. It
refuses less: recall 14/20 (B0 18/20). Of the six it should have refused, the graders label three under-refusals
(an answer given) and three declines with "not reported" instead of a refusal (a wrong rule); its "not reported"
answers to answerable questions add up to 9 over-refusals (B0 10).

**S3 (decomposition) is best on multi-step questions but drops refusals.** Numeric 21/48 (7 better, 2 worse,
p = 0.18), multi-company 8/10 (B0 5/10), period comparisons 3/12. But the research step extracts figures
instead of stopping, and the answer step follows it: recall 13/20 (B0 18/20), with 4 under-refusals and 4 declines
with "not reported" instead of a refusal (the graders' labels), and 5 of 13 refusal tests passed (B0 9).
Two calls per question, ~7.4k tokens, 8 s (p50).

**Where the evidence is in the passages, reasoning helps a lot; where it is not, nothing does.** Of the 46
numeric questions with a scored gold figure, 18 have every gold dollar figure digit for digit in the passages
(table 8). There B0 gets 12/18, mostly losing on arithmetic slips (4 of its 6 misses), and S2 and S3 get 17/18
each, both right on 5 questions B0 missed and wrong on none of B0's (5 / 0, p = 0.06 each). On the 28 whose
figures are not all in the passages every run gets 1-3 right: the bottleneck is retrieval, not prompting. These
packs use oracle company and year filters without W1-05's fiscal-period filter or statements first (ADR-017),
with which a gold source was in the top 5 for every gold question but one; the Week 2 agent should retrieve
that way.

**Summaries, graded by the two graders (W2-12): every strategy is below B0.** S1 4/12, S2 3/12 and S3 3/12
against B0's 8/12 (right only with B0 on 4-5 summaries, never the reverse; p = 0.06-0.12). Most misses are
incomplete summaries (8-9 per run). The graders agree with each other (exact 10-14 of their 12-14 graded rows per
run, kappa 0.67-1.00). The LLM judge does not: it gives 8-9 twos to every run, grades S1-S3's summaries higher
than the graders on 5-7 of 12, and agrees with their final grades on 5-7 of 12 (kappa 0.10-0.29). It neither
separates the strategies nor matches the humans on them, so the human grades decide, and the judge alone should
not be trusted on summaries.

**What carries forward (ADR-018).** No strategy is better than B0 everywhere. For the Week 2 agent: S2-style
reasoning before a figure (the grounding and arithmetic gains) but not five samples per question (latency);
explicit refusal examples or rules in the prompt (S1's gain), without real figures in them; refusal checked
before any extraction step (S3's loss); and W1-05's retrieval default (the largest remaining loss). None of the
strategies helps summaries.

**Caveats.** One frozen prompt per strategy, tuned on 15 development questions; strict guided decoding for
S1-S3 but not B0 (every final answer of all four runs was valid JSON, so it changes no score here; three intermediate calls
hit the 1,024-token reply cap: one S2 sample, D07, decided on the other 4, and two S3 research steps, E12
and G02, whose answer steps were told the research step had failed); the failure labels
are the graders' (W2-12), which differ from the scorer's suggestions mainly on "not reported" answers: one to an
answerable question is labelled an over-refusal (the scorer suggested malformed_output or wrong_line_item), and one
to a question that should be refused a wrong refusal rule (suggested: under-refusal); every answer is decided
since the W2-12 round (`eval/GRADING.md`, "The second round (W2-12)");
a call that fails (the three capped calls) adds no tokens or seconds, so S2's and S3's cost rows are slightly
low; after the PR review the S2 file was re-written from the reply cache with the vote no longer splitting
refusals over the company they mention (0 requests; no answer changed, 4 refusals became unanimous).
