# Prompting baselines on the development set (W1-09 tuning; not a result)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| Short: numeric | 5/9 = 56% [27%–81%] | 7/9 = 78% [45%–94%] | 6/9 = 67% [35%–88%] | 5/9 = 56% [27%–81%] |
| Short: entity | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] |
| Summary | pending (2) | pending (2) | pending (2) | pending (2) |
| Refusal & not reported | 1/2 = 50% [9%–91%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| fact | 2/3 = 67% [21%–94%] | 2/3 = 67% [21%–94%] | 3/3 = 100% [44%–100%] | 2/3 = 67% [21%–94%] |
| ratio | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] |
| period_comparison | 0/2 = 0% [0%–66%] | 1/2 = 50% [9%–91%] | 0/2 = 0% [0%–66%] | 0/2 = 0% [0%–66%] |
| multi_turn | 1/2 = 50% [9%–91%] | 2/2 = 100% [34%–100%] | 1/2 = 50% [9%–91%] | 1/2 = 50% [9%–91%] |
| multi_company | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] |
| qualitative | pending (2) | pending (2) | pending (2) | pending (2) |
| refuse | 1/2 = 50% [9%–91%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] | 2/2 = 100% [34%–100%] |
| refusal tests: refusal | – | – | – | – |
| refusal tests: partial | – | – | – | – |
| refusal tests: control | – | – | – | – |

### 3. Hallucination and grounding checks (answers with a reply, not refused)

| Check | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| unsupported numbers | 4/13 | 3/14 | 2/14 | 4/14 |
| wrong fiscal year | 0/13 | 0/14 | 0/14 | 0/14 |
| wrong company | 0/13 | 0/14 | 0/14 | 0/14 |
| cited a passage it was not given (closed-book: any citation) | 0/13 | 0/14 | 0/14 | 0/14 |
| value given for a 'not reported' trap | 0/1 | 0/1 | 0/1 | 0/1 |

### 4. Refusals (G01-G10 + 22 refusal tests)

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| precision (refused and should) | 1/2 | 1/1 | 1/1 | 1/1 |
| recall (should refuse and did) | 1/1 | 1/1 | 1/1 | 1/1 |
| right rule among correct refusals | 1/1 | 1/1 | 1/1 | 1/1 |
| over-refusals | 1 | 0 | 0 | 0 |
| under-refusals | 0 | 0 | 0 | 0 |

### 5. Output validity, tokens, latency and human corrections

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| valid B0 JSON | 15/15 | 15/15 | 15/15 | 15/15 |
| JSON retries | 0 | 0 | 0 | 0 |
| no reply (errors) | 0 | 0 | 0 | 0 |
| prompt tokens (mean) | 3973 | 4990 | 21586 | 8185 |
| completion tokens (mean) | 86 | 80 | 929 | 329 |
| latency p50 / p95 (s, answered calls) | 2.2 / 2.6 | 2.1 / 2.6 | 21.3 / 26.5 | 6.7 / 12.5 |
| cached replies | 0/15 | 0/15 | 15/15 | 0/15 |
| automatic verdicts overturned by graders | 0 | 0 | 0 | 0 |

### 6. Failure labels of wrong answers (confirmed in the grading sheet, else suggested)

| Label | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| unit_confusion | 1 | 0 | 1 | 1 |
| arithmetic_slip | 1 | 0 | 1 | 1 |
| invented_number | 2 | 2 | 0 | 2 |
| wrong_line_item | 0 | 0 | 1 | 0 |
| over_refusal | 1 | 0 | 0 | 0 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| graders' exact agreement (independent grades) | – | – | – | – |
| graders' Cohen's kappa | – | – | – | – |
| final grade vs judge: exact agreement | – | – | – | – |
| final grade vs judge: Cohen's kappa | – | – | – | – |


### 8. Short numeric answers by whether the gold figures were in the passages (digit for digit)

| Gold dollar figures | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| all in the passages | 4/5 = 80% [38%–96%] | 5/5 = 100% [57%–100%] | 4/5 = 80% [38%–96%] | 4/5 = 80% [38%–96%] |
| not all in the passages | 1/4 = 25% [5%–70%] | 2/4 = 50% [15%–85%] | 2/4 = 50% [15%–85%] | 1/4 = 25% [5%–70%] |

### 9. Cost per question and self-consistency

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|---|
| model calls per question (mean, JSON retries included) | 1.0 | 1.0 | 5.0 | 2.0 |
| tokens per question (mean, prompt + completion) | 4060 | 5071 | 22515 | 8514 |
| seconds per question p50 / p95 | 2.2 / 2.6 | 2.1 / 2.6 | 21.3 / 26.5 | 6.7 / 12.5 |
| S2: all valid samples agreed | – | – | 9/15 | – |
| S2: a majority short of all (e.g. 3/5, 4/5) | – | – | 5/15 | – |
| S2: no majority, the most common decision won (e.g. 2/5; ties to the first) | – | – | 1/15 | – |
| LLM judge on summaries, grades 2 / 1 / 0 (second opinion; humans decide) | – | – | – | – |

### 10. Paired with M1_qwen3-8b_rag_think-off_b0-v1 (B0): questions right only with the strategy / only with B0, two-sided sign test

| Answers | S1_qwen3-8b_rag_think-off_s1-v1 | S2_qwen3-8b_rag_think-off_s2-v1 | S3_qwen3-8b_rag_think-off_s3-v1 |
|---|---|---|---|
| Short: numeric | 2 / 0 (p = 0.50, n = 9) | 1 / 0 (p = 1.00, n = 9) | 0 / 0 (p = 1.00, n = 9) |
| Short: entity | 0 / 0 (p = 1.00, n = 2) | 0 / 0 (p = 1.00, n = 2) | 0 / 0 (p = 1.00, n = 2) |
| Summary | – | – | – |
| Refusal & not reported | 1 / 0 (p = 1.00, n = 2) | 1 / 0 (p = 1.00, n = 2) | 1 / 0 (p = 1.00, n = 2) |
| All decided answers | 3 / 0 (p = 0.25, n = 13) | 2 / 0 (p = 0.50, n = 13) | 1 / 0 (p = 1.00, n = 13) |
<!-- GENERATED:END -->

## Interpretation

Tuning runs for the W1-09 prompts on the 15 development questions (`eval/dev_set.jsonl`, development pack
`eval/results/context_pack_dev.jsonl`), not a result: 13 answers are decided automatically (the two summaries
are not graded), so one question is 8 points. These are the runs of the third and last tuning pass, with the
prompts as frozen (s1-v1, s2-v1, s3-v1); M1 is B0 on the same questions. S1's four examples are development
questions (dev-A03, dev-C01, dev-G02, dev-G01), so its score includes its own examples: 9 of 9 on the other 11.
The tuning history (pass 1: B0 8, S1 11, S2 9, S3 9 of 13; pass 2: S2 8, S3 9; pass 3: S2 10, S3 9) and why
each change was made are in ADR-018. Two development questions (dev-A03, dev-C01) lack their gold figure in the
pack, so "not reported" is the honest answer there although it scores as wrong.
