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
| cached replies | 0/15 | 0/15 | 0/15 | 0/15 |
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
| S2: a bare majority decided (3/5 or less) | – | – | 5/15 | – |
<!-- GENERATED:END -->

## Interpretation

(written for W1-09)
