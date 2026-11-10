# Model comparison (W1-08 scorer)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| Short: numeric | 16/48 = 33% [22%–47%] | 6/48 = 12% [6%–25%] | 0/48 = 0% [0%–7%] |
| Short: entity | 5/10 = 50% [24%–76%] | 4/9 = 44% [19%–73%], 1 pending | 5/10 = 50% [24%–76%] |
| Summary | pending (12) | 0/1 = 0% [0%–79%], 11 pending | 0/4 = 0% [0%–49%], 8 pending |
| Refusal & not reported | 21/32 = 66% [48%–80%] | 7/32 = 22% [11%–39%] | 21/32 = 66% [48%–80%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| fact | 8/15 = 53% [30%–75%] | 5/15 = 33% [15%–58%] | 0/15 = 0% [0%–20%] |
| ratio | 7/15 = 47% [25%–70%] | 1/15 = 7% [1%–30%] | 0/15 = 0% [0%–20%] |
| period_comparison | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] |
| multi_turn | 1/6 = 17% [3%–56%] | 0/6 = 0% [0%–39%] | 0/6 = 0% [0%–39%] |
| multi_company | 5/10 = 50% [24%–76%] | 4/9 = 44% [19%–73%], 1 pending | 5/10 = 50% [24%–76%] |
| qualitative | pending (12) | 0/1 = 0% [0%–79%], 11 pending | 0/4 = 0% [0%–49%], 8 pending |
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
| over-refusals | 10 | 3 | 16 |
| under-refusals | 2 | 2 | 5 |

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
| unit_confusion | 1 | 1 | 0 |
| arithmetic_slip | 8 | 3 | 2 |
| invented_number | 9 | 9 | 27 |
| wrong_line_item | 8 | 11 | 9 |
| ignored_not_reported | 1 | 0 | 1 |
| wrong_company | 5 | 2 | 5 |
| over_refusal | 10 | 3 | 16 |
| under_refusal | 2 | 2 | 5 |
| malformed_output | 0 | 38 | 0 |
| (not labelled yet) | 4 | 4 | 3 |

### 7. Summary answers: human grades vs LLM judge

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| exact agreement | – | – | – |
| Cohen's kappa | – | – | – |
<!-- GENERATED:END -->

## Interpretation

(written for W1-15)
