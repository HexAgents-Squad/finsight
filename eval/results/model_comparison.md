# Model comparison (W1-08 scorer)

<!-- GENERATED:START (eval/score_answers.py) -->
### 1. Accuracy by answer type (correct/n, Wilson 95% interval)

| Answer type | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| Short: numeric | 16/48 = 33% [22%–47%] | 6/48 = 12% [6%–25%] | 0/48 = 0% [0%–7%] |
| Short: entity | 5/10 = 50% [24%–76%] | 4/10 = 40% [17%–69%] | 5/10 = 50% [24%–76%] |
| Summary | 8/12 = 67% [39%–86%] | 5/12 = 42% [19%–68%] | 0/12 = 0% [0%–24%] |
| Refusal & not reported | 21/32 = 66% [48%–80%] | 7/32 = 22% [11%–39%] | 21/32 = 66% [48%–80%] |

### 2. Accuracy by gold category and refusal-test kind

| Category | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| fact | 8/15 = 53% [30%–75%] | 5/15 = 33% [15%–58%] | 0/15 = 0% [0%–20%] |
| ratio | 7/15 = 47% [25%–70%] | 1/15 = 7% [1%–30%] | 0/15 = 0% [0%–20%] |
| period_comparison | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] | 0/12 = 0% [0%–24%] |
| multi_turn | 1/6 = 17% [3%–56%] | 0/6 = 0% [0%–39%] | 0/6 = 0% [0%–39%] |
| multi_company | 5/10 = 50% [24%–76%] | 4/10 = 40% [17%–69%] | 5/10 = 50% [24%–76%] |
| qualitative | 8/12 = 67% [39%–86%] | 5/12 = 42% [19%–68%] | 0/12 = 0% [0%–24%] |
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
| over-refusals | 10 | 5 | 20 |
| under-refusals | 3 | 3 | 6 |

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
| unit_confusion | 0 | 3 | 0 |
| arithmetic_slip | 10 | 4 | 2 |
| invented_number | 8 | 14 | 28 |
| wrong_line_item | 7 | 8 | 3 |
| ignored_not_reported | 1 | 0 | 1 |
| wrong_company | 5 | 3 | 5 |
| over_refusal | 10 | 5 | 20 |
| under_refusal | 3 | 3 | 6 |
| malformed_output | 1 | 31 | 0 |
| incomplete_summary | 4 | 6 | 8 |
| wrong_refusal_rule | 3 | 3 | 3 |

### 7. Human grades: the two graders, and the LLM judge on summaries

| Measure | M1_qwen3-8b_rag_think-off_b0-v1 | M2_llama3.2-3b_rag_think-off_b0-v1 | M3_qwen3-8b_closed_think-off_b0-v1 |
|---|---|---|---|
| graders' exact agreement (independent grades) | 7/12 | 7/12 | 6/8 |
| graders' Cohen's kappa | 0.17 | 0.39 | 0.54 |
| final grade vs judge: exact agreement | 12/12 | 9/11 | 4/8 |
| final grade vs judge: Cohen's kappa | 1.00 | 0.70 | 0.20 |
<!-- GENERATED:END -->

## Interpretation

(written for W1-15)
