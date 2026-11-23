# Trace dashboard: run W3-09-demo-a5

10 traced question(s). Source: Langfuse traces tagged `run:W3-09-demo-a5` (scripts/trace_dashboard.py).

| Question | Verdict | Seconds | Steps | Tool calls | Tokens | Failure |
|---|---|---|---|---|---|---|
| demo-01 | verified | 5.4 | 4 | 1 | 3,039 |  |
| demo-02 | verified | 1.2 | 4 | 3 | 3,224 |  |
| demo-03 | unverified | 16.8 | 7 | 4 | 6,016 | could not verify: operation: the question asks for cagr or growth, but no calculation com… |
| demo-04 | verified | 1.6 | 4 | 3 | 3,207 |  |
| demo-05 | verified | 10.4 | 4 | 1 | 4,124 |  |
| demo-06 | verified | 3.7 | 4 | 1 | 3,039 |  |
| demo-07 | verified | 3.9 | 4 | 1 | 3,079 |  |
| demo-08 | refused | 0.0 | 3 | 0 | 2,103 |  |
| demo-09 | verified | 4.4 | 4 | 1 | 3,099 |  |
| demo-10 | verified | 1.6 | 4 | 2 | 3,050 |  |

## Totals

- Verdicts: refused 1, unverified 1, verified 8
- Seconds per question: p50 3.7, p95 16.8, max 16.8
- Mean per question: 4.2 node runs, 1.7 tool calls, 3,398.0 tokens
- Failures by reason: could not verify (1)
