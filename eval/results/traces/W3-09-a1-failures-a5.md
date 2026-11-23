# Trace dashboard: run W3-09-a1-failures-a5

16 traced question(s). Source: Langfuse traces tagged `run:W3-09-a1-failures-a5` (scripts/trace_dashboard.py).

| Question | Verdict | Seconds | Steps | Tool calls | Tokens | Failure |
|---|---|---|---|---|---|---|
| A06 | verified | 1.3 | 4 | 1 | 3,056 |  |
| B04 | verified | 27.5 | 4 | 4 | 4,468 |  |
| B08 | verified | 2.2 | 4 | 6 | 5,874 |  |
| B14 | unverified | 1.3 | 7 | 4 | 6,177 | could not verify: segment: the question is about a segment (Amazon Web Services), but the… |
| B15 | verified | 0.1 | 4 | 3 | 3,220 |  |
| C08 | verified | 0.1 | 7 | 3 | 4,404 |  |
| RF10 | refused | 0.0 | 3 | 0 | 2,121 |  |
| RF13 | refused | 0.0 | 3 | 0 | 2,102 |  |
| RF14 | verified | 0.1 | 4 | 1 | 2,994 |  |
| E01 | unverified | 0.8 | 7 | 1 | 9,095 | could not verify: summary: the cited passage gives the size of the change in cloud revenu… |
| E02 | verified | 1.4 | 7 | 1 | 6,105 |  |
| E03 | verified | 0.6 | 4 | 1 | 4,106 |  |
| E05 | verified | 0.7 | 4 | 1 | 4,952 |  |
| E06 | verified | 0.6 | 4 | 1 | 4,536 |  |
| E07 | verified | 0.9 | 4 | 1 | 4,865 |  |
| E10 | unverified | 0.9 | 7 | 1 | 6,355 | could not verify: summary: no cited passage states the change in revenue the question ask… |

## Totals

- Verdicts: refused 2, unverified 3, verified 11
- Seconds per question: p50 0.7, p95 27.5, max 27.5
- Mean per question: 4.8 node runs, 1.8 tool calls, 4,651.9 tokens
- Failures by reason: could not verify (3)
