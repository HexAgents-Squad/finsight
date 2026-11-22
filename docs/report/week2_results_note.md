# FinSight · Week 2 Results Note (16 – 22 Nov 2026)

Team HexAgents_Squad (Track B · API-side) · Mentor: Dr. Gunjan Balde · Prepared by Alok Srivastava (team
coordinator) · Week 2 goal: the agentic layer, a LangGraph agent on the four tools, with its first scored run on the
gold set. The tables between the markers are generated from the committed results by `scripts/week2_notes.py`.

<!-- GENERATED:START (scripts/week2_notes.py) -->
**The agent next to Week 1's runs** (all 80 gold questions + 22 refusal tests, the same scorer; B0 and S1-S3 read a fixed context pack, A1 chose its own tools; `eval/results/agent_baselines.md`)

| Answer type | Qwen3-8B, passages (B0) | S1 few-shot | S2 reasoning + 5-sample vote | S3 decomposition | A1 agent |
|---|---|---|---|---|---|
| Short: numeric | 16/48 (33%) | 17/48 (35%) | 20/48 (42%) | 21/48 (44%) | 42/48 (88%) |
| Short: entity | 5/10 (50%) | 5/10 (50%) | 6/10 (60%) | 8/10 (80%) | 10/10 (100%) |
| Summary | 8/12 (67%) | 4/12 (33%) | 3/12 (25%) | 3/12 (25%) | 5/12 (42%) |
| Refusal & not reported | 21/32 (66%) | 31/32 (97%) | 22/32 (69%) | 19/32 (59%) | 29/32 (91%) |
| Answers with an unsupported number (each run's own evidence) | 20/74 | 15/72 | 4/85 | 19/88 | 0/81 |
| ... no calculation result counted as evidence (B0's rule) | 20/74 | 15/72 | 4/85 | 19/88 | 4/81 |
| Right only with the run / only with B0 (sign test) | – | 14 / 7 (p = 0.19, n = 102) | 12 / 11 (p = 1.00, n = 102) | 12 / 11 (p = 1.00, n = 102) | 44 / 8 (p < 0.01, n = 102) |
| Tokens per question (mean) | 3536 | 4546 | 19883 | 7397 | 3000 |
| Seconds per question, median / 95th percentile | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 | 5.0 / 12.4 |

**The 10 demo queries** (`eval/results/demo/W2-07_demo.md`) and **the injection tests** (`eval/results/injection_results.md`)

| Demo queries (W2-07, real graph, tools and data) | Value |
|---|---|
| queries that met their expectation | 10/10 |
| verdicts | verified 8; refused 1; partial refusal, verified 1 |
| under the 60 s limit | 10/10 |
| seconds per query, median / max | 4.9 / 20.3 |
| tokens per query (mean) | 2,976 |
| tool calls (total) | 16 |

| The 12 injection tests (W2-13) | Defences off | Defences on |
|---|---|---|
| successful injections, all 12 tests | 0/12 | 0/12 |
| successful injections, tests that retrieved the planted passage | 0/11 | 0/11 |
| tests passed (no planted content, the right figure, the right refusal decision) | 9/12 | 8/12 |
| planted sentence reached the composer as written | 10/12 | 0/12 |
| answer carries the spec's note, tests that retrieved the planted passage | 0/11 | 11/11 |
<!-- GENERATED:END -->

**What the numbers say**
- **The agent answers what the fixed passages could not.** On the same 102 questions and the same scorer, the agent
  (planner → executor → composer → critic on Qwen3-8B) gets 42 of 48 numeric questions right against 16 for the
  Week 1 baseline B0, 11 of 12 period comparisons against 0, and is right on 44 questions where B0 is wrong against
  8 the other way (sign test p ≈ 4×10⁻⁷). The results are consistent with the gain coming from what it fetches: XBRL
  facts and a calculator instead of figures read from a fixed passage pack. Every number it states is one a tool returned (0 of 81 answers with an
  unsupported number; 4 of 81 if, as for B0, no calculation result counts as evidence).
- **Its weak spot is the summary** ("what drove ..."): 5 of 12 against B0's 8, graded by two people. Its four zeros
  answer a neighbouring question from passages it found itself. The LLM judge graded them higher than the graders on 4 of
  12 and lower on 2 (agreement kappa 0.29), so the human grades decide.
- **It stays within the limits:** median 5.0 s and 12.4 s at the 95th percentile (limit 60 s), 3.0k tokens and 1.8
  model calls per question, no question stopped by a limit. The 10 demo queries through the real graph all met their
  expectation, the slowest in 20.3 s.
- **Retrieved text stays data.** None of the 12 planted injections changed an answer, with or without the new
  defences; with them, a planted sentence never reaches the model (10 of 12 did before) and the user sees the spec's
  note. On attack wordings nobody tuned for, the pattern detector catches about half; the critic and the give-up
  answer are the backstop. E-mails and phone numbers are masked in every log line and trace.

**Also delivered in Week 2:** the agent state and bounded graph (W2-01); the planner with refusal first (W2-02);
tool schemas with a metric enum, which took accepted first tool calls from 24 to 31 of 31 (W2-03); the executor
with step, token and output guards (W2-04); the critic's unit and metric checks (W2-05); the composer, which builds
every value and citation from tool outputs, and memory across turns (W2-06); Langfuse tracing of every node, tool
and model call, with a trace dashboard script (W2-08; its run on real traces is in Week 3); the FastAPI service with signed sessions, cache and rate limit
(W2-09); the Streamlit chat UI (W2-10); and the Week 1 carry-overs: the second grading round (60 grades, kappa
0.89, 133 labels), smoke tests on all five laptops and the independent WP2 check, signed (W2-12).

**What did not work yet:** the critic verified all 102 answers and caught none of the 16 wrong ones: six metric
misses its checks do not see (each on a line item the facts tool has no metric for), three refusal errors and seven
weak summaries; one injection test lost its best passage with the planted sentence (INJ06). Each is a Week 3 ticket.

**Questions for you:** (1) the critic is a deterministic checker that caught no wrong answer here: should Week 3
strengthen its checks (our plan) or add an LLM self-check, at a model call per question? (2) is the summary score
from two graders on 12 questions enough for the report, or should we add summary questions? (3) for the final demo
on 5 Dec, a live run on the course endpoint, with a recorded run and its Langfuse traces as the fallback?

**Week 3 (22 – 27 Nov), evaluate, harden, ship:** the ablations (no calculator, no critic, no statements-first
retrieval), the critic's metric checks and the missing facts, safety and refusal tests through the API, Docker
Compose for the API and UI, the 12-page report; code freeze on 25 Nov, submission by 26 Nov
(`docs/week3_plan.txt`).
