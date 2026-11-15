# FinSight · Week 1 Results Note (8 – 14 Nov 2026)

Team HexAgents_Squad (Track B · API-side) · Mentor: Dr. Gunjan Balde · Prepared by Alok Srivastava (team
representative), 16 Nov 2026. The tables are generated from the committed results by `scripts/week1_notes.py`;
the detailed companion is `docs/report/week1_detailed_results.md`.

**Week 1 exit gate met:** the model comparison you asked for (delivered Wed 11 Nov), the prompting-baseline
table (B0 zero-shot and S1/S2/S3 on Qwen3-8B, with failure modes) and the retrieval table (chunking × retriever
× rerank × filters). All 16 Week 1 tickets are delivered: 15 in 16 reviewed, merged pull requests (#17–#32, CI green)
and this note, the last; every number below traces back to a committed run file.

<!-- GENERATED:START (scripts/week1_notes.py) -->
**Model comparison** (all 80 gold questions + 22 refusal tests, the same retrieved passages for every model; `eval/results/model_comparison.md`)

| Answer type | Qwen3-8B, passages (B0) | Llama-3.2-3B, passages | Qwen3-8B, closed book |
|---|---|---|---|
| Short: numeric | 16/48 (33%) | 6/48 (12%) | 0/48 (0%) |
| Short: entity | 5/10 (50%) | 4/10 (40%) | 5/10 (50%) |
| Summary | 8/12 (67%) | 5/12 (42%) | 0/12 (0%) |
| Refusal & not reported | 21/32 (66%) | 7/32 (22%) | 21/32 (66%) |
| Answers with a number not in the passages | 20/74 | 15/66 | 45/70 |

**Prompting baselines on Qwen3-8B** (same passages and questions; B0 is the zero-shot run above; `eval/results/prompting_baselines.md`)

| Answer type | Qwen3-8B, passages (B0) | S1 few-shot | S2 reasoning + 5-sample vote | S3 decomposition |
|---|---|---|---|---|
| Short: numeric | 16/48 (33%) | 17/48 (35%) | 20/48 (42%) | 21/48 (44%) |
| Short: entity | 5/10 (50%) | 5/10 (50%) | 6/8 (75%), 2 pending | 8/10 (80%) |
| Summary | 8/12 (67%) | pending (12) | pending (12) | pending (12) |
| Refusal & not reported | 21/32 (66%) | 27/32 (84%) | 22/32 (69%) | 18/32 (56%) |
| Answers with a number not in the passages | 20/74 | 15/72 | 4/85 | 19/88 |
| Right only with the strategy / only with B0 (sign test) | – | 10 / 3 (p = 0.09, n = 90) | 12 / 5 (p = 0.14, n = 88) | 11 / 6 (p = 0.33, n = 90) |
| Seconds per question, median / 95th percentile | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 |

**Retrieval** (73 gold questions with sources; recall@k checks the right filing section, the answer figure the exact passage; `eval/results/retrieval_summary.md`)

| Setting (C3 table-aware chunks) | recall@5 | recall@10 | MRR | answer figure in top 5 | s/query |
|---|---|---|---|---|---|
| no filters | 0.336 | 0.548 | 0.230 | 12/59 | 0.53 |
| company + year (W1-04 index) | 0.774 | 0.870 | 0.550 | 23/59 | 0.25 |
| company, year, period read from the question | 0.890 | 0.959 | 0.633 | 25/59 | 0.55 |
| + financial statements first (default, ADR-017) | 0.986 | 1.000 | 0.975 | 31/59 | 0.97 |
| + cross-encoder reranker (opt-in) | 1.000 | 1.000 | 0.986 | 42/59 | 9.46 |
<!-- GENERATED:END -->

**What the numbers say**
- **Retrieval decides most answers.** Where every gold figure is in the passages, Qwen3-8B with step-by-step
  reasoning (S2) or decomposition (S3) gets 17 of 18 numeric questions right (B0 12 of 18); where the figures are
  missing, every model and prompt fails. Reading company, year and fiscal period from the question and putting the
  financial statements first raises recall@5 from 0.774 to 0.986 (ADR-017), the retrieval default for Week 2.
- **Qwen3-8B with passages is the model to build on:** best or tied on every answer type; closed book it gives a
  number it cannot support in 45 of 70 answers, so the agent never answers figures without retrieval or the XBRL tool.
- **No prompting strategy wins everywhere** (paired sign test against B0): few-shot fixes refusals (the only
  significant gain, 6 better / 0 worse, p = 0.03) but copied an example's figure into three answers;
  self-consistency cuts ungrounded numbers from 20 of 74 answers to 4 of 85 at 5 calls and ~23 s per question;
  decomposition is best on numbers and multi-company questions but under-refuses (ADR-018).
- **Cleaner text and a better embedding model** (W1-04, W1-16): 16,817 page-furniture lines removed and the
  index rebuilt with bge-small-en-v1.5; dense recall@5 with filters 0.630 → 0.788 (p = 0.011).

**Also delivered in Week 1:** the hosted Qwen3-8B endpoint behind one request queue, retries and a reply cache
(W1-01/02); an endpoint probe deciding JSON-schema output, tool calling and the latency budget (W1-03, ADR-016);
a 15-question development set kept apart from gold (W1-06); the answer contract and critic checks (W1-07); the
answer scorer with hallucination checks and a failure taxonomy (W1-08); two-grader human grading (W1-14); the
34 safety tests verified by a second team member (W1-10).

**Open items:** the 36 strategy summaries and 124 wrong strategy answers wait for the graders' second round;
smoke tests on 4 more laptops and Chandra Shekhar's independent WP2 spot check (W1-10).

**Questions for you:** (1) the agent must answer within 60 s: may we keep the reranker and 5-sample voting as
measured options only? (2) the graders agreed on 7 of 12 Qwen summaries (kappa 0.17): is a third grader or a
written rubric revision expected? (3) Langfuse tracing: cloud free tier or self-hosted Docker?

**Week 2 (16 – 21 Nov), agentic layer:** a LangGraph planner → executor → critic agent on the four tools, with
memory across turns, JSON-schema outputs, step and token caps, Langfuse traces, a FastAPI service and a Streamlit
UI; exit gate: an end-to-end demo on 10 queries with a trace dashboard (`docs/week2_plan.txt`).
