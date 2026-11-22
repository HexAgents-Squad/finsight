# FinSight · Week 1 Detailed Results (8 – 14 Nov 2026)

Companion to the one-page `docs/report/week1_results_note.md`. The tables between the markers are generated
from the committed results by `scripts/week1_notes.py`; the full reports they come from are
`eval/results/model_comparison.md`, `prompting_baselines.md`, `retrieval_summary.md`, `embedding_comparison.md`,
`endpoint_probe.md` and `furniture_removal.md`, and every design decision is an ADR in `docs/decisions.md`.

## How Week 1 was run
- **Two sprints** (`docs/week1_plan.txt`): Sprint A (10–11 Nov) built the evaluation harness and the model
  comparison the mentor asked for on 7 Nov; Sprint B (12–14 Nov) reused it for the prompting baselines, the
  retrieval experiments, the re-index and the Week 0 carry-overs.
- **One harness for every answer number:** the same retrieved passages for every model (the context pack,
  W1-12), one runner with frozen prompts (W1-13, W1-09), one scorer (W1-08) with automatic checks, a blind
  two-grader sheet for what needs a person (W1-14), and an LLM judge as a second opinion only.
- **No tuning on the test set:** prompts were tuned on a separate 15-question development set (W1-06) and
  frozen, pinned by sha256 and tagged before any gold run; gold-v1 stayed frozen.
- **The shared key was used politely:** one request at a time, 1 s apart, retries with back-off, and a reply cache
  so every re-run is identical and free (W1-02).
- **Engineering practice:** every ticket a pull request with tests (offline, fake clients), ruff, an independent
  review and CI; result files regenerate byte for byte from committed inputs.

<!-- GENERATED:START (scripts/week1_notes.py) -->
### Question sets

| Set | Items | Composition |
|---|---|---|
| gold-v1 (frozen) | 80 | fact 15, ratio 15, period_comparison 12, multi_company 10, qualitative 12, multi_turn 6, refuse 10 |
| development set (tuning only) | 15 | fact 3, ratio 2, period_comparison 2, multi_company 2, qualitative 2, multi_turn 2, refuse 2 |
| refusal tests | 22 | refusal 13, partial 2, control 7 |
| prompt-injection tests | 12 | every test verified by a second team member |

### Model comparison by answer type

| Answer type | Qwen3-8B, passages (B0) | Llama-3.2-3B, passages | Qwen3-8B, closed book |
|---|---|---|---|
| Short: numeric | 16/48 (33%) | 6/48 (12%) | 0/48 (0%) |
| Short: entity | 5/10 (50%) | 4/10 (40%) | 5/10 (50%) |
| Summary | 8/12 (67%) | 5/12 (42%) | 0/12 (0%) |
| Refusal & not reported | 21/32 (66%) | 7/32 (22%) | 21/32 (66%) |
| Answers with a number not in the passages | 20/74 | 15/66 | 45/70 |

### Prompting baselines by answer type

| Answer type | Qwen3-8B, passages (B0) | S1 few-shot | S2 reasoning + 5-sample vote | S3 decomposition |
|---|---|---|---|---|
| Short: numeric | 16/48 (33%) | 17/48 (35%) | 20/48 (42%) | 21/48 (44%) |
| Short: entity | 5/10 (50%) | 5/10 (50%) | 6/10 (60%) | 8/10 (80%) |
| Summary | 8/12 (67%) | 4/12 (33%) | 3/12 (25%) | 3/12 (25%) |
| Refusal & not reported | 21/32 (66%) | 31/32 (97%) | 22/32 (69%) | 19/32 (59%) |
| Answers with a number not in the passages | 20/74 | 15/72 | 4/85 | 19/88 |
| Right only with the strategy / only with B0 (sign test) | – | 14 / 7 (p = 0.19, n = 102) | 12 / 11 (p = 1.00, n = 102) | 12 / 11 (p = 1.00, n = 102) |
| Seconds per question, median / 95th percentile | 2.4 / 3.6 | 2.1 / 3.1 | 22.9 / 31.7 | 8.2 / 16.1 |

### Retrieval settings

| Setting (C3 table-aware chunks) | recall@5 | recall@10 | MRR | answer figure in top 5 | s/query |
|---|---|---|---|---|---|
| no filters | 0.336 | 0.548 | 0.230 | 12/59 | 0.53 |
| company + year (W1-04 index) | 0.774 | 0.870 | 0.550 | 23/59 | 0.25 |
| company, year, period read from the question | 0.890 | 0.959 | 0.633 | 25/59 | 0.55 |
| + financial statements first (default, ADR-017) | 0.986 | 1.000 | 0.975 | 31/59 | 0.97 |
| + cross-encoder reranker (opt-in) | 1.000 | 1.000 | 0.986 | 42/59 | 9.46 |

### Embedding model (W1-16): MiniLM (local) against bge-small (course endpoint), before W1-04

| Retriever, filters (C3) | MiniLM recall@5 | bge-small recall@5 | better / worse / same | sign test p |
|---|---|---|---|---|
| dense, none | 0.349 | 0.555 | 23 / 5 / 45 | 0.0009 |
| dense, company+year | 0.630 | 0.788 | 18 / 5 / 50 | 0.0106 |
| hybrid, none | 0.301 | 0.384 | 13 / 7 / 53 | 0.2632 |
| hybrid, company+year | 0.747 | 0.760 | 10 / 9 / 54 | 1.0 |

### Decisions recorded in Week 1 (docs/decisions.md)

| ADR | Date | Decision (first sentence) | Status |
|---|---|---|---|
| ADR-016 | 14 Nov 2026 | Endpoint usage, from `scripts/probe_endpoint.py` (committed run in `eval/results/endpoint_probe.{json,md}`, 33 requests; two earlier development runs of the... | Accepted |
| ADR-017 | 15 Nov 2026 | Retrieval default for the agent. | Accepted |
| ADR-018 | 15 Nov 2026 | Prompting baselines on Qwen3-8B, all on the W1-12 context pack (102 questions, oracle filters) and scored by W1-08, against B0 = the frozen zero-shot run M1... | Accepted |
| ADR-019 | 13 Nov 2026 | Model-comparison setup and what it decides. | Accepted |
| ADR-020 | 14 Nov 2026 | Development set (`eval/dev_set.jsonl`, 15 items, split A3 B2 C2 D2 E2 F2 G2 with one 'not reported' trap and one refusal) in the gold format with `dev-` ids,... | Accepted |
| ADR-021 | 14 Nov 2026 | The dense index moves to bge-small-en-v1.5 on the course's embedding endpoint (EMBED_BACKEND=openai, EMBED_CA_CERT, EMBED_BATCH_SIZE=16, EMBED_MAX_INPUT_TOKE... | Accepted |
| ADR-022 | 14 Nov 2026 | The answer contract and the critic's checks. | Accepted |
| ADR-023 | 14 Nov 2026 | Page furniture is removed from section text (ADR-015 item 6) in `ingest/sections.py`, per filing, after the signature cut. | Accepted |
<!-- GENERATED:END -->

## Ticket by ticket
| Ticket | What was delivered | Evidence |
|---|---|---|
| W1-01 | Hosted Qwen3-8B endpoint: group CA certificate, model id, thinking off, smoke test | `scripts/smoke_llm.py`, PR #17 |
| W1-02 | One request at a time per endpoint, 2/4/8 s retries, reply cache, prompt-size check | `core/llm.py`, PR #20 |
| W1-03 | Endpoint probe: JSON schema output, tool calling, thinking switch ignored, limits, latency | ADR-016, `endpoint_probe.md`, PR #25 |
| W1-04 | Page furniture removed (16,817 lines), all indexes rebuilt with bge-small | ADR-023, `furniture_removal.md`, PR #29 |
| W1-05 | Question-derived filters, period filter, statements first, cross-encoder reranker | ADR-017, `retrieval_summary.md`, PR #31 |
| W1-06 | 15-question development set, no overlap with gold, verified | ADR-020, `eval/dev_set.jsonl`, PR #26 |
| W1-07 | Answer contract and the critic's seven checks | ADR-022, `agent/contract.py`, PR #28 |
| W1-08 | Scorer: short / summary / refusal answers, hallucination checks, failure taxonomy | `eval/score_answers.py`, PR #19 |
| W1-09 | Prompting baselines S1/S2/S3 against B0 | ADR-018, `prompting_baselines.md`, PR #32 |
| W1-10 | Week 0 carry-overs: safety tests verified, spot check (independent review pending), smoke tests 1 of 5 (rest in W2-12) | `docs/smoke_tests.md`, PR #30 |
| W1-11 | Week 1 notes (this), Week 2 plan and tickets | `docs/week2_plan.txt` |
| W1-12 | Context pack: the same passages for every model | `eval/context_pack.py`, PR #18 |
| W1-13 | Comparison runner and the frozen zero-shot prompt B0 | `eval/run_models.py`, PR #21 |
| W1-14 | Blind two-grader grading and failure labels | `eval/grading.py`, PRs #22, #23 |
| W1-15 | Model comparison report to the mentor (due Wed 11 Nov, sent Fri 13 Nov) | ADR-019, `model_comparison.md`, PR #24 |
| W1-16 | Course embedding endpoint (bge-small) measured against MiniLM | ADR-021, `embedding_comparison.md`, PR #27 |

## Failure modes seen in Week 1 (labels from the graders and the scorer)
- **Qwen3-8B with passages** went wrong through arithmetic slips (10), over-refusals (10), invented numbers (8)
  and wrong line items (7); closed book, invented numbers lead (28 of its wrong answers).
- **Arithmetic slips** are the main loss when the evidence is present (B0 4 of its 6 misses on such questions);
  reasoning (S2) and decomposition (S3) remove them.
- **Over-refusal** (B0 10) and **under-refusal** (S2 3, S3 4; each also declined 3-4 questions it should refuse
  with "not reported" instead, the graders' labels in W2-12): refusal must be decided before any extraction step.
- **Format:** Llama-3.2-3B returned valid JSON for 72 of 102 questions; every Qwen3-8B B0 answer was valid JSON,
  and with guided decoding every strategy answer was valid too.
- **Copying from examples:** few-shot examples taken from real filings leaked a figure into three answers.
- **Units:** Qwen writes "$47.52 billion" as a wrong plain-dollar integer under every prompt; the agent's
  calculator and the critic must own units.

## Limits stated with the numbers
- Small cells: the answer tables split 102 questions into cells of 8-48, so one question moves a cell by 2 to 12
  points (retrieval: 73 questions, about 1.4 points); differences are judged question by question with a sign
  test, and only differences with p < 0.05 are called significant.
- The comparison and baselines used oracle company and year filters; W1-05 showed question-derived filters
  equal them on the gold set, but real questions are phrased more freely.
- Summary grades: two graders agreed on 7 of 12 Qwen summaries (kappa 0.17); the strategy summaries are not graded
  yet.
- One frozen prompt per strategy, tuned on 15 development questions.
