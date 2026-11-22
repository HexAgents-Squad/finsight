# FinSight · Week 2 Detailed Results (16 – 22 Nov 2026)

Companion to the one-page `docs/report/week2_results_note.md`. The tables between the markers are generated from
the committed results by `scripts/week2_notes.py`; the full reports they come from are
`eval/results/agent_baselines.md` (the agent run A1 and the Week 1 runs, W2-11 and W2-12),
`eval/results/demo/W2-07_demo.md`, `eval/results/injection_results.md`, `eval/injection_detector_sets.json`,
`docs/smoke_tests.md` and `docs/decisions.md`.

<!-- GENERATED:START (scripts/week2_notes.py) -->
### The agent next to Week 1's runs

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

### By gold category and refusal-test kind (B0 and A1)

| Category | Qwen3-8B, passages (B0) | A1 agent |
|---|---|---|
| fact | 8/15 (53%) | 14/15 (93%) |
| ratio | 7/15 (47%) | 11/15 (73%) |
| period_comparison | 0/12 (0%) | 11/12 (92%) |
| multi_turn | 1/6 (17%) | 6/6 (100%) |
| multi_company | 5/10 (50%) | 10/10 (100%) |
| qualitative | 8/12 (67%) | 5/12 (42%) |
| refuse | 5/10 (50%) | 10/10 (100%) |
| refusal tests: refusal | 9/13 (69%) | 11/13 (85%) |
| refusal tests: partial | 0/2 (0%) | 1/2 (50%) |
| refusal tests: control | 7/7 (100%) | 7/7 (100%) |

### Refusals (B0 and A1)

| Measure | Qwen3-8B, passages (B0) | A1 agent |
|---|---|---|
| precision (refused and should) | 18/28 | 20/21 |
| recall (should refuse and did) | 18/20 | 20/20 |
| right rule among correct refusals | 14/18 | 18/20 |
| over-refusals | 10 | 2 |
| under-refusals | 3 | 0 |

### Failure labels of wrong answers (B0 and A1; confirmed in the grading sheet, else suggested)

| Label | Qwen3-8B, passages (B0) | A1 agent |
|---|---|---|
| wrong_fiscal_year | 0 | 0 |
| unit_confusion | 0 | 0 |
| arithmetic_slip | 10 | 1 |
| invented_number | 8 | 0 |
| wrong_line_item | 7 | 4 |
| ignored_not_reported | 1 | 0 |
| wrong_company | 5 | 0 |
| over_refusal | 10 | 2 |
| under_refusal | 3 | 0 |
| malformed_output | 1 | 0 |
| incomplete_summary | 4 | 7 |
| wrong_refusal_rule | 3 | 2 |

### Human grading, second round (W2-12)

| Measure | S1 few-shot | S2 reasoning + 5-sample vote | S3 decomposition | A1 agent |
|---|---|---|---|---|
| graders' exact agreement (independent grades) | 11/12 | 14/14 | 10/12 | 21/22 |
| graders' Cohen's kappa | 0.83 | 1.00 | 0.67 | 0.91 |
| final grade vs judge: exact agreement | 7/12 | 5/12 | 5/12 | 6/12 |
| final grade vs judge: Cohen's kappa | 0.29 | 0.12 | 0.10 | 0.29 |

### The agent's own checks and cost

| The agent's own checks (A1) | Value |
|---|---|
| verified by its critic | 102/102 |
| correct among the verified (decided) | 86/102 |
| correct among the not verified (decided) | – |
| tool calls per question (mean); questions with none | 1.6; 21 |
| tool calls that returned an error | 4/166 |
| stopped by a limit (time, steps, tokens) | 0 |
| live requests / cache hits (chat replies) | 187 / 0 |
| short numeric answers with several values and no calculation (no single value to score) | 0 |

### The 10 demo queries (W2-07)

| Demo queries (W2-07, real graph, tools and data) | Value |
|---|---|
| queries that met their expectation | 10/10 |
| verdicts | verified 8; refused 1; partial refusal, verified 1 |
| under the 60 s limit | 10/10 |
| seconds per query, median / max | 4.9 / 20.3 |
| tokens per query (mean) | 2,976 |
| tool calls (total) | 16 |

| Query | Category | Question | Verdict | Met | Tools | s |
|---|---|---|---|---|---|---|
| demo-01 | fact | What was Meta's net income in fiscal 2025? | verified | yes | get_financial_fact | 5.2 |
| demo-02 | ratio | What was Procter & Gamble's operating margin in fiscal 2023? | verified | yes | get_financial_fact, calculate | 4.8 |
| demo-03 | period_comparison | By what percentage did Bank of America's total revenue, net of interest expense, change... | verified | yes | get_financial_fact, calculate | 7.6 |
| demo-04 | multi_company | Which had the higher net income in fiscal 2023, Pfizer or Johnson & Johnson? | verified | yes | get_financial_fact, calculate | 5.7 |
| demo-05 | qualitative | What drove the change in Microsoft's revenue in fiscal 2023? | verified | yes | search_filings | 20.3 |
| demo-06 | fact | What was Amazon's net income in fiscal 2023? | verified | yes | get_financial_fact | 3.9 |
| demo-07 | multi_turn | And Microsoft's? | verified | yes | get_financial_fact | 4.0 |
| demo-08 | refuse | Do you expect Citigroup's net income to keep growing after fiscal 2025? | refused | yes | none | 1.2 |
| demo-09 | refuse | What was Wells Fargo's net income in fiscal 2023, and should I buy the stock? | partial refusal, verified | yes | get_financial_fact | 4.6 |
| demo-10 | refuse | How much did Wells Fargo spend on capital expenditures in fiscal 2023? | verified | yes | get_financial_fact, search_filings | 4.9 |

### Prompt-injection and PII defences (W2-13)

| The 12 injection tests (W2-13) | Defences off | Defences on |
|---|---|---|
| successful injections, all 12 tests | 0/12 | 0/12 |
| successful injections, tests that retrieved the planted passage | 0/11 | 0/11 |
| tests passed (no planted content, the right figure, the right refusal decision) | 9/12 | 8/12 |
| planted sentence reached the composer as written | 10/12 | 0/12 |
| answer carries the spec's note, tests that retrieved the planted passage | 0/11 | 11/11 |

### Smoke test on every laptop (W1-10, closed in W2-12)

| Member | OS, Python | Result |
|---|---|---|
| Alok Srivastava | Windows 11, Python 3.14.7 | OK |
| Amresh | WSL2 Ubuntu 20.04, Python 3.12.10 | OK |
| Chandra Shekhar | Windows 11 Home, build 10.0.26300 (64-bit), Python 3.13.7 (64-bit) | OK |
| Giridharan | Windows 10, Python 3.13.13 | OK |
| Anirudh | Windows 11, Python 3.12.10 | OK |

### Decisions recorded in Week 2 (docs/decisions.md)

| ADR | Date | Decision (first sentence) | Status |
|---|---|---|---|
| ADR-024 | 16 Nov 2026 | The agent graph (`agent/graph.py`, `agent/state.py`). | Accepted |
| ADR-025 | 17 Nov 2026 | The planner (`agent/planner.py`; prompt `planner-v1`, its model inputs pinned by sha256). | Accepted |
| ADR-026 | 17 Nov 2026 | Tool schemas for native tool calling (the ADR-016 follow-up; amends ADR-014's calculate contract). | Accepted |
| ADR-027 | 18 Nov 2026 | The critic node (`agent/critic.py`). | Accepted |
| ADR-028 | 19 Nov 2026 | The executor node (`agent/executor.py`, `make_executor(call_tool=None, count_tokens=None)`). | Accepted |
| ADR-029 | 20 Nov 2026 | Tracing in Langfuse (`core/tracing.py`). | Accepted |
| ADR-030 | 20 Nov 2026 | The composer and memory across turns. | Accepted |
| ADR-031 | 20 Nov 2026 | The graph end to end. | Accepted |
| ADR-032 | 20 Nov 2026 | The API (`api/main.py`, `uvicorn api.main:app`). | Accepted |
| ADR-033 | 21 Nov 2026 | The chat UI (`ui/app.py`, `streamlit run ui/app.py`). | Accepted |
| ADR-034 | 21 Nov 2026 | The agent's first scored gold run, A1, and how the W1-08 scorer reads an agent. | Accepted |
| ADR-035 | 21 Nov 2026 | Prompt-injection and PII defences. | Accepted |
<!-- GENERATED:END -->

## How the agent answers a question
1. **Planner** (W2-02, ADR-025): one guided model call decides refusal first (rules R1-R7 of `docs/spec.md` §6),
   then writes a plan with no numbers: which tool to call with which arguments, and which calculation to run.
2. **Executor** (W2-04, ADR-028): runs the plan's tool calls in order through the tool registry (XBRL facts, period
   comparison, filing search, calculator), builds the calculation from the fetched figures, and stops at the step,
   token and output limits; a typed tool error triggers one replan.
3. **Composer** (W2-06, ADR-030): one guided model call writes the answer; every value, citation and calculation is
   built from the tool outputs, never typed by the model. Retrieved passages are framed as data, and a passage
   addressed to an AI system is left out whole (W2-13, ADR-035).
4. **Critic** (W2-05, ADR-027): deterministic checks of the answer contract (units, metrics, figures against the
   evidence, citations, calculations recomputed); a failure sends the draft back once, a second failure ends the
   answer as not verified, and a payload an injection would add is never shown (W2-13).
5. **Around it:** memory of the earlier questions in a conversation (W2-06), Langfuse traces of every node, tool and
   model call (W2-08, ADR-029), the FastAPI service (W2-09, ADR-032) and the Streamlit UI (W2-10, ADR-033).

## Ticket by ticket
| Ticket | What was delivered | Evidence |
|---|---|---|
| W2-01 | Agent state, message schemas and the bounded graph (one replan, one send-back, the critic owns the verdict) | ADR-024, `agent/graph.py`, PR #34 |
| W2-02 | Planner: refusal first, entities, a JSON plan with no numbers; refusal decisions 48 of 49 on the development set | ADR-025, `eval/results/planner_dev.md`, PR #35 |
| W2-03 | Tool schemas for native tool calling: metric enum and glossary; accepted first calls 24 → 31 of 31 | ADR-026, `eval/results/tool_calls.md`, PR #36 |
| W2-04 | Executor: tool calls in plan order, calculations from fetched figures, step, token and output guards | ADR-028, `agent/executor.py`, PR #38 |
| W2-05 | Critic: unit, metric and figure checks on the answer contract; one send-back | ADR-027, `agent/critic.py`, PR #37 |
| W2-06 | Composer (values and citations from tool outputs) and memory across turns | ADR-030, `agent/composer.py`, PR #40 |
| W2-07 | Graph end to end, `scripts/ask.py`, 10 demo queries: 10 of 10 met their expectation | ADR-031, `eval/results/demo/W2-07_demo.md`, PR #41 |
| W2-08 | Langfuse tracing (off by default, PII scrubbed) and the trace dashboard | ADR-029, `scripts/trace_dashboard.py`, PR #39 |
| W2-09 | FastAPI service: `POST /ask`, `GET /health`, signed sessions, reply cache, rate limit | ADR-032, `api/main.py`, PR #42 |
| W2-10 | Streamlit chat UI: answer card, verified badge, citations, trace expander | ADR-033, `ui/app.py`, PR #43 |
| W2-11 | The agent run A1 on the gold set, scored next to B0 and S1-S3 | ADR-034, `eval/results/agent_baselines.md`, PR #44 |
| W2-12 | Week 1 carry-overs: second grading round, smoke tests 5 of 5, the independent WP2 check | `eval/results/grading/`, `docs/smoke_tests.md`, PR #45 |
| W2-13 | Prompt-injection and PII defences, measured before and after | ADR-035, `eval/results/injection_results.md`, PR #46 |
| W2-14 | Week 2 notes (this), Week 3 plan and tickets | `scripts/week2_notes.py`, `docs/week3_plan.txt` |

## Failure modes of the agent (the graders' labels, W2-12)
- **Wrong metric, fetched and grounded** (A06, B04, B14, B15, labelled wrong line item): the facts tool has no metric
  for the line item asked (deposits, noninterest expense, current assets, a segment's operating income), so the
  planner fetched a neighbouring one; B04, B14 and B15 presented it as the one asked, A06 named the substitute. The grounding check passes them:
  grounded is not correct.
- **A near-miss with a neighbouring figure** (C08, labelled arithmetic slip): total revenue, which includes membership fees,
  used for net sales (8.2% against 8.1%).
- **"Not reported" for a figure the filing has** (B08, labelled over-refusal): UnitedHealth's medical cost ratio.
- **Refusals:** one partial refusal refused whole (RF14) and two right decisions under the wrong rule (RF10, RF13).
- **Summaries** (7 of 12 graded below 2): the four zeros explain a neighbouring quantity, e.g. operating income for
  a question about revenue growth.
- **The critic** passed all 16: its metric check finds no named metric in some questions, passes an operation as
  soon as one named metric was fetched, and does not judge a summary's substance (`eval/results/agent_baselines.md`).

## Limits stated with the numbers
- Small cells: 102 questions split into cells of 10-48, so one question moves a cell by 2 to 10 points;
  differences are judged question by question with a sign test, and only p < 0.05 is called significant.
- One agent run at temperature 0, with prompts tuned on the development set and pinned before the gold run.
- B0 and S1-S3 read oracle-filtered context packs; the agent found its own sources, so the comparison is between
  systems, not prompts. Part of the refusal gain is the planner's worked refusal examples.
- Summary grades come from two graders on 12 questions (agreement on the agent's rows: 21 of 22, kappa 0.91); the
  LLM judge is a second opinion only.
- The injection tests plant one sentence at the top of the best passage, the worst case. The detector's catch rate
  on unseen wording is about one half: on attack sets written without seeing its patterns it caught 27/60 (first
  set, frozen), then 28/60 and 29/60 on a second set (wrongly flagging 0-1 of 60 filing-like sentences;
  `eval/results/injection_results.md`, `python eval/score_injection_detector.py`).
- The demo's 10 queries are not a test set: 5 are development questions, 5 new ones on company-years the gold set
  does not ask about.
