# FinSight v1.0

Release notes for the capstone submission (W3-13, ADR-048). The tag `v1.0` is set on the commit that merges this
release on `main`, once CI is green; the code freeze was Wed 25 Nov 2026, 18:00, and the submission is Thu 26 Nov.
Every number below is in a committed file named on its section's Sources line (`python scripts/release_check.py`
verifies it, with the other release checks).

*Sources:* `docs/week3_plan.txt`

## What v1.0 is

A research assistant that answers questions about the 10-K and 10-Q filings of 20 US companies with citations and
verified calculations, on the course's hosted Qwen3-8B endpoint (Track B):
- **Agent:** a LangGraph planner → executor → composer → critic graph on four tools (filing search, XBRL facts,
  period comparison, calculator), memory across turns, bounded loops and limits (`agent/`, `tools/`).
- **Data:** 120 filings parsed into 1,260 Item sections, C3 table-aware chunks in a hybrid BM25 + dense index, and
  the SEC's XBRL facts for 17 metrics (`ingest/`, `retrieval/`).
- **Serving:** a FastAPI service with sessions, a reply cache and rate limits, a Streamlit UI, Langfuse tracing,
  all started with `make up` (`api/`, `ui/`, `docker-compose.yml`).
- **Evaluation:** the frozen gold set, the refusal and injection tests, the scorer, two-grader grading, the
  evaluation report with the failure taxonomy, and the 12-page technical report (`eval/`, `docs/report/`).

*Sources:* `docs/report/final_report.md`, `eval/results/retrieval_summary.md`

## Results of the final version (A5, gold-v1, 102 questions)

| Measure | B0 (Week 1 baseline) | A1 (Week 2 agent) | A5 (v1.0) |
|---|---|---|---|
| correct answers, all types | 50/102 | 86/102 | 95/102 |
| short numeric | 16/48 | 42/48 | 44/48 |
| summaries | 8/12 | 5/12 | 9/12 |
| refusals and "not reported" | 21/32 | 29/32 | 32/32 |
| hallucinations (a number in no source) | 9 | 0 | 0 |
| tokens per question, mean | 3,536 | 3,000 | 3,419 |
| latency per question, median (s) | 2.38 | 4.97 | 5.54 |

Through the API: 22/22 refusal tests and 0/12 successful injections. `make up` on a fresh WSL2 install: 8 min 34 s
the first time, 36 s after `make down`.

*Sources:* `eval/results/evaluation_report.md`, `eval/results/api_safety_results.md`, `docs/deploy_check.md`

## Evaluation integrity at the release

- `eval/gold_set.jsonl` is byte for byte the version tagged `gold-v1` in Week 0, before any answer was measured.
- The 22 refusal and 12 injection tests keep every question, expected decision and planted passage they had at the
  tag. What changed since is review metadata: a second member's verification on every test (W1-10), two refusal
  rationales, and the match lists of two injection tests (INJ07, INJ11), corrected in the Week 0 review before any
  injection run.
- Prompts were tuned on development sets only; A5 ran on gold once, its configuration pinned before the run; A1 stays
  the reported baseline (ADR-037).

*Sources:* `docs/decisions.md`, `eval/results/evaluation_report.md`

## Week 3, ticket by ticket

| Ticket | What it delivered | Pull request (ADR) |
|---|---|---|
| W3-01 | ablations A2-A4 (no calculator, no critic, no statements-first retrieval) | #49 (ADR-036) |
| W3-02 | development set v2: 30 questions, one group per failure class | #52 (ADR-038) |
| W3-03 | the critic's checks of what the draft answers against what the question asks | #54 (ADR-039) |
| W3-04 | facts coverage: four new XBRL metrics (13 to 17) | #56 (ADR-040) |
| W3-05 | summary answers from the asked period's MD&A | #58 (ADR-041) |
| W3-06 | planner refusal edge cases: rule priority, partial refusals | #60 (ADR-042) |
| W3-07 | optimization: latency, tokens, code and test time | #62 (ADR-043) |
| W3-08 | version A5: the go / no-go and one pinned gold run | #64 (ADR-037) |
| W3-09 | the trace dashboard and the evaluation report with the failure taxonomy | #66 (ADR-044) |
| W3-10 | safety, refusal and PII tests through the API; responsible AI | #68 (ADR-045) |
| W3-11 | Docker Compose, the Makefile and a reproducible README | #70 (ADR-046) |
| W3-12 | the 12-page technical report and the rubric checklist | #72 (ADR-047) |
| W3-13 | this release: the release check, the notes, the tag | this pull request (ADR-048) |
| W3-14 | the demo script and backup recording | open at the time of writing |

*Sources:* `docs/decisions.md`

## How to verify it

- `python scripts/release_check.py` (or `make release-check` on a branch): the version, the frozen sets against
  `gold-v1`, every generated report and data validator, and these notes' traceability; on `main` it also checks a
  clean tree, that `main` equals `origin/main` and that `v1.0` does not exist yet; offline, needs `git fetch --tags`.
- `make test`: ruff and the offline tests (fake clients, no key).
- `make up`, then http://localhost:8501: the system on Docker (README, "Run it with Docker").
- The runs replay from the reply cache (`data/cache/llm`), so a re-run gives the same answers without a request.

*Sources:* `README.md`

## The submission package

- The repository at the tag `v1.0`: github.com/HexAgents-Squad/finsight (public).
- The technical report: `docs/report/final_report.pdf`; the rubric checklist: `docs/report/rubric_checklist.md`.
- The evaluation report with the failure taxonomy: `eval/results/evaluation_report.md` and its PDF.
- The trace dashboard: `scripts/trace_dashboard.py`, the traced runs in `eval/results/traces/`, screenshots in
  `docs/screenshots/w3/`; the Langfuse project itself is private, its link shared with the graders on request.
- The final UI: `docs/screenshots/w3-12/`; the responsible-AI section: `docs/responsible_ai.md`.
- The demo recording (W3-14), added when it is made.

## Known limitations

The report's §11: a small test set (one answer is about 1 percentage point), a gold set written by the team, citation
accuracy measured as "names a source the run had", coverage of 20 companies and three fiscal years, follow-up
questions that can lose the asked metric (F02), and a median of 5.54 s per question.

*Sources:* `eval/results/evaluation_report.md`, `docs/report/final_report.md`
