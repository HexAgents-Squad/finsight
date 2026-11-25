# Rubric checklist (W3-12)

Each assessment criterion and each deliverable, with the committed evidence that meets it and its status on
26 Nov 2026. The detailed rubric was asked for in the Week 2 review and has not been received, so this list follows
the published weights (FinSight_Project_Plan.txt, "Assessment weights") and the project plan's deliverables checklist
(§13). Every path below exists in the repository (`python scripts/final_report.py --check` verifies it).

Status: **met** (evidence committed), **open** (the named ticket finishes it), **partly** (what is missing is said).

## Assessment criteria

### Technical depth and rigour (30%)

| Criterion | Evidence | Status |
|---|---|---|
| A measured baseline and three prompting strategies, with how the 8B model fails | report §5; `eval/results/model_comparison.md`, `eval/results/prompting_baselines.md` | met |
| Retrieval experiments: chunking × retriever × filters × reranking, recall@k | report §6; `eval/results/retrieval_summary.md` | met |
| Agentic layer: tools, planner, executor, critic, bounded loops | report §7; `agent/`, `tools/`; ADR-024 to ADR-031 in `docs/decisions.md` | met |
| "What did we do to compensate for a small model?" with ablations | report §8; `eval/results/agent_ablations.md` | met |
| End-to-end evaluation with paired significance tests and human grading | report §4, §9; `eval/results/agent_baselines.md`, `eval/GRADING.md`, `eval/results/grading/` | met |
| Failure taxonomy (retrieval miss, reasoning error, tool-call error, hallucination, refusal error) | report §9; `eval/results/evaluation_report.md` | met |
| No tuning on the test set; gold frozen; pinned runs | report §4; `eval/gold_set.jsonl`, `eval/dev_set_v2.jsonl`; ADR-037 | met |
| Every design decision recorded with its numbers | `docs/decisions.md` (ADR-001 to ADR-047) | met |

### Punctuality and consistent progress (25%)

| Criterion | Evidence | Status |
|---|---|---|
| Weekly plans and results notes, on time | `docs/week0_plan.txt` … `docs/week3_plan.txt`; `docs/report/week1_results_note.md`, `docs/report/week2_results_note.md` | met |
| One reviewed pull request per ticket, CI green | the repository's pull requests; `.github/workflows/ci.yml` | met |
| Code freeze 25 Nov, release v1.0 and submission 26 Nov | `docs/release_v1.0.md`, `scripts/release_check.py`; W3-13 | partly: the release check and notes are in; the v1.0 tag and the upload follow the merge |
| Mentor sessions used and recorded | `docs/mentor_meetings.md` | partly: only the 7 Nov session is recorded; the Week 1 and Week 2 reviews should be added (W3-13) |

### A working deployed system (20%)

| Criterion | Evidence | Status |
|---|---|---|
| One-command deployment (Docker Compose, Makefile, `.env.example`, no secrets) | `docker-compose.yml`, `Dockerfile`, `Makefile`, `.env.example`; README "Run it with Docker"; ADR-046 | met |
| A run on a clean machine | `docs/deploy_check.md` | partly: run by the PR author on a fresh WSL2 install; an independent member's run is open |
| Front-end with rate limiting and caching | `api/main.py`, `ui/app.py`; ADR-032, ADR-033 | met |
| Latency within budget (60 s per question) | report §9; `eval/results/evaluation_report.md` | met |
| Safety shown through the API | report §10; `eval/results/api_safety_results.md` | met |
| A working front-end: the final UI | `ui/app.py`; screenshots `docs/screenshots/w3-12/`; report Figures 2 and 6 | met |
| A 10-minute demo (submitted as a recording; live at the panel if nominated) | `docs/demo/demo_script.md`, `scripts/demo_autopilot.py`, `docker-compose.offline.yml`, `docs/submission.md`; W3-14 | partly: the script, the offline demo mode and the autopilot are in; the narrated recording follows |

### Report and demo (25%)

| Criterion | Evidence | Status |
|---|---|---|
| 12-page report following the plan's outline (11 sections) | `docs/report/final_report.md`, `docs/report/final_report.pdf` | met: body within 12 pages, recorded in `docs/report/final_report.build.json` |
| Every figure traceable to a committed file | `scripts/final_report.py`, `tests/test_final_report.py` | met |
| Each member's section reviewed by its owner | the "Owner (reviews)" line under each heading | open: each owner's review in the W3-12 pull request |
| A rehearsed 10-minute demo | `docs/demo/demo_script.md` | partly: the script; the recording before 27 Nov; a live run rehearsed if the team is nominated for the 12 Dec panel |

## Deliverables (the course brief, Track B, and project plan §13)

The brief's requirements I-VI and guardrails are mapped one by one in the report's Appendix D.

| Deliverable | Evidence | Status |
|---|---|---|
| Git repository with the full agent stack and reproducible code (README, Makefile, `.env.example`, no secrets) | `agent/`, `tools/`, `api/`, `ui/`; `README.md`, `Makefile`, `.env.example`, `LOCAL_SETUP.md` | met |
| Trace dashboard with real query executions (link and screenshots) | `scripts/trace_dashboard.py`, `eval/results/traces/`, `docs/screenshots/w3/` | partly: screenshots and the dashboard table are committed; the Langfuse project is private, so its link is shared on request |
| Evaluation report with the failure taxonomy (+ PDF) | `eval/results/evaluation_report.md`, `eval/results/report/evaluation_report.pdf` | met (generated by `eval/report.py` in place of the plan's notebook, ADR-044) |
| Responsible AI section: intended use, failure modes, refusals, adversarial tests (prompt injection, PII) | `docs/responsible_ai.md`; report §10 | met |
| 12-page technical report | `docs/report/final_report.pdf` | met |
| 10-minute demo (a recording in the submission, by 27 Nov) | `docs/demo/` | partly: script, offline demo mode, autopilot and injection demo in; the recording follows |
