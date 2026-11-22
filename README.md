# FinSight

A multi-agent copilot that answers questions on US companies' SEC 10-K/10-Q filings with citations and verified calculations, built on a hosted ≤8B model.

Capstone project (Track B · API-Side), Executive PG Programme in Generative AI & Agentic AI, IIT Kharagpur.
Team **HexAgents_Squad**: Alok Srivastava · Anirudh M K · Chandra Shekhar Pandey · Amresh Kumar · Giridharan Selvaraj
Mentor: Dr. Gunjan Balde

The full plan is in [FinSight_Project_Plan.txt](FinSight_Project_Plan.txt). What FinSight answers, refuses and returns is defined in [docs/spec.md](docs/spec.md); the corpus scope is [config/companies.yaml](config/companies.yaml).

## Setup in 15 minutes
Python 3.11–3.14 venv → `pip install -r requirements.txt` → install Ollama and pull `llama3.2:3b` (chat) → copy `.env.example` to `.env` → `python scripts/smoke_llm.py` → `pytest -q`. Embeddings are `bge-small-en-v1.5`, from the course's embedding endpoint (`EMBED_BACKEND=openai`) or run locally as `BAAI/bge-small-en-v1.5` (the same vectors, downloaded on first use). Step by step, including the data pipeline and the hosted Qwen3-8B endpoint (group CA certificate, `.env` only): [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Status (end of Week 0)
Built and tested: corpus download and Item-level parsing (`ingest/01`, `02`), XBRL facts (`ingest/05`), chunking and hybrid retrieval with recall@k (`ingest/03`, `04`, `retrieval/`, `eval/run_retrieval.py`), the four agent tools (`tools/`), the LLM client (`core/`), the gold set (`eval/gold_set.jsonl`, tag `gold-v1`) and the refusal/injection test sets. Planned: the agents (`agent/`), the API (`api/`), the UI (`ui/`) and a reranker. Decisions: [docs/decisions.md](docs/decisions.md); retrieval results: [eval/results/retrieval_summary.md](eval/results/retrieval_summary.md).

## Week 1 (10–14 Nov, complete; open items carried into Week 2)
The provided Qwen3-8B endpoint; a model comparison (Qwen3-8B vs Llama-3.2-3B, with and without retrieval) on short and summary answers, refusals and hallucinations; prompting baselines (zero-shot, few-shot, chain-of-thought with self-consistency, decomposition); retrieval experiments (re-index, question-derived filters, statements first, reranker). Results: [docs/report/week1_results_note.md](docs/report/week1_results_note.md) (one page) and [docs/report/week1_detailed_results.md](docs/report/week1_detailed_results.md); tickets and contracts: [docs/week1_plan.txt](docs/week1_plan.txt).

## Week 2 (16–22 Nov, complete)
The agentic layer: a LangGraph planner → executor → composer → critic agent on the four tools, memory across turns, Langfuse tracing, a FastAPI service and a Streamlit UI, prompt-injection and PII defences; the 10 demo queries met their expectation and the agent's gold run (A1) got 42/48 numeric answers right against the Week 1 baseline's 16/48. Results: [docs/report/week2_results_note.md](docs/report/week2_results_note.md) (one page) and [docs/report/week2_detailed_results.md](docs/report/week2_detailed_results.md); plan, design and tickets: [docs/week2_plan.txt](docs/week2_plan.txt).

## Week 3 (22–27 Nov, next)
Evaluate, harden, ship: ablations, the critic's metric checks, the evaluation report with the failure taxonomy, the trace dashboard, safety tests through the API, Docker Compose, the 12-page report; code freeze 25 Nov, submission 26 Nov. Plan and tickets: [docs/week3_plan.txt](docs/week3_plan.txt).

## Folder structure
```
config/        companies.yaml (corpus scope), concept_map.yaml (metric → XBRL tags)
docs/          spec, decisions (ADRs), week 0-3 plans, report/ (the weekly results notes)
data/          raw/ (SEC filings), sections/ (parsed sections), xbrl/ (structured financial data); not committed
ingest/        download → parse sections → XBRL facts → chunk → index
retrieval/     chunking, BM25, dense (Chroma), hybrid fusion (RRF), entities from the question, statements first, cross-encoder reranker
agent/         the LangGraph agent: planner, executor, composer, critic, graph, injection defences; prompts/
tools/         search_filings, get_financial_fact, calculate, compare_periods
core/          LLM and embedding clients, settings, tracing (Langfuse), PII masking
api/           FastAPI service: POST /ask, GET /health, sessions, cache, rate limit
ui/            Streamlit chat app (talks to the API only)
eval/          gold set (80 questions), development and safety sets, runners, scorer, results
scripts/       command-line tools: ask a question, smoke test, trace dashboard, weekly notes
tests/         offline unit tests (fake LLM and tools; no key needed)
```
