# FinSight

A multi-agent copilot that answers questions on US companies' SEC 10-K/10-Q filings with citations and verified calculations, built on a hosted ≤8B model.

Capstone project (Track B · API-Side), Executive PG Programme in Generative AI & Agentic AI, IIT Kharagpur.
Team **HexAgents_Squad**: Alok Srivastava · Anirudh M K · Chandra Shekhar Pandey · Amresh Kumar · Giridharan Selvaraj
Mentor: Dr. Gunjan Balde

The full plan is in [FinSight_Project_Plan.txt](FinSight_Project_Plan.txt). What FinSight answers, refuses and returns is defined in [docs/spec.md](docs/spec.md); the corpus scope is [config/companies.yaml](config/companies.yaml).

## Setup in 15 minutes
Python 3.11–3.14 venv → `pip install -r requirements.txt` → install Ollama and pull `llama3.2:3b` (chat) → copy `.env.example` to `.env` → `python scripts/smoke_llm.py` → `pytest -q`. Embeddings are `bge-small-en-v1.5`, from the course's embedding endpoint (`EMBED_BACKEND=openai`) or run locally as `BAAI/bge-small-en-v1.5` (the same vectors, downloaded on first use). Step by step, including the data pipeline and the hosted Qwen3-8B endpoint (group CA certificate, `.env` only): [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Run it with Docker (W3-11)
One command, `make up`, starts the API and the UI, which talks to it. The data and `.env` stay on your machine and are mounted at run time, so no image holds a secret or the corpus (ADR-046).

**You need:** Docker with Compose v2 (`docker compose version`), GNU make, git and python3. On Windows, use WSL2: install Docker Desktop with the WSL2 backend and Ubuntu from the Microsoft Store, then run every command below in the Ubuntu shell. Clone into the Linux file system (`~/`), not `/mnt/c`: it is much faster, and the search index's SQLite file needs real file locks.

1. **Clone:**
   ```bash
   git clone https://github.com/HexAgents-Squad/finsight.git
   cd finsight
   ```
2. **Settings:** `cp .env.example .env`, then edit `.env`:
   - the hosted endpoint: the five `LLM_*` lines from [LOCAL_SETUP.md](LOCAL_SETUP.md#hosted-endpoint-qwen3-8b), with the group's CA certificate in `certs/`;
   - or a local Ollama on the same machine: `LLM_BASE_URL=http://host.docker.internal:11434/v1` (inside a container, `localhost` is the container itself);
   - `SEC_USER_AGENT` with your contact e-mail (needed only to download the filings).
3. **Data:** the API loads the C3 search index, its chunks, the XBRL facts and the filing manifest from `data/`. Either:
   - copy them from a teammate who built them: `data/index/`, `data/chunks/chunks_C3.jsonl`, `data/xbrl/` and `data/raw/manifest.jsonl` (about 1.4 GB), or
   - build them with `make ingest index DOCKER=1` (in the image, so no Python setup is needed). It downloads 120 filings from SEC EDGAR and builds the index, which takes well over an hour on a CPU.

   `make check-data` lists anything still missing.
4. **Start:** `make up`. It checks `.env` and `data/`, builds the image, starts both containers and waits until both are healthy. The first run takes several minutes: it installs the libraries and downloads the embedding model and the tokenizer into a Docker volume. It then prints the two addresses.
5. **Use it:** open http://localhost:8501 and ask, for example, "What were Apple's total net sales in fiscal 2024?". The API's own documentation is at http://localhost:8000/docs.
6. **Stop:** `make down` stops the containers. The data, the reply cache and the model volume are kept, so the next `make up` is quick.

**Other targets** (`make help` lists them all):
- `make test`: ruff and pytest, as CI runs them. Like `make eval`, it runs on the host and needs the virtual environment from [LOCAL_SETUP.md](LOCAL_SETUP.md).
- `make eval`: offline checks of the gold, test and development sets, and of the evaluation report against the committed results.
- `make ps` and `make logs`: the containers' health and their logs.
- `make langfuse`: optional, a self-hosted Langfuse at http://localhost:3000. It needs the `LF_*` lines in `.env`; the team traces to Langfuse Cloud instead.

**If something goes wrong:**
- The API stays unhealthy: `make logs` shows the reason. Usually a file is missing from `data/`, or the `LLM_CA_CERT` path does not exist under `certs/`.
- A port is in use: set `FINSIGHT_UI_PORT` or `FINSIGHT_API_PORT` in `.env`.
- Questions fail but `/health` says ok: check the `LLM_*` lines. The smoke test checks them outside Docker: `python scripts/smoke_llm.py`.

The second-person run on a clean machine is recorded in [docs/deploy_check.md](docs/deploy_check.md).

## Status (end of Week 0)
Built and tested: corpus download and Item-level parsing (`ingest/01`, `02`), XBRL facts (`ingest/05`), chunking and hybrid retrieval with recall@k (`ingest/03`, `04`, `retrieval/`, `eval/run_retrieval.py`), the four agent tools (`tools/`), the LLM client (`core/`), the gold set (`eval/gold_set.jsonl`, tag `gold-v1`) and the refusal/injection test sets. Planned: the agents (`agent/`), the API (`api/`), the UI (`ui/`) and a reranker. Decisions: [docs/decisions.md](docs/decisions.md); retrieval results: [eval/results/retrieval_summary.md](eval/results/retrieval_summary.md).

## Week 1 (10–14 Nov, complete; open items carried into Week 2)
The provided Qwen3-8B endpoint; a model comparison (Qwen3-8B vs Llama-3.2-3B, with and without retrieval) on short and summary answers, refusals and hallucinations; prompting baselines (zero-shot, few-shot, chain-of-thought with self-consistency, decomposition); retrieval experiments (re-index, question-derived filters, statements first, reranker). Results: [docs/report/week1_results_note.md](docs/report/week1_results_note.md) (one page) and [docs/report/week1_detailed_results.md](docs/report/week1_detailed_results.md); tickets and contracts: [docs/week1_plan.txt](docs/week1_plan.txt).

## Week 2 (16–22 Nov, complete)
The agentic layer: a LangGraph planner → executor → composer → critic agent on the four tools, memory across turns, Langfuse tracing, a FastAPI service and a Streamlit UI, prompt-injection and PII defences; the 10 demo queries met their expectation and the agent's gold run (A1) got 42/48 numeric answers right against the Week 1 baseline's 16/48. Results: [docs/report/week2_results_note.md](docs/report/week2_results_note.md) (one page) and [docs/report/week2_detailed_results.md](docs/report/week2_detailed_results.md); plan, design and tickets: [docs/week2_plan.txt](docs/week2_plan.txt).

## Week 3 (22–27 Nov, next)
Evaluate, improve, harden, ship: ablations; an improved agent (A5: stronger deterministic critic checks, the missing XBRL metrics, better summaries and refusal edge cases, tuned on a development set v2 and run on gold once); optimization of latency, tokens and code; the evaluation report with the failure taxonomy; the trace dashboard; safety tests through the API; Docker Compose; the 12-page report; code freeze 25 Nov, submission 26 Nov. Plan and tickets: [docs/week3_plan.txt](docs/week3_plan.txt).

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
