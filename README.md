# FinSight

A multi-agent copilot that answers questions on US companies' SEC 10-K/10-Q filings with citations and verified calculations, built on a hosted ≤8B model.

Capstone project (Track B · API-Side), Executive PG Programme in Generative AI & Agentic AI, IIT Kharagpur.
Team **HexAgents_Squad**: Alok Srivastava · Anirudh M K · Chandra Shekhar Pandey · Amresh Kumar · Giridharan Selvaraj
Mentor: Dr. Gunjan Balde

The full plan is in [FinSight_Project_Plan.txt](FinSight_Project_Plan.txt). What FinSight answers, refuses and returns is defined in [docs/spec.md](docs/spec.md); the corpus scope is [config/companies.yaml](config/companies.yaml).

## Setup in 15 minutes
Python 3.11–3.14 venv → `pip install -r requirements.txt` → install Ollama and pull `llama3.2:3b` (chat) → copy `.env.example` to `.env` → `python scripts/smoke_llm.py` → `pytest -q`. Embeddings run locally with `all-MiniLM-L6-v2` (downloaded on first use). Step by step, including the data pipeline: [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Status (end of Week 0)
Built and tested: corpus download and Item-level parsing (`ingest/01`, `02`), XBRL facts (`ingest/05`), chunking and hybrid retrieval with recall@k (`ingest/03`, `04`, `retrieval/`, `eval/run_retrieval.py`), the four agent tools (`tools/`), the LLM client (`core/`), the gold set (`eval/gold_set.jsonl`, tag `gold-v1`) and the refusal/injection test sets. Planned: the agents (`agent/`), the API (`api/`), the UI (`ui/`) and a reranker. Decisions: [docs/decisions.md](docs/decisions.md); retrieval results: [eval/results/retrieval_summary.md](eval/results/retrieval_summary.md).

## Folder structure
```
config/        companies.yaml (corpus scope), concept_map.yaml (metric → XBRL tags)
docs/          spec, decisions (ADRs), week-0 plan, report/
data/          raw/ (SEC filings), sections/ (parsed sections), xbrl/ (structured financial data); not committed
ingest/        download → parse sections → XBRL facts → chunk → index
retrieval/     chunking, BM25, dense (Chroma), hybrid fusion (RRF); reranker planned
agent/         (planned) planner, executor, critic, LangGraph graph; prompts/
tools/         search_filings, get_financial_fact, calculate, compare_periods
api/           (planned) FastAPI service
ui/            (planned) Streamlit app
eval/          gold set (80 questions), evaluation scripts, results
tests/         unit tests
```
