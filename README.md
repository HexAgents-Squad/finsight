# FinSight

A multi-agent copilot that answers questions on US companies' SEC 10-K/10-Q filings with citations and verified calculations, built on a hosted ≤8B model.

Capstone project (Track B · API-Side), Executive PG Programme in Generative AI & Agentic AI, IIT Kharagpur.
Team **HexAgents_Squad**: Alok Srivastava · Anirudh M K · Chandra Shekhar Pandey · Amresh Kumar · Giridharan Selvaraj
Mentor: Dr. Gunjan Balde

The full plan is in [FinSight_Project_Plan.txt](FinSight_Project_Plan.txt). What FinSight answers, refuses and returns is defined in [docs/spec.md](docs/spec.md); the corpus scope is [config/companies.yaml](config/companies.yaml).

## Setup in 15 minutes
Python 3.11/3.12 venv → `pip install -r requirements.txt` → install Ollama and pull `llama3.2:3b` + `nomic-embed-text` → copy `.env.example` to `.env` → `python scripts/smoke_llm.py` → `pytest -q`. Step by step: [LOCAL_SETUP.md](LOCAL_SETUP.md).

## Folder structure
```
config/        companies.yaml (corpus scope), concept_map.yaml (metric → XBRL tags)
docs/          spec, decisions, mentor meeting notes, report/
data/          raw/ (SEC filings), sections/ (parsed sections), xbrl/ (structured financial data); not committed
ingest/        download → parse sections → chunk → index
retrieval/     BM25, dense, hybrid fusion, reranker
agent/         planner, executor, critic, LangGraph graph; prompts/
tools/         search_filings, get_financial_fact, calculate, compare_periods
api/           FastAPI service
ui/            Streamlit app
eval/          gold set (80 questions), evaluation scripts, results
tests/         unit tests
```
