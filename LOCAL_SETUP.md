# FinSight - Local Development Setup (about 15 minutes)

> **Note:** These instructions are for local development and smoke testing using Ollama as a stand-in for the
> provided endpoint. Connecting to the hosted Qwen3-8B endpoint (group CA certificate, key in `.env` only) is
> Week 1 ticket W1-01 (UPGRAD-43), which adds a "Hosted endpoint" section here; see `docs/week1_plan.txt` §2.1.

## Prerequisites & Installation

1. **Clone the repo and create a virtual environment** (Python 3.11–3.14; CI uses 3.12)
   ```bash
   git clone https://github.com/HexAgents-Squad/finsight.git
   cd finsight
   python -m venv .venv
   ```
   Activate it: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux).

2. **Install Python dependencies** (includes the dev tools from `requirements-dev.txt`)
   ```bash
   pip install -r requirements.txt
   ```
   This also installs PyTorch through `sentence-transformers` (about 200 MB on CPU), the largest part of the
   setup time.

3. **Download, install and start Ollama**
   * Get the installer from [https://ollama.com/download](https://ollama.com/download) and start it; it serves
     `http://localhost:11434`.

4. **Pull the chat model** used in `.env.example`
   ```bash
   ollama pull llama3.2:3b
   ```
   `llama3.2:3b` fits a 4 GB GPU or runs (slowly) on CPU. Use `llama3.1:8b` only on a stronger machine.
   Embeddings need no server: `EMBED_BACKEND=sentence-transformers` (the default) runs `all-MiniLM-L6-v2`
   locally, and the first run downloads it from Hugging Face (~90 MB, about 40 s; internet needed once). The
   retrieval index is built with this model, so keep it unless you rebuild the index (ADR-013).

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   (Windows cmd: `copy .env.example .env`.) Put your own contact email in `SEC_USER_AGENT`. Never commit `.env`.
   Values set in the shell environment override `.env` (pydantic-settings), so unset a stray `LLM_MODEL` or
   `EMBED_BACKEND` if the smoke test shows an unexpected model.

## Running the Smoke Test

With Ollama running, check chat, embeddings and structured JSON end to end; it prints tokens and seconds:
```bash
python scripts/smoke_llm.py
```
It ends with `OK: ...` (exit code 0) when everything works. A `FAILED [...]` line names the problem, e.g.
`NetworkError` when Ollama is not running.

## Running the tests
```bash
pytest -q
ruff check .
```
The tests run offline in seconds (the LLM client is tested against fake clients), so they need no Ollama.

## Building the data (needed for retrieval and the tools)
Each step writes git-ignored files under `data/` and can be re-run; the outputs are deterministic.
```bash
python ingest/01_download.py          # 120 filings + XBRL company facts from SEC EDGAR (several minutes; needs SEC_USER_AGENT)
python ingest/02_parse_sections.py    # 1,260 Item sections -> data/sections/sections.jsonl (~9 min)
python ingest/05_xbrl_facts.py        # 1,303 facts + coverage and spot checks -> data/xbrl/*.csv (seconds)
python ingest/03_chunk.py             # chunk variants C1/C2/C3 -> data/chunks/ (seconds)
python ingest/04_index.py             # BM25 (seconds) + dense Chroma index (~1 h per variant on a CPU)
python eval/run_retrieval.py          # recall@k tables -> eval/results/retrieval_summary.md
```
Only the C3 indexes are needed by `search_filings`: `python ingest/04_index.py --variants C3`.
