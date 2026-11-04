# FinSight - Local Development Setup (about 15 minutes)

> **Note:** These instructions are for local development and smoke testing using Ollama as a stand-in for the
> official endpoint. On 8 Nov only `.env` changes; the code stays the same.

## Prerequisites & Installation

1. **Clone the repo and create a virtual environment** (Python 3.11 or 3.12)
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

3. **Download, install and start Ollama**
   * Get the installer from [https://ollama.com/download](https://ollama.com/download) and start it; it serves
     `http://localhost:11434`.

4. **Pull the models** used in `.env.example`
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```
   `llama3.2:3b` fits a 4 GB GPU or runs (slowly) on CPU. Use `llama3.1:8b` only on a stronger machine.
   No Ollama? Set `EMBED_BACKEND=sentence-transformers` to compute embeddings locally instead.

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   (Windows cmd: `copy .env.example .env`.) Put your own contact email in `SEC_USER_AGENT`. Never commit `.env`.

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
