# FinSight - Local Development Setup (about 15 minutes)

> **Note:** Steps 1–5 set up local development with Ollama as a stand-in model. To use the provided Qwen3-8B
> endpoint instead, follow [Hosted endpoint](#hosted-endpoint-qwen3-8b) after them: only `.env` and one
> certificate file change, never the code.

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
   For the model comparison (M2) create the 8k-context version once, because Ollama's default window silently
   drops the start of longer prompts: `ollama create llama3.2-3b-8k -f config/ollama/llama3.2-3b-8k.Modelfile`.
   On a CPU a long prompt takes about a minute, so pass `--timeout 300` to `eval/run_models.py` for local runs,
   and `--context-window 8192` so the truncation check looks for a cut at the model's real window (only with
   the 8k model: on plain `llama3.2:3b` that value would hide a cut at Ollama's smaller window).
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

## Hosted endpoint (Qwen3-8B)

The provided endpoint speaks the same OpenAI-compatible API as Ollama, over HTTPS with a certificate issued by
your group's own CA, which no system trust store knows. The client verifies the connection against that CA
file; verification is never switched off.

1. **Put the group's CA certificate in `certs/`** (git-ignored), e.g. `certs/<group>-ca.crt`. Use the file for your
   group from the learner guide. It is the CA's certificate (valid for years); the server's own certificate is
   renewed every few hours and must not be used instead.
2. **Edit `.env`** with your group's values from the learner guide:
   ```
   LLM_BASE_URL=https://<group host>:<port>/v1
   LLM_API_KEY=<team key>
   LLM_MODEL=Qwen/Qwen3-8B
   LLM_CA_CERT=certs/<group>-ca.crt
   LLM_ENABLE_THINKING=false
   ```
   Relative certificate paths are taken from the repo root. Keep `EMBED_BACKEND=sentence-transformers`: the
   endpoint has no embeddings, and the retrieval index was built with the local model.
3. **Check TLS without the key** (Windows: `curl.exe`; keep the key off the command line and out of shell history):
   ```bash
   curl --cacert certs/<group>-ca.crt https://<group host>:<port>/v1/models
   ```
   `{"detail":"missing or invalid bearer token"}` means the certificate and address are right. A certificate
   error means the wrong file or group.
4. **Run the smoke test**: `python scripts/smoke_llm.py`. It shows `TLS: verified against <group>-ca.crt` and
   `Thinking: off`, hides the endpoint's address, and ends with `OK: ...`. Post that output (never the key) in
   the team chat as your smoke-test check.
5. **Embedding endpoint (W1-16)**: put the course's `vm02-ca.crt` in `certs/` (git-ignored) and add to `.env`
   `EMBED_BASE_URL`, `EMBED_API_KEY` (the shared embedding key), `EMBED_CA_CERT=certs/vm02-ca.crt`,
   `EMBED_MODEL=bge-small-en-v1.5`, `EMBED_BATCH_SIZE=16` and `EMBED_MAX_INPUT_TOKENS=512`. Keep
   `EMBED_BACKEND=sentence-transformers` and `EMBED_LOCAL_MODEL_NAME=all-MiniLM-L6-v2` until the index is rebuilt
   with bge-small (ADR-021, with W1-04): search refuses an index built with another model. To try the endpoint
   before then, override for one command, e.g. `EMBED_BACKEND=openai python scripts/smoke_llm.py`.
6. **Capability probe (optional, W1-03)**: `python scripts/probe_endpoint.py` measures tool calling, JSON modes, the
   thinking switch, latency, limits and determinism (~33 requests, about 2 minutes) and writes
   `eval/results/endpoint_probe.md`. It deliberately sends two requests at once to confirm the 429, so run it only
   when nobody else is using the key (a subset such as `--skip concurrency` needs `--out-dir` elsewhere).

Rules for the team key: **one active request at a time for the whole team**, 60 requests per minute and 8,192
tokens per call (prompt + reply). The client enforces them within one process: one request at a time per
endpoint, at most one start per second, retries after 2, 4 and 8 s when the key is busy (429) or the server
fails (5xx, timeouts), and a prompt that cannot fit is refused before it is sent. It cannot see a teammate's
process, so announce long runs in the team chat. Replies are cached in `data/cache/llm/` (git-ignored): an
identical request is answered from disk, so re-running an evaluation is free; delete that folder (or set
`LLM_CACHE_DIR=`) to force fresh calls. The smoke test always calls the endpoint. Never commit, screenshot or share the key, the address or the certificate
outside the team. Qwen3's reasoning mode stays off (`LLM_ENABLE_THINKING=false`); any `<think>` block is
removed from replies either way. The provided endpoint currently ignores the switch and never reasons (checked on
10 Nov), so setting it to `true` changes nothing there.

| Smoke-test failure | Meaning and fix |
|---|---|
| `TLSError` ... `does not verify` / `not trusted` | Check `LLM_CA_CERT` points at your group's CA file and the URL is your group's |
| `TLSError` ... `did not complete a TLS handshake` | The server is not speaking HTTPS there: check `http://` vs `https://` and the port |
| `TLSConfigError` | `LLM_CA_CERT` is not a PEM certificate (e.g. the wrong download) |
| `TLSError` for `bge-small-en-v1.5` ... `set EMBED_CA_CERT` / `check EMBED_CA_CERT` | The embedding endpoint is another server with its own CA: set `EMBED_CA_CERT=certs/vm02-ca.crt` (the embeddings CA, not the chat one) |
| `TLSConfigError` naming `EMBED_CA_CERT` | `EMBED_CA_CERT` is not a PEM certificate |
| `UnprocessableEntityError` from embeddings | More than 16 texts or a text over 512 tokens: set `EMBED_BATCH_SIZE=16` and `EMBED_MAX_INPUT_TOKENS=512` |
| settings error `no certificate file at ...` | The path in `LLM_CA_CERT` is wrong (relative paths start at the repo root) |
| `AuthenticationError` | TLS works; the key in `LLM_API_KEY` is wrong or missing |
| `RateLimitError` | Still refused after 3 retries: someone else is using the team key (one active request per key); try again shortly |
| `PromptTooLongError` | The prompt plus the 1,024-token reply would exceed 8,192 tokens; shorten the context |
| `NotFoundError` | `LLM_MODEL` is not `Qwen/Qwen3-8B`, or the URL does not end in `/v1` |
| `NetworkError` | The address or port is wrong, or the endpoint is unreachable from your network |

To go back to Ollama, restore the three `LLM_*` lines from `.env.example` and empty `LLM_CA_CERT`.

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
