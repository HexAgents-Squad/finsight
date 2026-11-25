# W3-14 screenshots: the demo's live steps on Docker, offline

`python scripts/demo_autopilot.py --headless --pause 1 --shots ...` against the UI of `make demo-offline`
(`docker-compose.offline.yml`) in WSL2 Ubuntu on 25 Nov 2026, on ports 8010 and 8511 (`FINSIGHT_API_PORT`,
`FINSIGHT_UI_PORT`) because 8000 and 8501 were in use. All seven steps showed their expected answer. The API's
environment had the endpoint blocked (HTTPS_PROXY to 127.0.0.1:9), local embeddings and tracing off; the reply cache
mounted into the container gained no entry, so every answer came from it and nothing was sent to the endpoint.

| File | Step |
|---|---|
| `01-demo-01.png` | a fact: Meta's net income, fiscal 2025 |
| `02-demo-02.png` | a ratio with its trace open: P&G's operating margin, fiscal 2023 |
| `03-demo-04.png` | two companies: Pfizer or Johnson & Johnson, fiscal 2023 |
| `04-demo-06.png` | a follow-up, the first question: Amazon's net income, fiscal 2023 |
| `05-demo-07.png` | the follow-up: "And Microsoft's?" |
| `06-RF01.png` | investment advice refused (R1) |
| `07-RF10.png` | an instruction to break the rules refused (R6) |
