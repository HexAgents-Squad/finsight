# W3-12 screenshots: the final UI

The Streamlit UI (`ui/app.py`) on main at the W3-12 branch point (after W3-11), against the API
(`uvicorn api.main:app`), driven by headless Chrome on 25 Nov 2026. Every answer came from the agent's reply cache:
the API ran with tracing off, local `BAAI/bge-small-en-v1.5` embeddings (the same vectors as the course endpoint)
and the chat endpoint blocked, so no request reached the shared endpoint while these were taken. The report's
Figures 2 and 6 (`docs/report/figures/final_ui_answers.png`, `final_ui_refusals.png`) are crops of them.

| File | What it shows |
|---|---|
| `01-verified-ratio-and-trace.png` | P&G's operating margin, fiscal 2023: verified, 22.1%, the calculation, figures, sources and the open trace |
| `02-follow-up.png` | "What was Amazon's net income in fiscal 2023?", then "And Microsoft's?" in the same conversation |
| `03-partial-refusal.png` | Wells Fargo's net income answered, "should I buy the stock?" declined (R1) |
| `04-refusal.png` | a forecast about Citigroup refused (R2), with a suggestion of what can be answered |
