# Smoke tests on each laptop (W1-10)

Every team member runs the smoke test against the course endpoints on their own laptop and adds one row here.
The smoke test makes one chat call, one embedding batch and one JSON call. It hides the endpoint's host and never prints the key, so its last line can be pasted as it is.

```bash
python scripts/smoke_llm.py
```

**Before you run it:** `.env` must be set up as in LOCAL_SETUP.md, "Hosted endpoint", steps 1–2 (`LLM_*`) and 5 (`EMBED_*`), with
`EMBED_BACKEND=openai` for the embedding endpoint.

**What to record:** your name, the date, the OS and Python version, and the result line (`OK: ...`, or the `FAILED` line). Don't record the endpoint address, the key or the certificate. The key allows one request at a time for the whole team, so run it one person at a time.

| Member | Date | OS, Python | Result |
|---|---|---|---|
| Alok Srivastava | 2026-11-14 | Windows 11, Python 3.14.7 | OK: 2 chat calls (0 retries), 279 tokens (238 prompt + 41 completion), 10.3 s in chat calls; 1 embedding request(s), 0.9 s |
