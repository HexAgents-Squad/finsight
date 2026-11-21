# Smoke tests on each laptop (W1-10)

Every team member runs the smoke test against the course endpoints on their own laptop and adds one row here.
The smoke test makes one chat call, one embedding batch and one JSON call. It hides the endpoint's host and never prints the key, so its last line can be pasted as it is.

```bash
python scripts/smoke_llm.py
```

**Before you run it:** `.env` must be set up as in LOCAL_SETUP.md, "Hosted endpoint", steps 1–2 (`LLM_*`) and 5 (`EMBED_*`), with
`EMBED_BACKEND=openai` for the embedding endpoint: the test's header then shows `Embeddings: openai (bge-small-en-v1.5)`.

**What to record:** your name, the date, the OS and Python version, the `Embeddings:` value from the header, and the result line (`OK: ...`, or the `FAILED` line). Don't record the endpoint address, the key or the certificate. The key allows one request at a time for the whole team, so run it one person at a time.

**Handing in your row (W2-12):** post your filled row (or the five values) and the unedited last line of the output in the team chat; Chandra Shekhar adds the four rows to this table in the W2-12 PR and links the chat messages there. If it prints `FAILED`, post that line too: the failure is the result to fix, so the row stays `pending` until a rerun passes. Done when all five rows say `OK` (W2-12 acceptance: 5 of 5).

| Member | Date | OS, Python | Embeddings | Result |
|---|---|---|---|---|
| Alok Srivastava | 2026-11-15 | Windows 11, Python 3.14.7 | openai (bge-small-en-v1.5) | OK: 2 chat calls (0 retries), 279 tokens (238 prompt + 41 completion), 1.4 s in chat calls; 1 embedding request(s), 1.0 s |
| Amresh | | | | pending |
| Chandra Shekhar | | | | pending |
| Giridharan | | | | pending |
| Anirudh | | | | pending |
