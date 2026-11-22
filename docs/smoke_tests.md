# Smoke tests on each laptop (W1-10)

Every team member runs the smoke test against the course endpoints on their own laptop and adds one row here.
The smoke test makes one chat call, one embedding batch and one JSON call. It hides the endpoint's host and never prints the key, so its last line can be pasted as it is.

```bash
python scripts/smoke_llm.py
```

**Before you run it:** `.env` must be set up as in LOCAL_SETUP.md, "Hosted endpoint", steps 1–2 (`LLM_*`) and 5 (`EMBED_*`), with
`EMBED_BACKEND=openai` for the embedding endpoint: the test's header then shows `Embeddings: openai (bge-small-en-v1.5)`.

**What to record:** your name, the date, the OS and Python version, the `Embeddings:` value from the header, and the result line (`OK: ...`, or the `FAILED` line). Don't record the endpoint address, the key or the certificate. The key allows one request at a time for the whole team, so run it one person at a time.

**Handing in your row (W2-12):** each member sent the five values and the unedited last line of the output (e-mail or the team chat, 21–22 Nov 2026), and the rows were added here in the W2-12 PR. A `FAILED` line would have kept a row `pending` until a rerun passed. **Done: 5 of 5 rows say `OK`** (W2-12 acceptance). Anirudh's run counted 13 more prompt tokens than the others' (251 against 238), with the same completion; it passed, and the difference is noted here, not yet explained.

| Member | Date | OS, Python | Embeddings | Result |
|---|---|---|---|---|
| Alok Srivastava | 2026-11-15 | Windows 11, Python 3.14.7 | openai (bge-small-en-v1.5) | OK: 2 chat calls (0 retries), 279 tokens (238 prompt + 41 completion), 1.4 s in chat calls; 1 embedding request(s), 1.0 s |
| Amresh | 2026-11-21 | WSL2 Ubuntu 20.04, Python 3.12.10 | openai (bge-small-en-v1.5) | OK: 2 chat calls (0 retries), 279 tokens (238 prompt + 41 completion), 1.5 s in chat calls; 1 embedding request(s), 1.4 s |
| Chandra Shekhar | 2026-11-22 | Windows 11 Home, build 10.0.26300 (64-bit), Python 3.13.7 (64-bit) | openai (bge-small-en-v1.5) | OK: 2 chat calls (0 retries), 279 tokens (238 prompt + 41 completion), 1.3 s in chat calls; 1 embedding request(s), 14.1 s |
| Giridharan | 2026-11-21 | Windows 10, Python 3.13.13 | openai (bge-small-en-v1.5) | OK: 2 chat calls (0 retries), 279 tokens (238 prompt + 41 completion), 1.4 s in chat calls; 1 embedding request(s), 0.6 s |
| Anirudh | 2026-11-22 | Windows 11, Python 3.12.10 | openai (bge-small-en-v1.5) | OK: 2 chat calls (0 retries), 292 tokens (251 prompt + 41 completion), 0.9 s in chat calls; 1 embedding request(s), 24.9 s |
