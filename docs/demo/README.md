# The demo (W3-14)

- [demo_script.md](demo_script.md): the 10-minute script for 5 Dec, who presents what, the live steps, the
  fallbacks, how to record the backup.
- `make demo-offline`: FinSight on Docker answering only from the reply cache (`docker-compose.offline.yml`), so
  the demo needs no endpoint and spends no shared key.
- `python scripts/demo_autopilot.py`: types the live steps into the UI one by one (Enter for the next), for the
  demo and its recording; `--headless --pause 2 --shots DIR` checks them and saves screenshots.
- `python scripts/demo_injection.py`: the blocked prompt injection, from the reply cache with local embeddings.
- The slides draft: shared with the team outside the repository (the editable deck changes in rehearsals); its PDF
  goes here once the slides are final.

## The backup recording

Not recorded yet. Once it is made (steps in demo_script.md, "Recording the backup"), put its link here:

- Recording: *(link to come)*
- Recorded by, on, with: *(name, date, `make demo-offline` or `make up`)*
