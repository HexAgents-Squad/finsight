# The demo (W3-14)

- [demo_script.md](demo_script.md): the 10-minute script, who presents what in a live run, the live steps, the
  fallbacks, how to record the demo (submitted as a video by 27 Nov: [../submission.md](../submission.md)).
- `make demo-offline`: FinSight on Docker answering only from the reply cache (`docker-compose.offline.yml`), so
  the demo needs no endpoint and spends no shared key.
- `python scripts/demo_autopilot.py`: types the live steps into the UI one by one (Enter for the next), for the
  demo and its recording; `--headless --pause 2 --shots DIR` checks them and saves screenshots.
- `python scripts/demo_injection.py`: the blocked prompt injection, from the reply cache with local embeddings.
- The seven live steps checked on Docker in the offline demo mode: [docs/screenshots/w3-14/](../screenshots/w3-14/).
- The slides draft: shared with the team outside the repository (the editable deck changes in rehearsals); its PDF
  goes here once the slides are final.

## The recording

The demo submitted with the capstone (steps in demo_script.md, "Recording the demo"):

- Recording: [03_Demo_Recording.mp4 on Google Drive](https://drive.google.com/file/d/1xY3_OGtOIoUtsXE67BxPfWcZwpU-6RQf/view) (anyone with the link can view)
- Recorded by Alok Srivastava on 26 Nov 2026, on the Docker setup (`make up`), with the questions traced to Langfuse; three of those traces are public: [P&G's operating margin](https://us.cloud.langfuse.com/project/cmwnz4frw017vad0e92ldh748/traces/bc2a247e7f454a58779e41297631086a), [Pfizer or Johnson & Johnson](https://us.cloud.langfuse.com/project/cmwnz4frw017vad0e92ldh748/traces/b187871705e2d068a22ec2b3ed2ec353), [a refused rule-breaking request](https://us.cloud.langfuse.com/project/cmwnz4frw017vad0e92ldh748/traces/1d8c7b29464db0b56a5d3fcceec249fb).
