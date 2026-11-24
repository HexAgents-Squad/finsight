# Clean-machine run of the Docker setup (W3-11, step 4)

A second team member brings FinSight up from a clean clone on a machine that has never run it, following
README.md "Run it with Docker" exactly. They record the times and every fix here. A fix to the steps goes into
README.md (or the Makefile or the Compose file) in the same PR, so the next person does not need it.

To be filled in by the person who ran it. Nothing below is a result until they do.

## Machine
- Who and when: Alok on 25/11/2026. Alok raised this PR, so this is the author's run on a fresh WSL2 install, not the independent second-member run the ticket asks for
- OS (Windows with WSL2 and its Ubuntu version, macOS or Linux), CPU, RAM: Windows 11 Home, WSL2 Ubuntu, Ubuntu 26.04.1 LTS, Intel i7-10750H, 16 GB RAM
- Docker and Compose versions (`docker version`, `docker compose version`): Docker 29.9.3, Docker Compose v5.7.1
- Where the data came from (copied from a teammate, or `make ingest index DOCKER=1`): copied from Windows checkout

## Times
- Clone to `make check-data` OK: not timed
- `make up`, first run (build, model download, healthy): 8 min 34 s. The image build took 472 s; the API was healthy after 35 s and the UI after 39 s.
- `make up` after `make down` (second run): 36 s. The API was healthy after 26 s and the UI after 31 s.
- First question answered in the UI (which question, verified or not): the P&G operating margin, Verified 22.1% (from the saved replies). Then a new question, live through the endpoint: Coca-Cola's net income in fiscal 2024, Verified, $10,631 million, 5.22 s.

## Fixes needed
- The repository is private, so `git clone` fails without a GitHub sign-in. It's fixed in the README in commit `3625045`.

## Result
- [x] `make up` served the UI against the API
- [x] A question was answered through the UI
- [x] `make down` stopped everything, and `make up` came back
