# Clean-machine run of the Docker setup (W3-11, step 4)

A second team member brings FinSight up from a clean clone on a machine that has never run it, following
README.md "Run it with Docker" exactly. They record the times and every fix here. A fix to the steps goes into
README.md (or the Makefile or the Compose file) in the same PR, so the next person does not need it.

To be filled in by the person who ran it. Nothing below is a result until they do.

## Machine
- Who and when:
- OS (Windows with WSL2 and its Ubuntu version, macOS or Linux), CPU, RAM:
- Docker and Compose versions (`docker version`, `docker compose version`):
- Where the data came from (copied from a teammate, or `make ingest index DOCKER=1`):

## Times
- Clone to `make check-data` OK:
- `make up`, first run (build, model download, healthy):
- `make up` after `make down` (second run):
- First question answered in the UI (which question, verified or not):

## Fixes needed
- (none, or each problem, its error message and the change made, with the commit)

## Result
- [ ] `make up` served the UI against the API
- [ ] A question was answered through the UI
- [ ] `make down` stopped everything, and `make up` came back
