# FinSight: one command per step (W3-11, ADR-046). Run from the repo root on Linux, macOS or WSL2 (Windows).
#   make up      check .env and data/, build the image, start the API and the UI, wait until both are healthy
#   make help    every target
# The Python steps use .venv when it exists, else python3. With DOCKER=1 they run inside the app image instead
# (e.g. `make ingest index DOCKER=1` on a machine with Docker and no virtual environment).

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
COMPOSE ?= docker compose
ifeq ($(DOCKER),1)
RUN = $(COMPOSE) run --rm tools python
else
RUN = $(PYTHON)
endif

# The containers run as you, so the files they write to data/ (the reply cache, Chroma's SQLite file) stay yours
export FINSIGHT_UID ?= $(shell id -u)
export FINSIGHT_GID ?= $(shell id -g)

LANGFUSE_SECRETS := LF_POSTGRES_PASSWORD LF_CLICKHOUSE_PASSWORD LF_REDIS_PASSWORD LF_MINIO_PASSWORD \
	LF_NEXTAUTH_SECRET LF_SALT LF_ENCRYPTION_KEY

.PHONY: help ingest index eval test release-check check-data up down logs ps langfuse

help: ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-11s %s\n", $$1, $$2}'

ingest: ## download the filings and XBRL facts, parse the sections, build the facts and the chunks (needs SEC_USER_AGENT)
	$(RUN) ingest/01_download.py
	$(RUN) ingest/02_parse_sections.py
	$(RUN) ingest/05_xbrl_facts.py
	$(RUN) ingest/03_chunk.py

index: ## build the C3 search index the API uses: BM25 and Chroma (about 1 h on a CPU)
	$(RUN) ingest/04_index.py --variants C3

eval: ## offline: validate the gold, test and development sets, and check the evaluation report from committed results
	$(PYTHON) eval/validate_gold.py --final
	$(PYTHON) eval/validate_test_sets.py --final
	$(PYTHON) eval/validate_dev_set.py --final
	$(PYTHON) eval/validate_dev_set.py eval/dev_set_v2.jsonl --final
	$(PYTHON) eval/report.py --check

test: ## offline: ruff and pytest, as CI runs them
	$(PYTHON) -m ruff check .
	$(PYTHON) -m pytest -q

release-check: ## offline: everything that must hold before the v1.0 tag (scripts/release_check.py)
	$(PYTHON) scripts/release_check.py --allow-dirty

check-data: ## check that .env and the data the API loads are present
	$(PYTHON) scripts/check_data.py

up: check-data ## build the image, start the API and the UI, and wait until both are healthy (the first start takes minutes)
	mkdir -p data/cache/llm certs
	$(COMPOSE) up --build --detach --wait --wait-timeout 900 api ui
	@echo "UI:  http://$$($(COMPOSE) port ui 8501)"
	@echo "API: http://$$($(COMPOSE) port api 8000)/docs"

down: ## stop and remove the containers (data/ and the model cache volume are kept)
	$(COMPOSE) --profile langfuse down

logs: ## follow the containers' logs
	$(COMPOSE) logs --follow

ps: ## the containers and their health
	$(COMPOSE) ps

langfuse: ## optional: start a self-hosted Langfuse at http://localhost:3000 (needs the LF_* lines in .env)
	@missing=""; for v in $(LANGFUSE_SECRETS); do grep -Eqs "^$$v=.+" .env || missing="$$missing $$v"; done; \
	if [ -n "$$missing" ]; then echo "set in .env:$$missing (README.md, Langfuse)"; exit 1; fi
	$(COMPOSE) --profile langfuse up --detach --wait --wait-timeout 600 langfuse-web langfuse-worker
