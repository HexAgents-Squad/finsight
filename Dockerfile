# FinSight's one image (W3-11, ADR-046): the API and the UI run from it with different commands
# (docker-compose.yml). It holds the code and the libraries only. Secrets come from .env at run time
# (env_file), and the data, the reply cache and the certificates are mounted, so none of them is ever in a layer.
# Python 3.14: the version the full set of libraries is tested on (requirements.txt; CI's 3.12 runs the dev subset).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/app \
    HF_HOME=/cache/huggingface \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# The libraries first, so a code change rebuilds only the last layers, at the tested versions
# (constraints-docker.txt). PyTorch (pulled in by sentence-transformers) comes from its CPU-only index: the default
# Linux wheel adds about 3 GB of CUDA libraries that a CPU never uses.
COPY requirements.txt requirements-dev.txt constraints-docker.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu -c constraints-docker.txt torch \
    && pip install -c constraints-docker.txt -r requirements.txt

# Compose runs the containers as the host user (FINSIGHT_UID/GID, set by the Makefile) so files written to the
# mounted data/ stay theirs. HOME and the Hugging Face cache must be writable by any uid, and a new named volume
# copies these modes.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /cache/huggingface /app/data \
    && chmod 1777 /cache /cache/huggingface /home/app /app/data

# Only what the app runs is copied (.dockerignore is an allowlist)
COPY . .

USER 1000:1000
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
