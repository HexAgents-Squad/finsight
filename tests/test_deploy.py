"""W3-11 (ADR-046): the Docker Compose file, the Dockerfile, .dockerignore, the Makefile and scripts/check_data.py,
read as files (no Docker needed). They must agree with the code they serve: the data paths the API loads, the
commands documented in LOCAL_SETUP.md, the checks CI runs. And no secret may reach an image or a committed file.
CI's docker job builds the image and starts the containers."""
from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
SERVICES = COMPOSE["services"]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
SECRET_NAME = re.compile(r"PASSWORD|SECRET|SALT|ENCRYPTION_KEY|REDIS_AUTH|API_KEY|TOKEN")

sys.path.insert(0, str(ROOT / "scripts"))
import check_data  # noqa: E402


def volumes(service: str) -> dict[str, tuple[str, str]]:
    """The service's mounts as {target: (source, mode)}."""
    out = {}
    for spec in SERVICES[service]["volumes"]:
        source, target, *mode = spec.split(":")
        out[target] = (source, mode[0] if mode else "rw")
    return out


def in_container(path: Path) -> str:
    return str(PurePosixPath("/app") / path.relative_to(ROOT).as_posix())


def test_the_two_services_run_the_documented_commands_from_one_image():
    api, ui = SERVICES["api"], SERVICES["ui"]
    assert api["image"] == ui["image"] and api["build"] == ui["build"] == "."
    assert api["command"][:2] == ["uvicorn", "api.main:app"] and api["command"][-1] == "8000"
    cmd = re.search(r"^CMD (\[.*\])$", DOCKERFILE, re.M)
    assert cmd and ast.literal_eval(cmd.group(1)) == api["command"]       # the image's default is the API
    assert ui["command"][:3] == ["streamlit", "run", "ui/app.py"] and ui["command"][-1] == "8501"
    assert ui["environment"]["FINSIGHT_API_URL"] == "http://api:8000"     # the UI talks only to the API


def test_only_the_api_reads_env_and_every_port_listens_on_localhost():
    assert SERVICES["api"]["env_file"] == ".env"
    assert [name for name, s in SERVICES.items() if "env_file" in s] == ["api"]
    assert not any(SECRET_NAME.search(key) for key in SERVICES["ui"]["environment"])
    ports = [p for s in SERVICES.values() for p in s.get("ports", [])]
    assert ports and all(p.startswith("127.0.0.1:") for p in ports), ports    # the API has no authentication


def test_the_api_mounts_exactly_the_data_the_code_loads():
    from core.settings import REPO_ROOT, Settings
    from ingest import MANIFEST_FILE, XBRL_DIR
    from retrieval import CHUNKS_DIR, INDEX_DIR
    cache = REPO_ROOT / Settings.model_fields["LLM_CACHE_DIR"].default
    expected = {in_container(INDEX_DIR): "rw",             # Chroma opens its SQLite file read-write
                in_container(CHUNKS_DIR): "ro", in_container(XBRL_DIR): "ro", in_container(MANIFEST_FILE.parent): "ro",
                in_container(cache): "rw", in_container(ROOT / "certs"): "ro"}
    mounts = volumes("api")
    assert {t: m for t, (_, m) in mounts.items() if t.startswith("/app/")} == expected
    for target, (source, _) in mounts.items():
        if target.startswith("/app/"):
            assert source == "./" + target.removeprefix("/app/")            # the same path on the host
    assert mounts["/cache/huggingface"][0] == "hf-cache" and "HF_HOME=/cache/huggingface" in DOCKERFILE


def test_health_checks_gate_the_ui_on_a_ready_api():
    api, ui = SERVICES["api"], SERVICES["ui"]
    assert "/health" in " ".join(api["healthcheck"]["test"])    # 200 only when ready, 503 while starting
    assert "/_stcore/health" in " ".join(ui["healthcheck"]["test"])
    assert ui["depends_on"]["api"]["condition"] == "service_healthy"
    assert int(api["healthcheck"]["start_period"].removesuffix("s")) >= 300    # a model download and ~1 GB of index


def test_langfuse_is_an_optional_profile_whose_secrets_all_come_from_env():
    optional = {name: s for name, s in SERVICES.items() if name not in ("api", "ui")}
    assert {"langfuse-web", "langfuse-worker", "postgres"} <= set(optional)
    assert all(s.get("profiles") == ["langfuse"] for s in optional.values())
    assert "profiles" not in SERVICES["api"] and "profiles" not in SERVICES["ui"]
    referenced = set()
    for s in optional.values():
        for key, value in (s.get("environment") or {}).items():
            value = str(value)
            referenced |= set(re.findall(r"\$\{(LF_[A-Z_]+):-\}", value))
            if SECRET_NAME.search(key) or key == "DATABASE_URL":
                assert re.search(r"\$\{LF_[A-Z_]+:-\}", value), f"{key} must come from .env, not the file"
    referenced |= set(re.findall(r"\$\{(LF_[A-Z_]+):-\}", " ".join(SERVICES["redis"]["command"])))
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert all(re.search(rf"^{name}=$", example, re.M) for name in referenced)    # listed, empty
    secrets = re.search(r"^LANGFUSE_SECRETS := (.*?)\n\n", MAKEFILE, re.M | re.S).group(1).split()
    assert set(secrets) - {"\\"} == referenced                     # `make langfuse` checks every one


def test_dockerignore_is_an_allowlist_that_covers_every_package_the_runtime_imports():
    lines = [x for x in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines() if x and x[0] != "#"]
    assert lines[0] == "*"
    allowed = {x[1:].rstrip("/") for x in lines if x.startswith("!")}
    assert not allowed & {".env", "certs", "data", ".git", ".venv", "tests"}
    packages = {p.parent.name for p in ROOT.glob("*/__init__.py")}
    runtime = {"api", "ui"}
    while True:                         # the packages api/ and ui/ import, and the ones those import
        imported = set()
        for package in sorted(runtime):
            for path in sorted((ROOT / package).rglob("*.py")):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                        [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
                    imported |= {n.split(".")[0] for n in names} & packages
        if imported <= runtime:
            break
        runtime |= imported
    assert runtime <= allowed, runtime - allowed
    assert {"config", ".streamlit", "requirements.txt", "requirements-dev.txt", "constraints-docker.txt"} <= allowed


def test_the_dockerfile_installs_the_tested_versions_runs_as_a_user_and_takes_no_secret():
    tested = re.search(r"tested locally on Python (3\.\d+)", (ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert re.search(rf"^FROM python:{re.escape(tested.group(1))}-slim", DOCKERFILE, re.M)
    pins = dict(x.split("==") for x in (ROOT / "constraints-docker.txt").read_text(encoding="utf-8").splitlines()
                if x and x[0] != "#")
    assert {"chromadb", "torch", "sentence-transformers", "openai", "langgraph", "fastapi", "streamlit"} <= set(pins)
    assert DOCKERFILE.count("-c constraints-docker.txt") == 2              # torch, then the rest
    user = re.findall(r"^USER (\S+)$", DOCKERFILE, re.M)
    assert user and user[-1].split(":")[0] not in ("0", "root")
    assert "download.pytorch.org/whl/cpu" in DOCKERFILE                  # no CUDA libraries in a CPU image
    assert not re.search(r"^(ARG|ENV)\b.*(KEY|SECRET|TOKEN|PASSWORD)", DOCKERFILE, re.M)
    assert not re.search(r"^COPY .*\.env", DOCKERFILE, re.M)


def test_the_makefile_has_the_ticket_targets_and_eval_runs_what_ci_runs():
    phony = set(re.search(r"^\.PHONY: (.*)$", MAKEFILE, re.M).group(1).split())
    assert {"ingest", "index", "eval", "test", "up", "down"} <= phony
    targets = set(re.findall(r"^([a-z-]+):", MAKEFILE, re.M))
    assert phony <= targets
    recipes = re.findall(r"^([a-z-]+):[^\n]*\n((?:\t[^\n]*\n)+)", MAKEFILE, re.M)
    assert {name for name, _ in recipes} == targets                    # every recipe line starts with a tab
    body = dict(recipes)
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    validations = [s["run"] for s in ci["jobs"]["checks"]["steps"] if str(s.get("run", "")).startswith("python eval/")]
    assert [x.strip().replace("$(PYTHON)", "python") for x in body["eval"].splitlines() if "validate" in x] == \
        [x.strip() for x in validations]
    assert re.search(r"^up: check-data\b", MAKEFILE, re.M)
    assert f"--variants {check_data.VARIANT}" in body["index"]


def test_check_data_names_each_missing_file(tmp_path):
    from tools.search_filings import DEFAULT_VARIANT
    assert check_data.VARIANT == DEFAULT_VARIANT
    companies = yaml.safe_load((ROOT / "config" / "companies.yaml").read_text(encoding="utf-8"))["companies"]
    assert check_data.tickers() == [c["ticker"] for c in companies]       # read without PyYAML

    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config" / "companies.yaml", tmp_path / "config" / "companies.yaml")
    missing = check_data.problems(tmp_path)
    assert len(missing) == 6 and missing[0].startswith(".env is missing")        # the 20 facts files on one line
    assert missing[-1].startswith(f"data/xbrl/ is missing the XBRL facts of {len(companies)} of {len(companies)}")
    files = [line.split(" is missing")[0] for line in missing[1:-1]]
    files += [f"data/xbrl/{c['ticker']}.json" for c in companies] + [".env"]
    for name in files:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert check_data.problems(tmp_path) == []
    (tmp_path / "data" / "raw" / "manifest.jsonl").unlink()
    (tmp_path / "data" / "xbrl" / "KO.json").unlink()
    assert [x.split(" (")[0] for x in check_data.problems(tmp_path)] == [
        "data/raw/manifest.jsonl is missing", f"data/xbrl/ is missing the XBRL facts of 1 of {len(companies)} companies"]


@pytest.mark.parametrize("text", ["make up", "cp .env.example .env", "make ingest index", "make down", "WSL"])
def test_the_readme_gives_the_steps_from_a_clean_clone(text):
    assert text in (ROOT / "README.md").read_text(encoding="utf-8")
