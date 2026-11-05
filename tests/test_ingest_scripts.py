"""ingest/01_download.py and 02_parse_sections.py command lines: exit codes and output safety (offline)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "_script"), ROOT / "ingest" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def filing_html():
    from tests.test_sections import STANDARD, tenk_html  # the parser tests' synthetic 10-K
    return tenk_html(STANDARD).encode()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A one-filing manifest in a temp data dir, with 02's paths pointed at it."""
    script = load_script("02_parse_sections.py")
    data = tmp_path / "data"
    (data / "raw" / "AAPL").mkdir(parents=True)
    (data / "raw" / "AAPL" / "10-K.htm").write_bytes(filing_html())
    record = {"ticker": "AAPL", "cik": "0000320193", "sector": "tech", "form": "10-K", "fiscal_year": 2024,
              "fiscal_period": "FY", "period_end": "2024-09-28", "filing_date": "2024-11-01",
              "accession_no": "0000320193-24-000123", "source_url": "https://www.sec.gov/x.htm",
              "path": "raw/AAPL/10-K.htm", "exhibit": None, "exhibit_url": None, "exhibit_path": None}
    manifest = data / "raw" / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    sections = data / "sections"
    sections.mkdir()
    (sections / "sections.jsonl").write_text("FULL CORPUS\n", encoding="utf-8")
    for name, value in {"DATA_DIR": data, "MANIFEST_FILE": manifest, "SECTIONS_DIR": sections,
                        "SAMPLE_DIR": data / "sample"}.items():
        monkeypatch.setattr(script, name, value)
    return SimpleNamespace(script=script, data=data, sections=sections, tmp=tmp_path)


def test_a_full_run_writes_the_three_outputs(corpus, monkeypatch, capsys):
    replaced = []
    real_replace = Path.replace

    def recording_replace(self, target):
        replaced.append((self.name, Path(target).name))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    assert corpus.script.main([]) == 1                    # one filing: the whole-corpus count check fails
    assert "CHECK FAILED: 8 documents; the corpus must have between 500 and 2000" in capsys.readouterr().out
    # each output is written to a .part file and moved into place, so a crash never leaves half a corpus
    assert ("sections.jsonl.part", "sections.jsonl") in replaced
    assert {t for _, t in replaced} >= {"sections.jsonl", "parse_report.csv", "corpus_stats.csv"}
    docs = [json.loads(line) for line in (corpus.sections / "sections.jsonl").read_text(encoding="utf-8").split("\n")
            if line]
    assert {d["item"] for d in docs} >= {"Item 7", "Item 8"}
    assert (corpus.sections / "parse_report.csv").exists() and (corpus.sections / "corpus_stats.csv").exists()
    assert not list(corpus.sections.glob("*.part"))       # written atomically


def test_a_subset_run_never_replaces_the_corpus(corpus, capsys):
    assert corpus.script.main(["--tickers", "AAPL"]) == 2
    assert "give --out-dir" in capsys.readouterr().err
    assert (corpus.sections / "sections.jsonl").read_text(encoding="utf-8") == "FULL CORPUS\n"
    out = corpus.tmp / "subset"
    assert corpus.script.main(["--tickers", "aapl", "--out-dir", str(out)]) == 0
    assert (out / "sections.jsonl").read_text(encoding="utf-8").count("\n") >= 2
    assert (corpus.sections / "sections.jsonl").read_text(encoding="utf-8") == "FULL CORPUS\n"


def test_an_unknown_ticker_is_an_error_not_an_empty_corpus(corpus, capsys):
    assert corpus.script.main(["--tickers", "ZZZ", "--out-dir", str(corpus.tmp / "x")]) == 2
    assert "not in the manifest: ['ZZZ']" in capsys.readouterr().err
    assert not (corpus.tmp / "x").exists()


def test_a_filing_that_parses_to_nothing_writes_nothing(corpus, capsys):
    (corpus.data / "raw" / "AAPL" / "10-K.htm").write_bytes(b"<html><body><p>No Items here.</p></body></html>")
    assert corpus.script.main([]) == 1
    assert "no section documents were parsed" in capsys.readouterr().err
    assert (corpus.sections / "sections.jsonl").read_text(encoding="utf-8") == "FULL CORPUS\n"


def test_a_missing_download_is_a_setup_error(corpus, capsys):
    (corpus.data / "raw" / "AAPL" / "10-K.htm").unlink()
    assert corpus.script.main([]) == 2
    assert "downloaded filing missing" in capsys.readouterr().err
    assert (corpus.sections / "sections.jsonl").read_text(encoding="utf-8") == "FULL CORPUS\n"


@pytest.mark.parametrize("argv", [["--tickers"], ["--sample"]])
def test_ticker_options_need_at_least_one_ticker(corpus, argv):
    with pytest.raises(SystemExit) as stop:
        corpus.script.main(argv)
    assert stop.value.code == 2                           # argparse usage error


def test_download_network_failure_is_exit_2(monkeypatch, capsys):
    script = load_script("01_download.py")

    def unreachable(argv=None):
        raise requests.ConnectionError("sec.gov unreachable")

    monkeypatch.setattr(script, "run", unreachable)
    assert script.main([]) == 2
    assert "SEC request failed: sec.gov unreachable" in capsys.readouterr().err


def test_retry_after_zero_means_retry_now():
    from ingest.edgar import BACKOFF_BASE_S, SecClient

    class Session:
        def __init__(self):
            self.headers = {}
            self.responses = [SimpleNamespace(status_code=429, headers={"Retry-After": "0"}),
                              SimpleNamespace(status_code=200, headers={})]

        def get(self, url, timeout):
            return self.responses.pop(0)

    waits = []
    client = SecClient("Team you@x.org", session=Session(), sleep=waits.append, clock=lambda: 1e9)
    assert client.get("u").status_code == 200
    assert BACKOFF_BASE_S not in waits and 0 in waits



def test_the_committed_sample_comes_only_from_a_full_run(corpus, capsys):
    assert corpus.script.main(["--tickers", "AAPL", "--out-dir", str(corpus.tmp / "x"), "--sample", "AAPL"]) == 2
    assert "--sample needs a full run" in capsys.readouterr().err
    assert corpus.script.main(["--sample", "JPM"]) == 1               # no JPM in this manifest: nothing to sample
    assert "sample not written" in capsys.readouterr().err
    assert not (corpus.data / "sample" / "sections_sample.jsonl").exists()
