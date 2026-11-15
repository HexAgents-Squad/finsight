"""scripts/probe_embeddings.py and scripts/chunk_token_lengths.py offline: a fake embeddings server with the
hosted endpoint's limits (16 texts, 512 tokens) and a stand-in tokenizer, so nothing is downloaded or sent."""
import asyncio
import json
import re
import sys
from types import ModuleType, SimpleNamespace

import httpx
import numpy as np
import openai
import pytest

import core.tokens as tk
import scripts.chunk_token_lengths as cl
import scripts.probe_embeddings as pe
from core.settings import Settings

SECRET_HOST, SECRET_KEY = "secret-embed-host.example", "sk-embed-secret-1"


class WordTokenizer:
    def encode(self, text):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        return SimpleNamespace(ids=[0] * (len(spans) + 2), offsets=[(0, 0), *spans, (0, 0)])

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(pe, "MIN_GAP_S", 0.0)
    monkeypatch.setattr(tk, "embed_tokenizer", lambda: WordTokenizer())
    chunks = tmp_path / "chunks_C3.jsonl"
    chunks.write_text("".join(json.dumps({"text": f"chunk {i} revenue"}) + "\n" for i in range(80)), encoding="utf-8")
    monkeypatch.setattr(pe, "C3_CHUNKS", chunks)


def settings():
    return Settings(_env_file=None, LLM_BASE_URL="http://llm.test/v1", LLM_API_KEY="k", LLM_MODEL="m",
                    SEC_USER_AGENT="a b@c.d", EMBED_BACKEND="openai", EMBED_BASE_URL=f"https://{SECRET_HOST}/v1",
                    EMBED_API_KEY=SECRET_KEY, EMBED_MODEL="bge-small-en-v1.5")


def refused(message):
    request = httpx.Request("POST", f"https://{SECRET_HOST}/v1/embeddings")
    return openai.UnprocessableEntityError(f"{message} at https://{SECRET_HOST}/v1 key {SECRET_KEY}",
                                           response=httpx.Response(422, request=request), body=None)


class Server:
    """Refuses more than 16 texts or a text over 512 tokens (as the hosted endpoint does, measured)."""

    def __init__(self):
        self.embeddings, self.sent = self, []

    async def create(self, model, input):
        self.sent.append(list(input))
        if len(input) > pe.MAX_BATCH:
            raise refused(f"batch size {len(input)} > maximum allowed batch size 16")
        if any(len(WordTokenizer().encode(t).ids) > pe.MAX_TOKENS for t in input):
            raise refused("inputs must have less than 512 tokens")
        await asyncio.sleep(0.001)
        data = [SimpleNamespace(index=i, embedding=[1.0, float(len(t) % 7), 0.5]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)), usage=SimpleNamespace(prompt_tokens=3 * len(input)))


def probe(name, server=None):
    return asyncio.run(pe.PROBES[name](pe.Session(server or Server(), settings())))


def test_limits_are_found_at_16_texts_and_512_tokens():
    batch = probe("batch_limit")["findings"]
    assert (batch["16_texts"]["accepted"], batch["17_texts"]["accepted"], batch["17_texts"]["status"]) == \
        (True, False, 422)
    tokens = probe("token_limit")["findings"]
    assert tokens["512_tokens"]["accepted"] is True and tokens["513_tokens"]["accepted"] is False


def test_texts_of_an_exact_token_count_are_built():
    for n in (512, 513):
        assert len(WordTokenizer().encode(pe.text_of_tokens(n)).ids) == n


def test_basic_and_determinism():
    basic = probe("basic")["findings"]
    assert basic["dimensions"] == 3 and basic["ok"] is True
    assert probe("determinism")["findings"] == {"identical": True, "cosine": 1.0}


def test_throughput_reports_texts_per_second_and_the_c3_estimate():
    f = probe("throughput")["findings"]
    assert {f"batch_{n}" for n in pe.THROUGHPUT_BATCHES} <= set(f)
    assert f["batch_16"]["texts"] == pe.THROUGHPUT_TEXTS and f["batch_16"]["failed"] == 0
    assert f["c3_chunks"] == 80
    assert f["c3_minutes_at_batch_16"] == round(80 / f["batch_16"]["texts_per_s"] / 60, 1)


def test_concurrency_records_both_statuses():
    assert probe("concurrency")["findings"] == {"statuses": [200, 200], "both_ok": True}


def test_parity_compares_with_a_local_model(monkeypatch):
    class Local:
        def __init__(self, name, revision):
            assert name == pe.LOCAL_PARITY_MODEL and revision == tk.EMBED_TOKENIZER_REVISION

        def encode(self, texts, normalize_embeddings):
            return np.array([[1.0, float(len(t) % 7), 0.5] for t in texts])

    fake = ModuleType("sentence_transformers")
    fake.SentenceTransformer = Local
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    f = probe("parity")["findings"]
    assert f["same_model"] is True and f["cosines"] == [1.0] * len(pe.PARITY_TEXTS)


def test_cosine_and_norm():
    assert pe.cosine([1, 0], [0, 1]) == 0.0 and pe.cosine([2, 0], [1, 0]) == 1.0 and pe.norm([3, 4]) == 5.0


def fake_main(monkeypatch, server):
    monkeypatch.setattr(pe, "get_settings", settings)

    class Ctx:
        def __init__(self, s):
            self.embed_client = server

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(pe, "LLMClient", Ctx)


def test_main_writes_the_report_without_key_or_host(tmp_path, monkeypatch, capsys):
    fake_main(monkeypatch, Server())
    assert pe.main(["--out-dir", str(tmp_path), "--skip", "parity"]) == 0
    texts = [(tmp_path / f"embedding_probe.{s}").read_text(encoding="utf-8") for s in ("json", "md")]
    texts.append(capsys.readouterr().out)
    assert all(SECRET_HOST not in t and SECRET_KEY not in t for t in texts)
    report = json.loads(texts[0])
    assert "parity" not in report["probes"] and report["meta"]["model"] == "bge-small-en-v1.5"
    refusal = report["probes"]["batch_limit"]["requests"][1]["error"]
    assert "<endpoint>" in refusal and "<key>" in refusal


def test_a_crashing_probe_keeps_the_others(tmp_path, monkeypatch):
    fake_main(monkeypatch, Server())

    async def boom(s):
        raise RuntimeError("tokenizer exploded")

    monkeypatch.setitem(pe.PROBES, "token_limit", boom)
    assert pe.main(["--out-dir", str(tmp_path), "--only", "basic", "token_limit"]) == 0
    report = json.loads((tmp_path / "embedding_probe.json").read_text(encoding="utf-8"))
    assert report["probes"]["token_limit"]["findings"] == {"crashed": "RuntimeError: tokenizer exploded"}


def test_guards(monkeypatch, capsys):
    fake_main(monkeypatch, Server())
    assert pe.main(["--only", "basic"]) == 1 and "must not overwrite" in capsys.readouterr().err
    local = settings().model_copy(update={"EMBED_BACKEND": "sentence-transformers"})
    monkeypatch.setattr(pe, "get_settings", lambda: local)
    assert pe.main(["--out-dir", "x"]) == 1 and "is not 'openai'" in capsys.readouterr().err


def test_nothing_is_written_when_every_request_fails(tmp_path, monkeypatch, capsys):
    class Down(Server):
        async def create(self, model, input):
            raise openai.APIConnectionError(request=httpx.Request("POST", f"https://{SECRET_HOST}"))

    fake_main(monkeypatch, Down())
    assert pe.main(["--out-dir", str(tmp_path / "o"), "--skip", "parity"]) == 1
    assert "nothing written" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_chunk_length_summary_counts_both_limits():
    row = cl.summary([10, 300, 600, 200])
    assert (row["chunks"], row["max"], row["over_256"], row["over_512"]) == (4, 600, 2, 1)
    assert row["over_256_pct"] == 50.0 and row["over_512_pct"] == 25.0
    md = cl.markdown({"C3": row})
    assert "| C3 | 4 |" in md and "2 (50.0%)" in md and "1 (25.0%)" in md


def test_sign_test_is_the_exact_two_sided_binomial():
    from eval.retrieval_eval import sign_test
    assert sign_test(0, 0) == 1.0 and sign_test(5, 5) == 1.0
    assert abs(sign_test(23, 5) - 0.000903) < 1e-5                     # the dense / no-filter row (measured)
    assert sign_test(1, 0) == 1.0 and abs(sign_test(6, 0) - 0.03125) < 1e-9


def test_paired_comparison_counts_questions_per_model():
    import scripts.compare_embedding_runs as ce
    from eval.validate_gold import DEFAULT_PATH, validate_file
    items = [i for i in validate_file(DEFAULT_PATH).items if i.gold_sources][:3]
    gold = {i.id: [s.key() for s in i.gold_sources] for i in items}
    a = {items[0].id: gold[items[0].id], items[1].id: [], items[2].id: gold[items[2].id]}
    b = {items[0].id: [], items[1].id: gold[items[1].id], items[2].id: gold[items[2].id]}
    assert ce.compare(items, a, b) == {"better": 1, "worse": 1, "same": 1, "sign_test_p": 1.0}


def test_the_committed_comparison_regenerates(tmp_path, monkeypatch):
    # regenerated from the committed runs into tmp_path: the test never rewrites the committed report
    import scripts.compare_embedding_runs as ce
    monkeypatch.setattr(ce, "OUT_DIR", tmp_path)
    assert ce.main() == 0
    for name in ("embedding_comparison.md", "embedding_comparison.json"):
        assert (tmp_path / name).read_text(encoding="utf-8") == (ce.RESULTS / name).read_text(encoding="utf-8")
