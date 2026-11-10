"""core/llm.py and core/settings.py against fake OpenAI clients (offline, no Ollama needed)."""
import asyncio
import ssl
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import certifi  # an httpx dependency: a real PEM CA file, no network needed
import httpx
import openai
import pytest
from pydantic import BaseModel

import core.llm
from core.exceptions import FinSightError
from core.llm import LLMClient, build_ssl_context, extract_json, strip_thinking
from core.settings import REPO_ROOT, SecSettings, Settings, get_settings

REQUIRED = {"LLM_BASE_URL": "http://llm.test/v1", "LLM_API_KEY": "key", "LLM_MODEL": "chat-model",
            "SEC_USER_AGENT": "HexAgents_Squad test@example.com"}


class Filing(BaseModel):
    ticker: str
    year: int


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **{**REQUIRED, **overrides})


def reply(text, prompt_tokens=7, completion_tokens=3):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


class FakeChat:
    """Stands in for AsyncOpenAI: returns (or raises) the queued outputs and records every request."""

    def __init__(self, *outputs):
        self.outputs, self.requests = list(outputs), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.requests.append(request)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class FakeEmbeddings:
    """Returns a 2-d vector per input text ([batch call number, position]), in shuffled order like some servers."""

    def __init__(self):
        self.batches = []
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, model, input):
        self.batches.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[len(self.batches), i]) for i in range(len(input))]
        return SimpleNamespace(data=list(reversed(data)))


def client(*outputs, **overrides) -> LLMClient:
    # the remote-embedding tests exercise the "openai" backend; the local backend is tested on its own
    return LLMClient(settings(**{"EMBED_BACKEND": "openai", **overrides}), chat_client=FakeChat(*outputs),
                     embed_client=FakeEmbeddings())


def run(coro):
    return asyncio.run(coro)


def status_error(cls, code):
    response = httpx.Response(code, request=httpx.Request("POST", "http://llm.test/v1/chat/completions"))
    return cls("error", response=response, body=None)


USER = [{"role": "user", "content": "Which company filed?"}]


# --- extract_json ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    '{"ticker": "MSFT", "year": 2024}',
    '```json\n{"ticker": "MSFT", "year": 2024}\n```',
    '```JSON\n{"ticker": "MSFT", "year": 2024}\n```',
    'Sure! Here it is:\n```\n{"ticker": "MSFT", "year": 2024}\n```\nAnything else?',
    'Here you go: {"ticker": "MSFT", "year": 2024} Hope that helps.',
    # cases that broke the first version (code review on #6)
    '{"ticker": "MSFT", "year": 2024, "note": "the filing quotes ```ls``` verbatim"}',
    '```python\nx = {"a": 1}\n```\n```json\n{"ticker": "MSFT", "year": 2024}\n```',
    'Use the {ticker} field. {"ticker": "MSFT", "year": 2024}',
    '{"ticker": "MSFT", "year": 2024} (note: {x})',
    '{"ticker": "MSFT", "year": 2024}\n{"ticker": "AAPL", "year": 2023}',
])
def test_extract_json_finds_the_object(text):
    assert Filing.model_validate_json(extract_json(text)) == Filing(ticker="MSFT", year=2024)


# --- chat -----------------------------------------------------------------------------------------------

def test_chat_sends_the_messages_at_temperature_0_and_counts_usage():
    llm = client(reply("  OPERATIONAL \n"))
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
               {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    assert run(llm.chat(history)) == "OPERATIONAL"
    request = llm.client.requests[0]
    assert request["messages"] == history and request["temperature"] == 0.0 and request["model"] == "chat-model"
    assert request["max_tokens"] == 1024                             # LLM_MAX_COMPLETION_TOKENS default
    assert (llm.usage.calls, llm.usage.prompt_tokens, llm.usage.completion_tokens) == (1, 7, 3)


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_reply_keeps_its_error_type(content):
    with pytest.raises(FinSightError) as err:
        run(client(reply(content)).chat(USER))
    assert err.value.error_type == "EmptyResponseError"


@pytest.mark.parametrize("error, kind", [
    (status_error(openai.AuthenticationError, 401), "AuthenticationError"),
    (status_error(openai.RateLimitError, 429), "RateLimitError"),
    (openai.APIConnectionError(request=httpx.Request("POST", "http://llm.test")), "NetworkError"),
])
def test_api_errors_are_wrapped_and_not_retried_again(error, kind):
    llm = client(error, reply("never used"))
    with pytest.raises(FinSightError) as err:
        run(llm.chat_json(USER, Filing))
    assert err.value.error_type == kind
    assert len(llm.client.requests) == 1          # the openai client already did the network retries


def test_http_retries_go_to_the_openai_client_only():
    llm = LLMClient(settings(LLM_HTTP_RETRIES=4, LLM_JSON_RETRIES=1))
    assert llm.client.max_retries == 4 and llm.embed_client.max_retries == 4


# --- chat_json ------------------------------------------------------------------------------------------

def test_chat_json_returns_the_validated_object_and_leaves_input_unchanged():
    llm = client(reply('```json\n{"ticker": "MSFT", "year": 2024}\n```'))
    messages = [dict(m) for m in USER]
    assert run(llm.chat_json(messages, Filing)) == Filing(ticker="MSFT", year=2024)
    assert messages == USER
    sent = llm.client.requests[0]["messages"]
    assert sent[0]["role"] == "system" and '"ticker"' in sent[-1]["content"]


def test_chat_json_keeps_the_callers_system_prompt():
    llm = client(reply('{"ticker": "A", "year": 1}'))
    run(llm.chat_json([{"role": "system", "content": "You are the planner."}, *USER], Filing))
    sent = llm.client.requests[0]["messages"]
    assert [m["role"] for m in sent].count("system") == 1 and sent[0]["content"] == "You are the planner."


def test_chat_json_reprompts_with_the_validation_error():
    llm = client(reply("not json at all"), reply('{"ticker": "MSFT", "year": "two thousand"}'),
                 reply('{"ticker": "MSFT", "year": 2024}'))
    assert run(llm.chat_json(USER, Filing)).year == 2024
    assert llm.usage.json_retries == 2 and llm.usage.calls == 3 and llm.usage.total_tokens == 30
    third = llm.client.requests[2]["messages"]
    assert third[-2] == {"role": "assistant", "content": '{"ticker": "MSFT", "year": "two thousand"}'}
    assert "failed validation" in third[-1]["content"] and "year" in third[-1]["content"]


def test_chat_json_gives_up_after_the_retry_budget():
    llm = client(*[reply("nope")] * 4, LLM_JSON_RETRIES=2)
    with pytest.raises(FinSightError) as err:
        run(llm.chat_json(USER, Filing))
    assert err.value.error_type == "JSONValidationError"
    assert len(llm.client.requests) == 3


# --- embed ----------------------------------------------------------------------------------------------

def test_embed_batches_and_keeps_input_order():
    llm = client(EMBED_BATCH_SIZE=2)
    vectors = run(llm.embed(["a", "b", "c", "d", "e"]))
    assert llm.embed_client.batches == [["a", "b"], ["c", "d"], ["e"]]
    assert vectors == [[1, 0], [1, 1], [2, 0], [2, 1], [3, 0]]


def test_embed_rejects_a_bare_string_and_skips_empty_input():
    llm = client()
    with pytest.raises(TypeError):
        run(llm.embed("one text"))
    assert run(llm.embed([])) == [] and llm.embed_client.batches == []


def test_embeddings_endpoint_defaults_to_the_chat_endpoint():
    assert str(LLMClient(settings()).embed_client.base_url).startswith("http://llm.test/v1")
    separate = LLMClient(settings(EMBED_BASE_URL="http://embed.test/v1", EMBED_API_KEY="other"))
    assert str(separate.embed_client.base_url).startswith("http://embed.test/v1")
    assert separate.embed_client.api_key == "other"


def test_local_backend_loads_the_model_once(monkeypatch):
    loads = []

    class FakeST:
        def __init__(self, name):
            loads.append(name)

        def encode(self, texts, batch_size):
            return SimpleNamespace(tolist=lambda: [[float(len(t))] for t in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers", ModuleType("sentence_transformers"))
    sys.modules["sentence_transformers"].SentenceTransformer = FakeST
    llm = client(EMBED_BACKEND="sentence-transformers", EMBED_LOCAL_MODEL_NAME="tiny-model")

    async def twice():
        return await asyncio.gather(llm.embed(["ab"]), llm.embed(["abc", "a"]))

    assert run(twice()) == [[[2.0]], [[3.0], [1.0]]]
    assert loads == ["tiny-model"] and llm.embed_client.batches == []


# --- settings -------------------------------------------------------------------------------------------

def test_settings_defaults_and_validation():
    s = settings()
    assert (s.MAX_AGENT_STEPS, s.MAX_TOKENS_PER_QUERY, s.LLM_JSON_RETRIES, s.EMBED_BACKEND) == \
        (10, 12_000, 2, "sentence-transformers")                       # the index's backend (ADR-013)
    assert (s.QUERY_TIMEOUT_S, s.LLM_MAX_COMPLETION_TOKENS) == (60.0, 1024)
    with pytest.raises(ValueError):
        settings(EMBED_BACKEND="faiss")
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{k: v for k, v in REQUIRED.items() if k != "LLM_MODEL"})


def test_env_example_covers_every_setting():
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / ".env.example"
    keys = {line.split("=", 1)[0] for line in example.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.startswith("#")}
    assert keys == set(Settings.model_fields)


def test_get_settings_is_read_once(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", None)       # independent of a developer's .env
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


# --- added after the code and output reviews on #6 -------------------------------------------------------

def test_extract_json_returns_the_reply_when_nothing_decodes():
    assert extract_json("  I cannot answer that.  ") == "I cannot answer that."


def test_failed_requests_and_missing_usage_are_counted():
    llm = client(status_error(openai.AuthenticationError, 401),
                 SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None))
    with pytest.raises(FinSightError):
        run(llm.chat(USER))
    assert run(llm.chat(USER)) == "ok"
    assert llm.usage.calls == 2 and llm.usage.total_tokens == 0 and llm.usage.seconds > 0


def test_embedding_usage_and_reset():
    llm = client(EMBED_BATCH_SIZE=2)
    run(llm.embed(["a", "b", "c"]))
    assert llm.usage.embed_calls == 2 and llm.usage.embed_seconds > 0
    llm.usage.reset()
    assert (llm.usage.embed_calls, llm.usage.calls, llm.usage.embed_seconds) == (0, 0, 0.0)


class Closable:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def test_async_with_closes_each_http_client_once():
    chat_client, embed_client = Closable(), Closable()

    async def use():
        async with LLMClient(settings(), chat_client=chat_client, embed_client=embed_client) as llm:
            return llm

    run(use())
    assert (chat_client.closed, embed_client.closed) == (1, 1)
    shared = Closable()
    run(LLMClient(settings(), chat_client=shared, embed_client=shared).aclose())
    assert shared.closed == 1


def test_real_clients_close_cleanly_across_event_loops():
    llm = LLMClient(settings())
    run(llm.aclose())
    assert llm.client.is_closed() and llm.embed_client.is_closed()


@pytest.mark.parametrize("field", ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "SEC_USER_AGENT"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_required_settings_fail_in_settings_not_in_the_openai_client(field, value):
    with pytest.raises(ValueError, match=field):
        settings(**{field: value})


@pytest.mark.parametrize("typed", ["Sentence-Transformers", " sentence-transformers ", "OPENAI"])
def test_embed_backend_name_is_normalised(typed):
    assert settings(EMBED_BACKEND=typed).EMBED_BACKEND == typed.strip().lower()


def test_env_file_is_the_repo_root_whatever_the_working_directory():
    from pathlib import Path

    from core.settings import ENV_FILE
    assert ENV_FILE.is_absolute() and ENV_FILE == Path(__file__).resolve().parents[1] / ".env"
    assert Settings.model_config["env_file"] == ENV_FILE


def test_blank_embeddings_endpoint_in_env_example_falls_back_to_chat_endpoint():
    llm = LLMClient(settings(EMBED_BASE_URL="", EMBED_API_KEY=""))
    assert str(llm.embed_client.base_url).startswith("http://llm.test/v1") and llm.embed_client.api_key == "key"


def test_sec_settings_need_only_the_user_agent():
    # ingest/01 and ingest/05 must run with the LLM section of .env still empty
    assert SecSettings(_env_file=None, SEC_USER_AGENT=" team you@domain.com ").SEC_USER_AGENT == "team you@domain.com"


# --- added after the Week 0 codebase review -------------------------------------------------------------

def test_a_short_embeddings_reply_is_an_error_not_misaligned_vectors():
    llm = client()

    async def one_short(model, input):
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])        # 1 vector for 2 texts

    llm.embed_client.embeddings.create = one_short
    with pytest.raises(FinSightError) as err:
        run(llm.embed(["a", "b"]))
    assert err.value.error_type == "EmbeddingCountError" and "1 embeddings for 2 texts" in err.value.message


@pytest.mark.parametrize("failure", [ImportError("No module named 'sentence_transformers'"),
                                     OSError("couldn't reach huggingface.co")])
def test_a_local_model_that_cannot_load_is_a_typed_error(monkeypatch, failure):
    class Broken:
        def __init__(self, name):
            raise failure

    monkeypatch.setitem(sys.modules, "sentence_transformers", ModuleType("sentence_transformers"))
    sys.modules["sentence_transformers"].SentenceTransformer = Broken
    llm = client(EMBED_BACKEND="sentence-transformers", EMBED_LOCAL_MODEL_NAME="tiny-model")
    with pytest.raises(FinSightError) as err:
        run(llm.embed(["a"]))
    assert err.value.error_type == "EmbeddingModelError" and "tiny-model unavailable" in err.value.message



def test_a_reply_cut_off_at_the_token_cap_is_an_error():
    cut = reply('{"ticker": "MS')
    cut.choices[0].finish_reason = "length"
    with pytest.raises(FinSightError) as err:
        run(client(cut).chat(USER))
    assert err.value.error_type == "TruncatedResponseError" and "1024-token cap" in err.value.message


# --- W1-01: hosted endpoint (CA certificate, thinking switch, <think> stripping, secret key) ----------------

HTTPS_URL = {"LLM_BASE_URL": "https://endpoint.test:1033/v1"}


def ca_bundle() -> str:
    return certifi.where()


def test_ca_cert_unset_or_blank_means_system_cas():
    assert settings().LLM_CA_CERT is None
    assert settings(LLM_CA_CERT="").LLM_CA_CERT is None and settings(LLM_CA_CERT="   ").LLM_CA_CERT is None


def test_ca_cert_must_be_an_existing_file(tmp_path):
    with pytest.raises(ValueError, match="no certificate file at .*missing.crt"):
        settings(**HTTPS_URL, LLM_CA_CERT=str(tmp_path / "missing.crt"))
    with pytest.raises(ValueError, match="no certificate file"):
        settings(**HTTPS_URL, LLM_CA_CERT=str(tmp_path))                 # a folder is not a certificate


def test_relative_ca_cert_paths_are_taken_from_the_repo_root():
    s = settings(**HTTPS_URL, LLM_CA_CERT="README.md")                # any repo file: only the path is checked
    assert s.LLM_CA_CERT == REPO_ROOT / "README.md" and REPO_ROOT == Path(__file__).resolve().parents[1]


def test_ca_cert_with_a_plain_http_url_is_a_configuration_error():
    with pytest.raises(ValueError, match="not an https:// URL"):
        settings(LLM_CA_CERT=ca_bundle())                               # REQUIRED uses http://
    assert settings(LLM_BASE_URL="HTTPS://Endpoint.test/v1", LLM_CA_CERT=ca_bundle()).LLM_CA_CERT is not None


@pytest.mark.parametrize("broken", [{"EMBED_BACKEND": "faiss"},
                                    {"LLM_CA_CERT": "README.md"},                 # CA with an http:// URL
                                    {**HTTPS_URL, "LLM_CA_CERT": "missing.crt"}])
def test_the_api_key_never_appears_in_a_repr_or_a_validation_error(broken):
    secret = "sk-team-key-0123456789"
    assert secret not in repr(settings(LLM_API_KEY=secret, EMBED_API_KEY=secret))
    with pytest.raises(ValueError) as err:
        settings(LLM_API_KEY=secret, **broken)
    assert secret not in str(err.value) and secret not in str(err.value.errors())    # errors(): untruncated inputs
    assert LLMClient(settings(LLM_API_KEY=secret)).client.api_key == secret   # the client still gets the key


def test_build_ssl_context_trusts_only_the_given_file_and_checks_hostnames():
    path = Path(ca_bundle())
    ctx = build_ssl_context(path)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    assert ctx.cert_store_stats()["x509_ca"] == path.read_text(encoding="utf-8").count("BEGIN CERTIFICATE")


def test_a_file_that_is_not_a_pem_certificate_is_a_typed_error(tmp_path):
    bad = tmp_path / "group-ca.crt"
    bad.write_text("this is the learner guide, not a certificate", encoding="utf-8")
    with pytest.raises(FinSightError) as err:
        build_ssl_context(bad)
    assert err.value.error_type == "TLSConfigError" and "group-ca.crt" in err.value.message


def test_clients_verify_against_the_ca_file(monkeypatch):
    made = []
    real = openai.DefaultAsyncHttpxClient

    def recording(verify):
        made.append(verify)
        return real(verify=verify)

    monkeypatch.setattr(core.llm.openai, "DefaultAsyncHttpxClient", recording)
    llm = LLMClient(settings(**HTTPS_URL, LLM_CA_CERT=ca_bundle()))
    assert len(made) == 2 and made[0] is made[1]                       # chat and embeddings share one context
    assert made[0].check_hostname
    run(llm.aclose())

    made.clear()                                                        # a separate embeddings host keeps system CAs
    llm = LLMClient(settings(**HTTPS_URL, LLM_CA_CERT=ca_bundle(), EMBED_BASE_URL="https://embed.test/v1"))
    assert len(made) == 1
    run(llm.aclose())

    made.clear()                                                        # no certificate: openai's own client
    run(LLMClient(settings()).aclose())
    assert made == []


def test_a_failed_tls_handshake_is_reported_as_a_certificate_problem():
    request = httpx.Request("POST", "https://endpoint.test/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    transport = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]", request=request)
    transport.__cause__ = ssl.SSLCertVerificationError("unable to get local issuer certificate")
    error.__cause__ = transport
    llm = client(error)
    with pytest.raises(FinSightError) as err:
        run(llm.chat(USER))
    assert err.value.error_type == "TLSError" and "LLM_CA_CERT" in err.value.message


def test_thinking_is_switched_off_by_default():
    llm = client(reply("OK"))
    run(llm.chat(USER))
    assert llm.client.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("typed, sent", [("true", True), ("FALSE", False), (True, True)])
def test_thinking_switch_is_sent_as_configured(typed, sent):
    llm = client(reply("OK"), LLM_ENABLE_THINKING=typed)
    run(llm.chat(USER))
    assert llm.client.requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is sent


def test_a_blank_thinking_switch_sends_no_extra_field():
    llm = client(reply("OK"), LLM_ENABLE_THINKING="")
    run(llm.chat(USER))
    assert llm.settings.LLM_ENABLE_THINKING is None and "extra_body" not in llm.client.requests[0]


@pytest.mark.parametrize("raw, answer", [
    ("<think>\n\n</think>\n\nOPERATIONAL", "OPERATIONAL"),               # Qwen3 with thinking off
    ("<think>Revenue 245,122 / ... maybe 44.5?</think>The margin is 44.6%.", "The margin is 44.6%."),
    ("<THINK>a</THINK>b<think>c</think>d", "bd"),
    ("draft reasoning without an opening tag</think>Answer", "Answer"),  # opening tag was in the prompt
    ("Answer<think>unfinished reasoning", "Answer"),
    ("no reasoning at all", "no reasoning at all"),
])
def test_strip_thinking(raw, answer):
    assert strip_thinking(raw) == answer


def test_chat_returns_the_answer_without_reasoning():
    assert run(client(reply("<think>\nLet me check.\n</think>\n\nOPERATIONAL")).chat(USER)) == "OPERATIONAL"


def test_a_reply_that_is_only_reasoning_is_empty():
    with pytest.raises(FinSightError) as err:
        run(client(reply("<think>I am still thinking about it")).chat(USER))
    assert err.value.error_type == "EmptyResponseError"


def test_extract_json_never_picks_an_object_from_the_reasoning():
    raw = '<think>Draft: {"ticker": "AAPL", "year": 2023}. No, the question says Microsoft.</think>' \
          '{"ticker": "MSFT", "year": 2024}'
    assert Filing.model_validate_json(extract_json(raw)) == Filing(ticker="MSFT", year=2024)


def smoke_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_llm_script", REPO_ROOT / "scripts" / "smoke_llm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("url, shown", [
    ("https://203.0.113.7:1033/v1", "https://<endpoint>:1033/v1"),         # the group's address stays private
    ("https://endpoint.example.org/v1", "https://<endpoint>/v1"),
    ("http://localhost:11434/v1", "http://localhost:11434/v1"),
    ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1"),
])
def test_smoke_output_hides_a_remote_host(url, shown):
    assert smoke_script().shown_url(url) == shown


def test_smoke_output_names_how_tls_is_verified():
    smoke = smoke_script()
    assert smoke.tls_line(settings(**HTTPS_URL, LLM_CA_CERT=ca_bundle())).startswith("verified against cacert.pem")
    assert smoke.tls_line(settings(**HTTPS_URL)) == "verified against the system CAs"
    assert smoke.tls_line(settings()).startswith("none (plain HTTP")
