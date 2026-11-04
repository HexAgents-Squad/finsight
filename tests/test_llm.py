"""core/llm.py and core/settings.py against fake OpenAI clients (offline, no Ollama needed)."""
import asyncio
import sys
from types import ModuleType, SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from core.exceptions import FinSightError
from core.llm import LLMClient, extract_json
from core.settings import Settings, get_settings

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
    return LLMClient(settings(**overrides), chat_client=FakeChat(*outputs), embed_client=FakeEmbeddings())


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
    assert (s.MAX_AGENT_STEPS, s.MAX_TOKENS_PER_QUERY, s.LLM_JSON_RETRIES, s.EMBED_BACKEND) == (10, 12_000, 2, "openai")
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
    for key, value in REQUIRED.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
