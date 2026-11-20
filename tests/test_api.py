"""W2-09: the FastAPI service. Offline: the real graph code on scripted nodes (no endpoint, no index, no .env)."""
from __future__ import annotations

import asyncio
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from agent import graph as g
from agent.contract import AnswerContract, refusal_answer
from agent.state import Plan, PlanStep, Verdict
from agent.stubs import stub_nodes
from api import main as api
from api.cache import ReplyCache, cache_key, normalise
from api.sessions import Sessions

LIMITS = g.Limits(max_steps=10, max_tokens=12_000, timeout_s=10)
QUESTION = "What was Apple's total revenue in fiscal 2024?"


class FakeAgent:
    """Scripted nodes: a verified answer, a refusal for questions starting "Should I", an error for "Boom";
    records each question and the history it was asked with, and how many runs overlap."""

    def __init__(self, delay: float = 0.0):
        self.calls: list[tuple[str, list[str]]] = []
        self.delay, self.active, self.most_active = delay, 0, 0

    def nodes(self) -> g.Nodes:
        async def planner(state):
            self.calls.append((state["question"], list(state.get("history") or [])))
            self.active += 1
            self.most_active = max(self.most_active, self.active)
            try:
                await asyncio.sleep(self.delay)
            finally:
                self.active -= 1
            if state["question"].startswith("Boom"):
                raise RuntimeError("secret internals: key=abc")
            if state["question"].startswith("Should I"):
                return {"plan": Plan(reasoning="x", refuse=True, refusal_rule="R1", steps=[])}
            return {"plan": Plan(reasoning="x", refuse=False, refusal_rule=None,
                                 steps=[PlanStep(tool="search_filings", purpose="x", arguments={"query": "x"})])}

        async def executor(state):
            return {}

        async def composer(state):
            plan = state["plan"]
            if plan.refuse:
                text = refusal_answer("R1", "Apple's total revenue for fiscal 2024")
                return {"draft": AnswerContract(answer=text, verified=False, refused=True, refusal_rule="R1")}
            return {"draft": AnswerContract(answer=f"Answer to: {state['question']}", verified=False, refused=False)}

        async def critic(state):
            return {"verdict": Verdict(violations=[])}

        return g.Nodes(planner, executor, composer, critic)


def client(agent: FakeAgent | None = None, **config) -> tuple[TestClient, FakeAgent]:
    agent = agent or FakeAgent()
    app = api.create_app(graph=g.build_graph(agent.nodes()), config=api.ApiConfig(limits=LIMITS, model="fake-model",
                                                                                   **config))
    return TestClient(app), agent


def ask(c: TestClient, question=QUESTION, session=None, **body):
    headers = {api.SESSION_HEADER: session} if session else {}
    return c.post("/ask", json={"question": question, **body}, headers=headers)


# ----------------------------------------------------------------------------------------------- /health, /ask
def test_health_reports_the_model_and_counts():
    c, _ = client()
    with c:
        body = c.get("/health").json()
    assert body == {"status": "ok", "warm_up": "ok", "model": "fake-model", "cache_entries": 0, "sessions": 0}


def test_ask_returns_the_answer_contract_and_a_session():
    c, agent = client()
    with c:
        r = ask(c)
    body = r.json()
    assert r.status_code == 200 and r.headers[api.SESSION_HEADER] == body["session_id"]
    assert body["answer"]["answer"] == f"Answer to: {QUESTION}" and body["answer"]["verified"] is True
    assert body["cached"] is False and body["history"] == []
    assert body["run"]["verdict"] == "verified" and body["run"]["visits"]["planner"] == 1
    assert agent.calls == [(QUESTION, [])]


# ----------------------------------------------------------------------------------------------- cache
def test_the_same_question_is_answered_from_the_cache():
    c, agent = client()
    with c:
        first = ask(c, session="a")
        again = ask(c, "  what was APPLE’S total revenue in fiscal 2024  ", session="b")   # normalised
        health = c.get("/health").json()
    assert first.json()["cached"] is False and again.json()["cached"] is True
    assert again.json()["answer"] == first.json()["answer"] and len(agent.calls) == 1
    assert health["cache_entries"] == 1


def test_the_history_is_part_of_the_key():
    c, agent = client()
    with c:
        ask(c, "And in fiscal 2023?", history=["What was Apple's revenue in fiscal 2024?"])
        r = ask(c, "And in fiscal 2023?", history=["What was Microsoft's revenue in fiscal 2024?"])
    assert r.json()["cached"] is False and len(agent.calls) == 2


def test_an_unverified_answer_is_not_cached_and_a_refusal_is():
    c = TestClient(api.create_app(graph=g.build_graph(stub_nodes()), config=api.ApiConfig(limits=LIMITS)))
    with c:
        assert [ask(c).json()["cached"] for _ in range(2)] == [False, False]          # the stubs verify nothing
    c, agent = client()
    with c:
        assert [ask(c, "Should I buy Apple stock?").json()["cached"] for _ in range(2)] == [False, True]
    assert len(agent.calls) == 1


def test_a_cache_with_no_room_keeps_nothing():
    c, agent = client(cache_max_entries=0)
    with c:
        assert [ask(c).json()["cached"] for _ in range(2)] == [False, False]
    assert len(agent.calls) == 2


# ----------------------------------------------------------------------------------------------- sessions
def test_a_follow_up_gets_the_sessions_earlier_questions():
    c, agent = client()
    with c:
        sid = ask(c).json()["session_id"]
        r = ask(c, "And in fiscal 2023?", session=sid)
    assert r.json()["history"] == [QUESTION] and agent.calls[-1] == ("And in fiscal 2023?", [QUESTION])


def test_a_cache_hit_still_counts_as_the_sessions_turn():
    """Why the API keeps the history (ADR-032): a checkpointer thread would miss a turn the cache answered."""
    c, agent = client()
    with c:
        ask(c, session="first")                                  # the graph answers; the cache keeps it
        assert ask(c, session="second").json()["cached"] is True
        ask(c, "And in fiscal 2023?", session="second")
    assert agent.calls[-1] == ("And in fiscal 2023?", [QUESTION])


def test_the_last_question_sent_again_is_the_same_turn():
    """A client's retry: no duplicate in the session's memory, and the cache answers it."""
    c, agent = client()
    with c:
        sid = ask(c).json()["session_id"]
        again = ask(c, QUESTION + "  ", session=sid)
        follow = ask(c, "And in fiscal 2023?", session=sid)
    assert again.json()["cached"] is True and again.json()["history"] == []
    assert follow.json()["history"] == [QUESTION] and len(agent.calls) == 2


def test_a_body_history_replaces_the_sessions():
    c, agent = client()
    with c:
        ask(c, session="s")
        r = ask(c, "And in fiscal 2023?", session="s", history=["What was Costco's revenue in fiscal 2024?"])
    assert r.json()["history"] == ["What was Costco's revenue in fiscal 2024?"]


def test_sessions_are_separate():
    c, agent = client()
    with c:
        ask(c, session="alice")
        r = ask(c, "And in fiscal 2023?", session="bob")
    assert r.json()["history"] == []


def test_the_history_is_capped():
    c, agent = client()
    long = [f"And the margin, take {chr(97 + i)}?" for i in range(15)]
    with c:
        r = ask(c, history=long)
    assert r.json()["history"] == g.cap_history(long) and len(r.json()["history"]) == g.MAX_HISTORY_TURNS


@pytest.mark.anyio
async def test_a_sessions_questions_run_one_at_a_time_and_others_alongside():
    agent = FakeAgent(delay=0.05)
    app = api.create_app(graph=g.build_graph(agent.nodes()), config=api.ApiConfig(limits=LIMITS, max_concurrent=2))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        same = [c.post("/ask", json={"question": f"Q{i} about Apple?"}, headers={api.SESSION_HEADER: "s"})
                for i in range(3)]
        await asyncio.gather(*same)
        assert agent.most_active == 1                               # one session: never two at once
        agent.most_active = 0
        others = [c.post("/ask", json={"question": f"R{i} about Apple?"}, headers={api.SESSION_HEADER: f"s{i}"})
                  for i in range(3)]
        await asyncio.gather(*others)
        assert agent.most_active == 2                               # other sessions: up to max_concurrent


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ----------------------------------------------------------------------------------------------- rate limit
def test_the_rate_limit_is_per_session():
    c, _ = client(rate_limit="2/minute")
    with c:
        codes = [ask(c, f"Q{i} about Apple?", session="s").status_code for i in range(3)]
        limited = ask(c, "Q9 about Apple?", session="s")
        other = ask(c, "Q1 about Apple?", session="other")
    assert codes == [200, 200, 429] and other.status_code == 200
    assert "Retry-After" in limited.headers


def test_without_a_session_the_limit_is_per_address():
    c, _ = client(rate_limit="1/minute")
    with c:
        assert [ask(c, f"Q{i} about Apple?").status_code for i in range(2)] == [200, 429]


def test_a_cached_answer_still_counts_toward_the_limit():
    c, agent = client(rate_limit="2/minute")
    with c:
        assert [ask(c, session="s").status_code for _ in range(3)] == [200, 200, 429]
    assert len(agent.calls) == 1


# ----------------------------------------------------------------------------------------------- errors
@pytest.mark.parametrize("body", [
    {"question": ""}, {"question": "   "}, {"question": "x" * 1_001},                    # ADR-032: 1,000 at most
    {"question": "q", "history": ["x"] * (api.MAX_HISTORY + 1)}, {"question": "q", "history": [" "]},
    {"question": "q", "extra": 1}, {},
])
def test_an_invalid_request_is_422(body):
    c, agent = client()
    with c:
        assert c.post("/ask", json=body).status_code == 422
    assert agent.calls == []


@pytest.mark.parametrize("session", ["has space", "x" * 65, "semi;colon"])
def test_an_invalid_session_id_is_422(session):
    c, agent = client()
    with c:
        assert ask(c, session=session).status_code == 422
    assert agent.calls == []


def test_an_error_inside_the_agent_is_a_generic_500():
    c, _ = client()
    with TestClient(c.app, raise_server_exceptions=False) as c2:
        r = ask(c2, "Boom, what was Apple's revenue?")
        after = ask(c2, session="next")                           # the app keeps serving
    assert r.status_code == 500 and r.json() == {"detail": "internal error"} and "secret" not in r.text
    assert after.status_code == 200


def test_the_session_lock_is_released_after_an_error():
    c, agent = client()
    with TestClient(c.app, raise_server_exceptions=False) as c2:
        assert ask(c2, "Boom, what was Apple's revenue?", session="s").status_code == 500
        assert ask(c2, session="s").status_code == 200


# ----------------------------------------------------------------------------------------------- start-up
def test_health_shows_the_warm_up():
    started, release = threading.Event(), threading.Event()

    def slow_warm_up():
        started.set()
        release.wait(5)
        return {"error": {"type": "data_missing", "message": "no index"}}

    agent = FakeAgent()
    app = api.create_app(graph=g.build_graph(agent.nodes()), config=api.ApiConfig(limits=LIMITS), warm_up=slow_warm_up)
    with TestClient(app) as c:
        started.wait(5)
        assert c.get("/health").json()["status"] == "starting"
        assert ask(c).status_code == 200                           # questions are served while it warms up
        release.set()
        for _ in range(100):
            if c.get("/health").json()["status"] != "starting":
                break
            threading.Event().wait(0.02)
        body = c.get("/health").json()
    assert body["status"] == "degraded" and body["warm_up"] == "data_missing"


def test_the_real_app_builds_its_graph_from_settings(monkeypatch):
    """create_app() with no graph: settings, one LLM client opened and closed with the app, the real nodes."""
    import agent.nodes
    import core.llm
    import core.settings
    import tools.registry

    class Settings:
        API_RATE_LIMIT, API_CACHE_TTL_S, API_CACHE_MAX_ENTRIES, API_MAX_CONCURRENT = "5/minute", 60.0, 10, 1
        API_SESSION_TTL_S, API_MAX_SESSIONS = 60.0, 10
        MAX_AGENT_STEPS, MAX_TOKENS_PER_QUERY, QUERY_TIMEOUT_S = 10, 12_000, 10.0
        LLM_MODEL, LANGFUSE_ENABLED = "Qwen/Qwen3-8B", False

    events = []

    class Client:
        def __init__(self, settings):
            events.append("open")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            events.append("close")

    fake = FakeAgent()
    monkeypatch.setattr(core.settings, "get_settings", lambda: Settings())
    monkeypatch.setattr(core.llm, "LLMClient", Client)
    monkeypatch.setattr(agent.nodes, "real_nodes", lambda llm: events.append("nodes") or fake.nodes())
    monkeypatch.setattr(tools.registry, "warm_up", lambda: {"ok": True})
    with TestClient(api.create_app()) as c:
        assert ask(c).status_code == 200
        for _ in range(100):
            if c.get("/health").json()["status"] == "ok":
                break
            threading.Event().wait(0.02)
        body = c.get("/health").json()
    assert body["model"] == "Qwen/Qwen3-8B" and body["status"] == "ok"
    assert events == ["open", "nodes", "close"]


def test_importing_the_module_needs_no_env():
    assert api.app.title == "FinSight"


def test_a_traced_run_is_tagged_with_its_session(monkeypatch):
    seen = {}
    real_run = api.run

    async def spy(*args, **kwargs):
        seen["trace"] = kwargs.get("trace")
        kwargs["trace"] = None
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(api, "run", spy)
    c, _ = client(trace=True)
    with c:
        ask(c, session="abc")
    assert seen["trace"].run_id == "api-abc"


# ----------------------------------------------------------------------------------------------- cache, sessions
def test_normalise():
    assert normalise("  What was  Apple’s revenue?? ") == normalise("what was apple's revenue") == \
        "what was apple's revenue"
    assert cache_key("Q?", ["A.", "b"]) == ("q", "a", "b")


def test_the_cache_expires_and_drops_the_least_recently_used():
    now = [0.0]
    cache = ReplyCache(max_entries=2, ttl_s=10, clock=lambda: now[0])
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1                                      # a is now the most recently used
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and len(cache) == 2
    now[0] = 10.0
    assert cache.get("a") is None and cache.get("c") is None
    assert not ReplyCache(0, 10).enabled and not ReplyCache(5, 0).enabled
    no_lifetime = ReplyCache(5, 0)
    no_lifetime.put("a", 1)
    assert len(no_lifetime) == 0                                    # a disabled cache holds nothing (health counts)


def test_sessions_expire_and_a_running_one_is_kept():
    now = [0.0]
    sessions = Sessions(max_sessions=2, ttl_s=10, clock=lambda: now[0])
    first = sessions.get("a")
    sessions.remember("a", ["q1"])
    assert sessions.get("a") is first and first.history == ["q1"]
    sessions.get("b")
    asyncio.run(first.lock.acquire())                               # a's question is running
    sessions.get("c")                                               # over the limit: b goes, a stays
    assert sessions.get("a") is first and len(sessions) == 2
    first.lock.release()
    now[0] = 20.0
    assert sessions.get("a") is not first and sessions.get("a").history == []        # expired: a fresh session
    many = [f"And the margin, take {chr(97 + i)}?" for i in range(10)]
    sessions.remember("a", many)
    assert sessions.get("a").history == g.cap_history(many)
