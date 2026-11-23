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
from api.sessions import SessionIds, Sessions

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


def issued(c: TestClient) -> str:
    """A session id this app issued (as a first answer would hand it out)."""
    return c.app.state.finsight.ids.issue()


def ask(c: TestClient, question=QUESTION, session=None, **body):
    headers = {api.SESSION_HEADER: session} if session else {}
    return c.post("/ask", json={"question": question, **body}, headers=headers)


# ----------------------------------------------------------------------------------------------- /health, /ask
def test_health_reports_the_model_and_counts():
    c, _ = client()
    with c:
        body = c.get("/health").json()
    assert body == {"status": "ok", "warm_up": "ok", "model": "fake-model", "cache_entries": 0, "cache_hits": 0,
                    "cache_misses": 0, "cache_hit_rate": None, "sessions": 0}


def test_ask_returns_the_answer_contract_and_a_signed_session():
    c, agent = client()
    with c:
        r = ask(c)
        valid = c.app.state.finsight.ids.valid(r.json()["session_id"])
    body = r.json()
    assert r.status_code == 200 and r.headers[api.SESSION_HEADER] == body["session_id"] and valid
    assert body["answer"]["answer"] == f"Answer to: {QUESTION}" and body["answer"]["verified"] is True
    assert body["cached"] is False and body["history"] == [] and body["seconds"] >= 0
    assert body["run"]["verdict"] == "verified" and body["run"]["visits"]["planner"] == 1
    assert agent.calls == [(QUESTION, [])] and "Retry-After" not in r.headers


# ----------------------------------------------------------------------------------------------- cache
def test_the_same_question_is_answered_from_the_cache():
    c, agent = client()
    with c:
        first = ask(c)
        again = ask(c, "  what was APPLE’S total revenue in fiscal 2024  ")          # normalised, a new session
        health = c.get("/health").json()
    assert first.json()["cached"] is False and again.json()["cached"] is True
    assert again.json()["answer"] == first.json()["answer"] and len(agent.calls) == 1
    assert again.json()["run"] == first.json()["run"] and health["cache_entries"] == 1
    # W3-07: the hit rate since start-up (one question answered by the agent, one from the cache)
    assert (health["cache_hits"], health["cache_misses"], health["cache_hit_rate"]) == (1, 1, 0.5)


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
    assert r.json()["session_id"] == sid


def test_a_cache_hit_still_counts_as_the_sessions_turn():
    """Why the API keeps the history (ADR-032): a checkpointer thread would miss a turn the cache answered."""
    c, agent = client()
    with c:
        ask(c)                                                    # the graph answers; the cache keeps it
        second = ask(c)
        assert second.json()["cached"] is True
        ask(c, "And in fiscal 2023?", session=second.json()["session_id"])
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
        sid = ask(c).json()["session_id"]
        r = ask(c, "And in fiscal 2023?", session=sid, history=["What was Costco's revenue in fiscal 2024?"])
    assert r.json()["history"] == ["What was Costco's revenue in fiscal 2024?"]


def test_sessions_are_separate():
    c, agent = client()
    with c:
        ask(c, session=issued(c))
        r = ask(c, "And in fiscal 2023?", session=issued(c))
    assert r.json()["history"] == []


def test_an_id_this_server_did_not_issue_starts_a_new_session():
    """Review: a client minting ids would get a fresh rate-limit bucket per request and could flood the store."""
    c, agent = client()
    other_server = SessionIds(b"another secret").issue()
    with c:
        sid = ask(c).json()["session_id"]
        for forged in (other_server, "made-up-id", sid[:-1] + ("0" if sid[-1] != "0" else "1")):
            r = ask(c, "And in fiscal 2023?", session=forged)
            assert r.json()["session_id"] not in (forged, sid) and r.json()["history"] == []


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
    ids = app.state.finsight.ids
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        one = ids.issue()
        same = [c.post("/ask", json={"question": f"Q{i} about Apple?"}, headers={api.SESSION_HEADER: one})
                for i in range(3)]
        assert all(r.status_code == 200 for r in await asyncio.gather(*same))
        assert agent.most_active == 1                               # one session: never two at once
        agent.most_active = 0
        others = [c.post("/ask", json={"question": f"R{i} about Apple?"}, headers={api.SESSION_HEADER: ids.issue()})
                  for i in range(3)]
        assert all(r.status_code == 200 for r in await asyncio.gather(*others))
        assert agent.most_active == 2                               # other sessions: up to max_concurrent


@pytest.mark.anyio
async def test_a_question_waiting_too_long_for_a_slot_is_503():
    agent = FakeAgent(delay=0.3)
    app = api.create_app(graph=g.build_graph(agent.nodes()),
                         config=api.ApiConfig(limits=LIMITS, max_concurrent=1, queue_timeout_s=0.05))
    ids = app.state.finsight.ids
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        first, second = await asyncio.gather(*[
            c.post("/ask", json={"question": f"Q{i} about Apple?"}, headers={api.SESSION_HEADER: ids.issue()})
            for i in range(2)])
    codes = sorted([first.status_code, second.status_code])
    busy = first if first.status_code == 503 else second
    assert codes == [200, 503] and busy.headers["Retry-After"] and "busy" in busy.json()["detail"]
    assert ids.valid(busy.headers[api.SESSION_HEADER])              # the issued id is not lost with the 503


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ----------------------------------------------------------------------------------------------- rate limits
def test_the_rate_limit_is_per_session():
    c, _ = client(rate_limit="2/minute")
    with c:
        sid, other = issued(c), issued(c)
        codes = [ask(c, f"Q{i} about Apple?", session=sid).status_code for i in range(3)]
        limited = ask(c, "Q9 about Apple?", session=sid)
        fine = ask(c, "Q1 about Apple?", session=other)
    assert codes == [200, 200, 429] and fine.status_code == 200
    assert limited.headers["Retry-After"] == "60" and limited.json()["detail"].startswith("rate limit exceeded")


def test_new_conversations_behind_one_address_are_limited_by_the_address_only():
    """PR review: behind the UI's server every user shares one address; each new conversation gets its own session
    bucket, and the per-address limit is the ceiling."""
    c, _ = client(rate_limit="1/minute", address_rate_limit="12/minute")
    with c:
        codes = [ask(c, f"Q{i} about Apple?").status_code for i in range(12)]
        over = ask(c, "Q12 about Apple?", session="minted-by-the-client")
    assert codes == [200] * 12 and over.status_code == 429


def test_a_forged_id_is_not_a_rate_limit_key():
    """An id this server did not issue is a new conversation each time (its own bucket), never a key a client
    chooses; the per-address limit stays the ceiling."""
    c, _ = client(rate_limit="1/minute", address_rate_limit="10/minute")
    with c:
        codes = [ask(c, f"Q{i} about Apple?", session="minted-by-the-client").status_code for i in range(3)]
    assert codes == [200, 200, 200]


def test_a_session_refusal_costs_no_address_hit_and_a_bad_header_costs_none():
    c, _ = client(rate_limit="1/minute", address_rate_limit="3/minute")
    with c:
        sid = issued(c)
        assert [ask(c, f"Q{i} about Apple?", session=sid).status_code for i in range(2)] == [200, 429]
        assert [ask(c, "Q about Apple?", session="bad id!").status_code for _ in range(3)] == [422, 422, 422]
        assert [ask(c, f"R{i} about Apple?").status_code for i in range(3)] == [200, 200, 429]


def test_the_address_limit_holds_across_sessions():
    """Review: new sessions per request would otherwise each get a fresh per-session bucket."""
    c, _ = client(rate_limit="5/minute", address_rate_limit="3/minute")
    with c:
        codes = [ask(c, f"Q{i} about Apple?", session=issued(c)).status_code for i in range(4)]
    assert codes == [200, 200, 200, 429]


def test_a_cached_answer_still_counts_toward_the_limit():
    c, agent = client(rate_limit="2/minute")
    with c:
        sid = issued(c)
        assert [ask(c, session=sid).status_code for _ in range(3)] == [200, 200, 429]
    assert len(agent.calls) == 1


@pytest.mark.parametrize("field", ["rate_limit", "address_rate_limit"])
def test_a_malformed_rate_limit_fails_at_once(field):
    """Review: slowapi would log a bad limit on every request and apply none."""
    with pytest.raises(ValueError):
        api.ApiConfig(**{field: "ten per minute"})


@pytest.mark.parametrize("name", ["API_RATE_LIMIT", "API_ADDRESS_RATE_LIMIT"])
def test_the_settings_refuse_a_malformed_rate_limit(name):
    from pydantic import ValidationError

    from tests.test_llm import settings
    with pytest.raises(ValidationError):
        settings(**{name: "ten per minute"})
    assert settings(**{name: "100/hour; 10/minute"})


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


@pytest.mark.parametrize("session", ["has space", "x" * 129, "semi;colon"])
def test_a_malformed_session_header_is_422(session):
    c, agent = client()
    with c:
        assert ask(c, session=session).status_code == 422
    assert agent.calls == []


def test_a_non_ascii_session_header_is_422_not_500():
    """PR review: hmac.compare_digest raised TypeError on a non-ASCII header: a 500 and a traceback per request."""
    c, agent = client()
    with TestClient(c.app, raise_server_exceptions=False) as c2:
        r = c2.post("/ask", json={"question": QUESTION},
                    headers=[(api.SESSION_HEADER.encode(), ("a" * 32 + ".").encode() + "éé".encode("latin-1"))])
    assert r.status_code == 422 and agent.calls == []
    assert not SessionIds().valid("a" * 32 + ".éé")


def test_a_body_too_large_is_413():
    c, agent = client()
    with c:
        r = c.post("/ask", content=b'{"question": "' + b"x" * api.MAX_BODY_BYTES + b'"}',
                   headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json() == {"detail": "request body too large"} and agent.calls == []


def test_a_body_history_ending_with_the_question_is_the_same_turn():
    c, agent = client()
    with c:
        r = ask(c, "And in fiscal 2023?", history=[QUESTION, "And in fiscal 2023?"])
    assert r.json()["history"] == [QUESTION] and agent.calls == [("And in fiscal 2023?", [QUESTION])]


@pytest.mark.parametrize("config", [{"max_concurrent": 0}, {"queue_timeout_s": 0}])
def test_a_config_without_room_to_run_fails_at_once(config):
    with pytest.raises(ValueError):
        api.ApiConfig(**config)


def test_an_error_inside_the_agent_is_a_generic_500():
    c, _ = client()
    with TestClient(c.app, raise_server_exceptions=False) as c2:
        r = ask(c2, "Boom, what was Apple's revenue?")
        after = ask(c2)                                           # the app keeps serving
    assert r.status_code == 500 and r.json() == {"detail": "internal error"} and "secret" not in r.text
    assert after.status_code == 200


def test_the_session_is_released_after_an_error():
    c, agent = client()
    with TestClient(c.app, raise_server_exceptions=False) as c2:
        sid = issued(c2)
        assert ask(c2, "Boom, what was Apple's revenue?", session=sid).status_code == 500
        assert ask(c2, session=sid).status_code == 200
        assert c2.app.state.finsight.sessions._sessions[sid].users == 0


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
        starting = c.get("/health")
        assert starting.status_code == 503 and starting.json()["status"] == "starting"     # a healthcheck waits
        assert ask(c).status_code == 200                           # questions are served while it warms up
        release.set()
        for _ in range(100):
            if c.get("/health").json()["status"] != "starting":
                break
            threading.Event().wait(0.02)
        degraded = c.get("/health")
    body = degraded.json()
    assert degraded.status_code == 503 and body["status"] == "degraded" and body["warm_up"] == "data_missing"


def test_the_real_app_builds_its_graph_from_settings(monkeypatch):
    """create_app() with no graph: settings (limits included), one LLM client opened and closed, the real nodes."""
    import agent.nodes
    import core.llm
    import core.settings
    import tools.registry

    class Settings:
        API_RATE_LIMIT, API_ADDRESS_RATE_LIMIT, API_QUEUE_TIMEOUT_S = "2/minute", "60/minute", 5.0
        API_CACHE_TTL_S, API_CACHE_MAX_ENTRIES, API_MAX_CONCURRENT = 60.0, 10, 1
        API_SESSION_TTL_S, API_MAX_SESSIONS, API_SESSION_SECRET = 60.0, 10, None
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
    monkeypatch.setattr(tools.registry, "warm_up", lambda: events.append("index") or {"ok": True})
    monkeypatch.setattr(agent.nodes, "warm_up", lambda: events.append("agent"))       # W3-07: prompts, configs
    import core.tokens
    monkeypatch.setattr(core.tokens, "count_tokens", lambda text: events.append("tokenizer") or 1)
    with TestClient(api.create_app()) as c:
        sid = ask(c).json()["session_id"]
        codes = [ask(c, f"Q{i} about Apple?", session=sid).status_code for i in range(3)]
        for _ in range(100):
            if c.get("/health").json()["status"] == "ok":
                break
            threading.Event().wait(0.02)
        body = c.get("/health").json()
    assert body["model"] == "Qwen/Qwen3-8B" and body["status"] == "ok"
    assert codes == [200, 200, 429]                                 # the settings' 2/minute is the one applied
    # the warm-up loads the index, the agent's one-time loads, then the executor's tokenizer (off the event loop);
    # the client closes last
    assert events[:2] == ["open", "nodes"] and events[-1] == "close"
    assert [e for e in events if e in ("index", "agent", "tokenizer")] == ["index", "agent", "tokenizer"]


def test_a_shared_secret_makes_ids_valid_across_apps():
    one = api.create_app(graph=g.build_graph(FakeAgent().nodes()), config=api.ApiConfig(session_secret=b"s3cret"))
    two = api.create_app(graph=g.build_graph(FakeAgent().nodes()), config=api.ApiConfig(session_secret=b"s3cret"))
    assert two.state.finsight.ids.valid(one.state.finsight.ids.issue())


def test_importing_the_module_needs_no_env():
    assert api.app.title == "FinSight"


def test_a_traced_run_is_grouped_by_its_session(monkeypatch):
    seen = []
    real_run = api.run

    async def spy(*args, **kwargs):
        seen.append(kwargs.get("trace"))
        kwargs["trace"] = None
        return await real_run(*args, **kwargs)

    monkeypatch.setattr(api, "run", spy)
    c, _ = client(trace=True)
    with c:
        sid = issued(c)
        ask(c, "Q1 about Apple?", session=sid)
        ask(c, "Q2 about Apple?", session=sid)
    from api.sessions import session_digest
    assert [t.session_id for t in seen] == [session_digest(sid)] * 2 and {t.run_id for t in seen} == {"api"}
    assert sid not in {t.session_id for t in seen}                  # the id itself is a bearer token: never traced
    assert seen[0].question_id != seen[1].question_id


def test_the_session_id_never_reaches_the_logs(caplog):
    c, _ = client(rate_limit="1/minute")
    with c, caplog.at_level("DEBUG"):
        sid = issued(c)
        assert [ask(c, f"Q{i} about Apple?", session=sid).status_code for i in range(2)] == [200, 429]
    assert sid not in caplog.text and sid.split(".")[0] not in caplog.text


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
    assert (cache.hits, cache.misses, cache.hit_rate) == (2, 3, 0.4)    # an expired entry is a miss (W3-07)
    assert ReplyCache(5, 10).hit_rate is None                       # no lookup yet
    assert not ReplyCache(0, 10).enabled and not ReplyCache(5, 0).enabled
    no_lifetime = ReplyCache(5, 0)
    no_lifetime.put("a", 1)
    assert len(no_lifetime) == 0                                    # a disabled cache holds nothing (health counts)


def test_session_ids_are_signed():
    ids = SessionIds(b"k")
    sid = ids.issue()
    assert ids.valid(sid) and not SessionIds(b"other").valid(sid)
    assert not ids.valid(None) and not ids.valid("") and not ids.valid("abc.def") and not ids.valid(sid + ".x")


def test_sessions_expire_are_capped_and_remember_capped_history():
    now = [0.0]
    sessions = Sessions(max_sessions=2, ttl_s=10, clock=lambda: now[0])

    async def turns():
        async with sessions.hold("a") as a:
            sessions.remember("a", ["q1"])
        async with sessions.hold("a") as again:
            assert again is a and a.history == ["q1"]
        async with sessions.hold("b"), sessions.hold("c"):
            pass                                                    # over the limit: the oldest (a) goes
        sessions.remember("b", ["old question"])
        assert len(sessions) == 2 and "a" not in sessions._sessions
        now[0] = 20.0
        async with sessions.hold("b") as b:
            assert b.history == []                                  # expired: a fresh session
        many = [f"And the margin, take {chr(97 + i)}?" for i in range(10)]
        sessions.remember("b", many)
        assert sessions._sessions["b"].history == g.cap_history(many)

    asyncio.run(turns())


def test_a_session_waited_for_is_never_dropped():
    """Review: between a holder's release and its waiter's resumption the lock reads as free; a session with a
    waiter must still not be dropped, or the waiter's turn is lost and a third request runs alongside it."""
    sessions = Sessions(max_sessions=1, ttl_s=10, clock=lambda: 0.0)

    async def scenario():
        async def second_turn():
            async with sessions.hold("a") as s:
                sessions.remember("a", [*s.history, "q2"])

        first = sessions.hold("a")
        held = await first.__aenter__()                              # q1 holds a's lock
        waiter = asyncio.create_task(second_turn())
        await asyncio.sleep(0)                                       # q2 now waits on the lock
        sessions.remember("a", [*held.history, "q1"])
        await first.__aexit__(None, None, None)                      # q1 releases; q2 has not resumed yet
        assert not held.lock.locked() and held.users == 1            # the window: free lock, one waiter
        async with sessions.hold("b"):                               # over the limit: a has a waiter, so it stays
            assert "a" in sessions._sessions
        await waiter
        assert sessions._sessions["a"].history == ["q1", "q2"] and sessions._sessions["a"] is held

    asyncio.run(scenario())


def test_the_settings_refuse_a_short_session_secret():
    from pydantic import ValidationError

    from tests.test_llm import settings
    with pytest.raises(ValidationError) as err:
        settings(API_SESSION_SECRET="short-secret")
    assert "short-secret" not in str(err.value)                     # the value never appears in the message
    assert settings(API_SESSION_SECRET="").API_SESSION_SECRET is None
    assert settings(API_SESSION_SECRET="x" * 16).API_SESSION_SECRET.get_secret_value() == "x" * 16


def test_a_disabled_cache_counts_no_lookups():
    """Review: a disabled cache reported "0 of 3 questions (0%)", which reads as a cold cache."""
    cache = ReplyCache(0, 0)
    for key in ("a", "b", "c"):
        assert cache.get(key) is None
    assert (cache.hits, cache.misses, cache.hit_rate) == (0, 0, None)
