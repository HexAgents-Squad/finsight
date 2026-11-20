"""FinSight's chat UI (W2-10, ADR-033).

    streamlit run ui/app.py          (the API must be running: uvicorn api.main:app; FINSIGHT_API_URL if not local)

A chat over the FinSight API only (ui/client.py; no model access): each answer is a card with the verdict badge, the
answer, the calculation, the figures, the citations linked to the SEC filings, any notes, and a "Show trace"
expander with the run behind it. The sidebar has the research-assistant disclaimer, the API's status, the companies
and periods FinSight covers with example questions, and "New conversation". The conversation's session id (W2-09)
is kept in the Streamlit session and replaced by each response's; follow-ups ("And in fiscal 2024?") rely on it.

A question is asked in two runs, so that no click can lose its answer: the run that receives it stores it as the
conversation's `pending` question and reruns with the question box and the example questions disabled; the next run
asks the API and records the reply and the session id on the conversation object (ui/client.py Conversation) it
fetched before the request, then reruns to show it. Streamlit stops a run at its next st.* call or st.session_state
access once the user interacts, so nothing between the API's answer and the recording touches Streamlit. Anything the
call raises is recorded as an error reply, so the question never stays pending; "New conversation" stays enabled as
the way out.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # `streamlit run ui/app.py` puts ui/ on the path, not the repo
    sys.path.insert(0, str(ROOT))

from ui import client as api  # noqa: E402
from ui import render  # noqa: E402

MAX_QUESTION_CHARS = 1000                          # the API's own limit (api/main.py AskRequest)
HEALTH_TTL_S = 15.0                                # a ready status is re-read at most this often, not on every rerun
HEALTH_RETRY_S = 5.0                               # any other status sooner (each read can wait up to 3 s)

st.set_page_config(page_title="FinSight", page_icon=":material/description:", layout="centered")


@st.cache_resource
def shared_client() -> api.ApiClient:
    """One HTTP client (and connection pool) for every browser session of this server."""
    return api.ApiClient()


def state() -> Any:
    """The Streamlit session: the API client, the conversation (ui/client.py Conversation) and the API's status."""
    s = st.session_state
    if "client" not in s:                          # tests put a fake client here first
        s.client = shared_client()
    if "chat" not in s:
        s.chat = api.Conversation()
    s.setdefault("health", None)
    s.setdefault("health_at", None)
    return s


def health(s: Any) -> dict[str, Any] | None:
    now = time.monotonic()
    ttl = HEALTH_TTL_S if (s.health or {}).get("status") == "ok" else HEALTH_RETRY_S
    if s.health_at is None or now - s.health_at > ttl:
        s.health, s.health_at = s.client.health(), now
    return s.health


def answer_card(body: dict[str, Any], session_id: str | None) -> None:
    answer = body["answer"]
    label, color, icon = render.badge(answer)
    with st.container(horizontal=True):
        st.badge(label, icon=icon, color=color)
        if body.get("cached"):
            st.badge("From cache", icon=":material/bolt:", color="gray")
    st.markdown(render.escape(answer["answer"]))
    for line in render.calculation_lines(answer):
        st.markdown(line)
    rows = render.figure_rows(answer)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    links = render.citation_links(answer)
    if links:
        st.markdown("**Sources**\n" + "\n".join(f"- {link}" for link in links))
    for note in render.as_list(answer.get("notes")):
        st.caption(render.escape(note))
    with st.expander("Show trace"):
        run = render.trace(body)
        cols = st.columns(4)
        cols[0].metric("Verdict", run["verdict"] or "-")
        cols[1].metric("Tool calls", run["steps"])
        cols[2].metric("Tokens", f"{run['tokens']:,}" if isinstance(run["tokens"], int) else str(run["tokens"]))
        seconds = run["request_seconds"] if run["cached"] else run["run_seconds"]
        cols[3].metric("Seconds", seconds if seconds is not None else "-",
                       help="The cached reply's request" if run["cached"] else "The agent's run, without the queue")
        st.caption(render.escape(f"Nodes: {run['nodes'] or '-'}"
                                 + (f" · stopped: {run['stop_reason']}" if run["stop_reason"] else "")))
        if run["cached"]:
            st.caption("Answered from the reply cache: the run below produced it earlier, possibly in another "
                       "conversation, so this conversation has no trace of its own for it.")
        if run["history"]:
            st.caption("Earlier questions used: " + render.escape(" / ".join(run["history"])))
        if run["tool_calls"]:
            st.dataframe(run["tool_calls"], hide_index=True, width="stretch")
        url = None if run["cached"] else render.session_trace_url(session_id)
        if url:
            st.link_button("Open the conversation's traces in Langfuse", url, icon=":material/open_in_new:")


def show(message: dict[str, Any]) -> None:
    with st.chat_message(message.get("role", "assistant")):
        try:
            if message.get("role") == "user":
                st.markdown(render.escape(message["text"]))
            elif message.get("error"):
                st.error(message["error"], icon=":material/error:")
            else:
                answer_card(message["body"], message.get("session_id"))
        except Exception:                          # one malformed reply must not break the whole conversation
            st.error("This answer could not be shown. Ask again, or start a new conversation.",
                     icon=":material/error:")


def sidebar(s: Any, busy: bool) -> str | None:
    """The sidebar; returns the example question clicked, if any."""
    clicked = None
    with st.sidebar:
        st.header("FinSight")
        hints = render.scope_hints()
        st.info(render.disclaimer(len(hints["companies"])), icon=":material/info:")
        status = health(s)
        if status is None:
            st.error(f"API not reachable at {s.client.base_url}", icon=":material/cloud_off:")
        elif status.get("status") == "ok":
            st.success(f"API ready · {status.get('model') or 'model'}", icon=":material/check_circle:")
        elif status.get("status") == "starting":
            st.warning("API starting: the model is warming up", icon=":material/hourglass_top:")
        else:
            st.warning(f"API degraded: warm-up failed ({status.get('warm_up') or 'unknown'}); answers may fail",
                       icon=":material/warning:")
        if st.button("New conversation", icon=":material/add_comment:", width="stretch"):   # always: the way out
            s.chat.clear()
            st.rerun()                             # this run decided `busy` before the click: start clean
        st.subheader("What you can ask about")
        st.caption(f"10-K fiscal years {', '.join(map(str, hints['tenk_years']))}; 10-Q "
                   f"{', '.join(hints['tenq_quarters'])}. Fiscal years are each company's own.")
        st.caption(" · ".join(f"{name} ({ticker})" for name, ticker in hints["companies"]))
        st.subheader("Try")
        for i, question in enumerate(render.EXAMPLE_QUESTIONS):
            if st.button(question, key=f"example-{i}", width="stretch", disabled=busy):
                clicked = question
    return clicked


def main() -> None:
    s = state()
    chat, client = s.chat, s.client                       # plain objects from here on (see the module docstring)
    busy = chat.pending is not None
    clicked = sidebar(s, busy)
    st.title("FinSight")
    st.caption("Questions about US companies' SEC 10-K and 10-Q filings, answered with citations.")
    for message in chat.messages:
        show(message)
    typed = st.chat_input("Ask about a company's filings, e.g. \"What was Amazon's net income in fiscal 2023?\"",
                          max_chars=MAX_QUESTION_CHARS, disabled=busy)
    question = None if busy else (typed or clicked)
    if question:
        chat.ask(question)
        st.rerun()                                        # show the question, with the inputs disabled, then ask
    if busy:
        with st.chat_message("assistant"), st.spinner("Checking the filings..."):
            try:
                reply = client.ask(chat.pending, chat.session_id)
            except Exception:                             # never leave the question pending with the inputs off
                reply = api.Reply(ok=False, error="Something went wrong asking FinSight. Try again.")
            chat.record(reply)
        st.rerun()


main()
