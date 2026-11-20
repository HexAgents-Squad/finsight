"""FinSight's chat UI (W2-10, ADR-033).

    streamlit run ui/app.py          (the API must be running: uvicorn api.main:app; FINSIGHT_API_URL if not local)

A chat over the FinSight API only (ui/client.py; no model access): each answer is a card with the verdict badge, the
answer, the calculation, the figures, the citations linked to the SEC filings, any notes, and a "Show trace"
expander with the run behind it. The sidebar has the research-assistant disclaimer, the API's status, the companies
and periods FinSight covers with example questions, and "New conversation". The conversation's session id (W2-09)
is kept in the Streamlit session and replaced by each response's; follow-ups ("And in fiscal 2024?") rely on it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # `streamlit run ui/app.py` puts ui/ on the path, not the repo
    sys.path.insert(0, str(ROOT))

from ui import client as api  # noqa: E402
from ui import render  # noqa: E402

st.set_page_config(page_title="FinSight", page_icon=":material/description:", layout="centered")


def state() -> Any:
    """The Streamlit session: the API client, the conversation, its session id and a question waiting to be sent."""
    s = st.session_state
    if "client" not in s:
        s.client = api.ApiClient()
    s.setdefault("messages", [])
    s.setdefault("session_id", None)
    s.setdefault("pending", None)
    return s


def answer_card(body: dict[str, Any], session_id: str | None) -> None:
    answer = body["answer"]
    label, color, icon = render.badge(answer)
    badges = st.container(horizontal=True)
    with badges:
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
    for note in answer.get("notes") or []:
        st.caption(render.escape(note))
    with st.expander("Show trace"):
        run = render.trace(body)
        cols = st.columns(4)
        cols[0].metric("Verdict", run["verdict"] or "-")
        cols[1].metric("Tool calls", run["steps"])
        cols[2].metric("Tokens", f"{run['tokens']:,}")
        cols[3].metric("Seconds", run["request_seconds"] if run["cached"] else run["run_seconds"])
        st.caption(f"Nodes: {run['nodes'] or '-'}" + (f" · stopped: {run['stop_reason']}" if run["stop_reason"] else "")
                   + (" · answered from the reply cache (the run below produced it)" if run["cached"] else ""))
        if run["history"]:
            st.caption("Earlier questions used: " + render.escape(" / ".join(run["history"])))
        if run["tool_calls"]:
            st.dataframe(run["tool_calls"], hide_index=True, width="stretch")
        url = render.session_trace_url(session_id)
        if url:
            st.link_button("Open the trace in Langfuse", url, icon=":material/open_in_new:")


def show(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(render.escape(message["text"]))
        elif message.get("error"):
            st.error(message["error"], icon=":material/error:")
        else:
            answer_card(message["body"], message.get("session_id"))


def sidebar(s: Any) -> None:
    with st.sidebar:
        st.header("FinSight")
        st.info(render.DISCLAIMER, icon=":material/info:")
        health = s.client.health()
        if health is None:
            st.error(f"API not reachable at {s.client.base_url}", icon=":material/cloud_off:")
        elif health.get("status") == "ok":
            st.success(f"API ready · {health.get('model') or 'model'}", icon=":material/check_circle:")
        else:
            st.warning(f"API {health.get('status')}: {health.get('warm_up')}", icon=":material/hourglass_top:")
        if st.button("New conversation", icon=":material/add_comment:", width="stretch"):
            s.messages, s.session_id = [], None
        hints = render.scope_hints()
        st.subheader("What you can ask about")
        st.caption(f"10-K fiscal years {', '.join(map(str, hints['tenk_years']))}; 10-Q "
                   f"{', '.join(hints['tenq_quarters'])}. Fiscal years are each company's own.")
        st.caption(" · ".join(f"{name} ({ticker})" for name, ticker in hints["companies"]))
        st.subheader("Try")
        for i, question in enumerate(render.EXAMPLE_QUESTIONS):
            if st.button(question, key=f"example-{i}", width="stretch"):
                s.pending = question


def main() -> None:
    s = state()
    sidebar(s)
    st.title("FinSight")
    st.caption("Questions about US companies' SEC 10-K and 10-Q filings, answered with citations.")
    for message in s.messages:
        show(message)
    typed = st.chat_input("Ask about a company's filings, e.g. \"What was Amazon's net income in fiscal 2023?\"")
    question = typed or s.pending
    s.pending = None
    if not question:
        return
    s.messages.append({"role": "user", "text": question})
    show(s.messages[-1])
    with st.spinner("Checking the filings..."):
        reply = s.client.ask(question, s.session_id)
    if reply.session_id:
        s.session_id = reply.session_id                   # always the latest (W2-09)
    message = ({"role": "assistant", "body": reply.body, "session_id": s.session_id} if reply.ok
               else {"role": "assistant", "error": reply.error})
    s.messages.append(message)
    show(message)


main()
