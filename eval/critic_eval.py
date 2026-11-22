"""W3-03: the critic measured on A1's recorded drafts: how many wrong answers it flags (catch rate) and how many right
ones (false flags), with no new model request.

    EMBED_BACKEND=sentence-transformers EMBED_LOCAL_MODEL_NAME=BAAI/bge-small-en-v1.5 \\
        python eval/critic_eval.py --questions dev2        # -> eval/results/critic/critic_dev2.md
    python eval/critic_eval.py --questions gold             # the regression check on gold (run once, not tuned on)

Each question goes through A1's graph again (eval/run_agent.py's real nodes: the same planner and composer prompts, the
tools on data/) up to the composer's first draft. The planner's and composer's model calls are answered by the reply
cache, so the drafts are A1's own, at no request: the LLM budget is 1 and a cache miss stops the run with nothing
written (a changed prompt or tool output would show as one). Search queries embed through EMBED_BACKEND: set the local
copy of the embedding model (as above) and no request leaves the machine; it gives the same vectors as the endpoint
(ADR-038). The critic of this checkout then judges that first draft, and the run stops there: no send-back, so nothing
is re-composed. Each verdict is joined with the run's graded score (eval/results/.../scores/<run>.jsonl, the graders'
final grades included): a flag on a wrong answer is a catch, a flag on a right one a false flag. An answer the scorer
leaves undecided is reported apart. The replayed draft is compared with the run's recorded answer where A1's critic
passed the first draft (then the two must be the same text), as a check that the replay is A1's run.

Output: eval/results/critic/critic_<questions>.md (summary, every flag with its check and message, the misses).
Exit codes: 0 = written, 2 = setup, a cache miss, or a replay that is not A1's.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_agent import QUESTION_SETS, RUN_ID, Item, load_items  # noqa: E402
from eval.run_retrieval import RESULTS_DIR  # noqa: E402

OUT_DIR = RESULTS_DIR / "critic"
SCORE_DIRS = {"gold": RESULTS_DIR / "scores", "dev2": RESULTS_DIR / "dev_v2" / "scores"}


class FirstDraftJudged(Exception):
    """Raised by the recording critic once it has judged the first draft: the replay stops there."""


@dataclass
class Judged:
    """One question's first draft and the critic's verdict on it (None: the run produced no draft to judge)."""
    id: str
    answer: str | None = None
    refused: bool = False
    violations: list[tuple[str, str]] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.violations)


def recording_critic(record: dict[str, Judged], qid: list[str], critique: Any) -> Any:
    """A critic node that judges the first draft with `critique`, records it under the current question id, and
    stops the run."""
    async def critic(state: dict[str, Any]) -> dict[str, Any]:
        verdict = critique(state)
        draft = state.get("draft")
        record[qid[0]] = Judged(qid[0], draft.answer if draft else None, bool(draft and draft.refused),
                                list(verdict.violations))
        raise FirstDraftJudged

    return critic


async def replay(items: list[Item], nodes: Any, limits: Any) -> dict[str, Judged]:
    """Every item to its first draft through the graph with `nodes` (whose critic records and stops)."""
    from agent.graph import build_graph, run
    record: dict[str, Judged] = {}
    current = [""]
    nodes = nodes(record, current)
    graph = build_graph(nodes)
    for item in items:
        current[0] = item.id
        try:
            await run(item.question, list(item.history), graph=graph, limits=limits)
        except FirstDraftJudged:
            continue
        record.setdefault(item.id, Judged(item.id))          # no draft reached the critic (a give-up answer)
    return record


def real_recording_nodes(llm: Any) -> Any:
    """A1's real nodes on `llm`, the critic replaced by the recording one (this checkout's critique)."""
    from dataclasses import replace

    from agent.critic import critique
    from eval.run_agent import ablation_nodes
    base = ablation_nodes(llm, None)
    return lambda record, current: replace(base, critic=recording_critic(record, current, critique))


def load_jsonl(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x)}


def same_as_recorded(judged: dict[str, Judged], runs: dict[str, dict]) -> list[str]:
    """Questions whose replayed first draft differs from the answer A1 recorded although its critic passed that
    draft (the only case where the two must be the same text)."""
    out = []
    for qid, j in judged.items():
        rec = runs[qid]
        agent = rec.get("agent") or {}
        first_passed = agent.get("visits", {}).get("critic", 0) == 1 and agent.get("verified")
        if first_passed and j.answer is not None and j.answer != json.loads(rec["raw_reply"])["answer"]:
            out.append(qid)
    return out


def summary(judged: dict[str, Judged], scores: dict[str, dict]) -> dict[str, Any]:
    wrong = [q for q, s in scores.items() if s["correct"] is False]
    right = [q for q, s in scores.items() if s["correct"] is True]
    undecided = [q for q, s in scores.items() if s["correct"] is None]
    flagged = {q for q, j in judged.items() if j.flagged}
    return {"questions": len(scores), "wrong": len(wrong), "caught": sorted(q for q in wrong if q in flagged),
            "missed": sorted(q for q in wrong if q not in flagged), "right": len(right),
            "false_flags": sorted(q for q in right if q in flagged), "undecided": undecided,
            "undecided_flagged": sorted(q for q in undecided if q in flagged)}


def rate(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "–"


def report(name: str, judged: dict[str, Judged], scores: dict[str, dict], meta: dict[str, Any]) -> str:
    s = summary(judged, scores)
    rules = Counter(rule for q in s["caught"] for rule in dict.fromkeys(r for r, _ in judged[q].violations))
    out = [f"# The critic on A1's first drafts: {name} (W3-03)", "",
           "Generated by `eval/critic_eval.py`: A1's run replayed from the reply cache to each first draft, judged by "
           "this checkout's critic, joined with the graded scores. A catch is a flag on a wrong answer; a false flag "
           "is a flag on a right one.", "",
           f"- Questions: {s['questions']} (run {meta['run_id']}, git {meta['git']}, live requests {meta['live']})",
           f"- Catch rate (wrong answers flagged): {rate(len(s['caught']), s['wrong'])}",
           f"- False flags (right answers flagged): {rate(len(s['false_flags']), s['right'])}"
           + (f": {', '.join(s['false_flags'])}" if s["false_flags"] else ""),
           f"- Undecided by the scorer: {len(s['undecided'])}" + (f", flagged {', '.join(s['undecided_flagged'])}"
                                                                   if s["undecided_flagged"] else ""),
           "- Checks behind the catches (a draft can fail several): "
           + (", ".join(f"{r} {n}" for r, n in rules.most_common()) or "none"), "",
           "## Flags", "", "| Question | Answer was | Check | Message |", "|---|---|---|---|"]
    verdict = {True: "right", False: "wrong", None: "undecided"}
    for qid in sorted(judged):
        for rule, message in judged[qid].violations:
            out.append(f"| {qid} | {verdict[scores[qid]['correct']]} | {rule} | {message.replace('|', '/')[:220]} |")
    out += ["", "## Wrong answers not flagged", ""]
    out += [f"- {q}: {(judged[q].answer or '(no draft)')[:200]}" for q in s["missed"]] or ["- none"]
    return "\n".join(out) + "\n"


async def run_live(items: list[Item], settings: Any, limits: Any) -> tuple[dict[str, Judged], Any]:
    from core.llm import LLMClient
    from eval.run_demo import BudgetedLLM
    async with LLMClient(settings) as client:
        judged = await replay(items, real_recording_nodes(BudgetedLLM(client, 1)), limits)
        return judged, client.usage


def main(argv: list[str] | None = None, *, nodes: Any = None, settings: Any = None, limits: Any = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions", choices=sorted(SCORE_DIRS), default="dev2")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--pack-dir", type=Path, default=RESULTS_DIR)
    args = ap.parse_args(argv)
    packs, runs_dir = QUESTION_SETS[args.questions]
    try:
        items = load_items([args.pack_dir / p for p in packs])
        runs = load_jsonl(runs_dir / f"{RUN_ID}.jsonl")
        scores = load_jsonl(SCORE_DIRS[args.questions] / f"{RUN_ID}.jsonl")
        from agent.graph import Limits
        if settings is None:
            from core.settings import get_settings
            settings = get_settings()
    except (ValueError, OSError) as err:
        print(f"setup incomplete: {str(err).splitlines()[0]}", file=sys.stderr)
        return 2
    limits = limits or Limits.from_settings(settings)
    from eval.run_demo import BudgetExceeded
    try:
        if nodes is None:
            judged, usage = asyncio.run(run_live(items, settings, limits))
            live = usage.calls
        else:                                              # given nodes (a test): no client
            judged, live = asyncio.run(replay(items, nodes, limits)), 0
    except BudgetExceeded as err:
        print(f"a model call was not in the reply cache ({err}); not A1's replay, nothing written", file=sys.stderr)
        return 2
    if live:
        print(f"{live} live request(s): not A1's replay, nothing written", file=sys.stderr)
        return 2
    differ = same_as_recorded(judged, runs)
    if differ:
        print(f"replayed drafts differ from A1's recorded answers: {', '.join(differ)}; nothing written",
              file=sys.stderr)
        return 2
    from eval.run_agent import git_commit
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"critic_{args.questions}.md"
    text = report(args.questions, judged, scores, {"run_id": RUN_ID, "git": (git_commit() or "?")[:7], "live": live})
    path.write_text(text, encoding="utf-8", newline="\n")
    s = summary(judged, scores)
    print(f"wrote {path}: caught {len(s['caught'])}/{s['wrong']}, false flags {len(s['false_flags'])}/{s['right']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
