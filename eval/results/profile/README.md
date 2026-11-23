# Profiles (W3-07, ADR-043)

Written by `eval/profile_agent.py`. A profile replays a run's questions through the real graph from the reply cache
(no request: a cache miss raises), after the API's warm-up, and is written only if every answer is the run's own.
Each LLM call counts the endpoint seconds its request took when it was sent; everything else is the replay's wall
time on the machine that ran it (a search embeds with the local embedder there, the hosted embedder in a live run).

- `A5_qwen3-8b_agent_think-off_dev-w306-before.md`: W3-06's run (development set v2) on W3-06's code.
- `A5_qwen3-8b_agent_think-off_dev-w306-after.md`: the same run's questions on W3-07's code; every answer is word
  for word W3-06's, so the scores are too.
- `A1_qwen3-8b_agent_think-off_a-v1.md` and `-dev2.md`: A1's runs files only (gold and the development set v2);
  they no longer replay, because the planner prompt changed in W3-04 and W3-06.

## Where the time goes

On W3-06's run, warmed, every call live (`before`):

- The planner's one LLM call is 55% of the time (3.1 s and 2,197 tokens a question), the composer's 38% (2.2 s,
  1,580 tokens).
- search_filings is 6% (0.3 to 0.5 s a search, nearly all of it Chroma's filtered query).
- Everything else (the planner's rules, the critic, the facts) is under 1%. The critic makes no LLM call.

## Before and after (the same 30 development questions)

- Estimated latency, every call live: p50 5.47 s before, 5.39 s after.
- Tokens per question: mean 3,776 before, 3,684 after (p50 3,272 both).
- LLM calls: 59 before, 58 after. dev-E24's re-ask is skipped: 7,367 tokens become 4,582, and 17.9 s become 10.4 s.
- The planner's local time: 85 ms a question before, 3 ms after (the config files were parsed five times a question).
- A server's first question: after the warm-up nothing more is loaded (the profiler's first question took 0.14 s
  of local time).

For comparison, A1 on the same questions took 6.36 s and 4,022 tokens (recorded, every call live). A1 on gold took
4.97 s and 3,000 tokens: the development set v2 asks for more searches and summaries than gold.

## The next levers (not done in W3-07)

- The planner prompt is the largest token cost (2,064 prompt tokens a question). Shortening it is a prompt change:
  it needs a live re-measure and puts the 49/49 refusal decisions at risk.
- Dense retrieval: an exact search over the filtered chunks instead of Chroma's filtered query (~0.5 s a search).
  It can change rankings, and so answers.
