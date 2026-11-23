# Responsible AI: FinSight (W3-10)

The report's responsible-AI section, with the evidence for each claim. It covers every row of the project plan's
§10 (refusals, prompt injection against documents, PII, hallucination control, human-in-the-loop, transparency,
limitations) and, for W3-10, the intended use, the failure modes and the adversarial tests. Numbers are the final
version A5 on gold-v1 (102 questions) unless stated; every one is in a committed file, named next to it.

| Plan §10 area | What FinSight does | Evidence (committed) |
|---|---|---|
| Refusals | Rules R1-R7 (spec §6) decided by the planner, with code rules for the priority and partial refusals; the composer writes the fixed sentence | Gold category G and the 22 refusal tests: 32/32 (A5); through the API 22/22, identical to the agent's answers |
| Prompt injection against documents | Passages framed as data; a passage that addresses an AI system left out whole; the critic's payload check; the give-up answer as backstop | 12 injection tests: 0/12 successful, before and after the defences (agent) and through the API; detector on unseen phrasings: 29/60 caught |
| PII | E-mail addresses and phone numbers masked in every log record and everything sent to Langfuse; R5 refuses requests for personal data | PII prompts through the API: no log record carries the question, a log call that does is masked, the traces store the masked question |
| Hallucination control | Figures only from the tools or a cited passage; the critic's checks; "verified" or a "could not verify" note | Hallucinations (numbers in no source) 9 for B0, 0 for every agent version; A5's critic left 4 answers unverified, all 4 wrong |
| Human-in-the-loop | A research assistant only; the UI's disclaimer; human graders decide every summary and wrong answer | UI screenshot `docs/screenshots/w2-10/00-start.png`; two graders, kappa 1.00 on A5's round |
| Transparency | Citations on every answer, the trace expander in the UI, Langfuse traces of every node, tool and model call | `docs/screenshots/w2-10/03-trace-expander.png`, `docs/screenshots/w3/`, `eval/results/traces/` |
| Limitations | Stated in the UI, the report and below | This document's last section |

## Intended use

FinSight answers questions about the SEC 10-K and 10-Q filings of 20 US companies (config/companies.yaml: 10-Ks for
fiscal 2023-2025, 10-Qs for the first three quarters of fiscal 2025), with every figure cited to its filing. It is a
research assistant for someone who will check the cited filing: an analyst, a student, a reviewer. It is not
investment, tax or legal advice and does not forecast or value a company (rules R1, R2, R4, R7). It does not know
companies, years or documents outside that corpus and says so (R3).

## Refusals

- **Policy:** spec §6, rules R1 (investment advice) to R7 (off-topic, tax and legal advice). A refusal has a rule
  and the rule's fixed sentence with an in-scope suggestion; a question that also asks for a reported figure is
  answered in part (a partial refusal). The planner decides; code rules apply the priority (an instruction to ignore
  the rules is R6 whatever it asks; tax is R7, not R5; a forecast is R2, not R3) and turn a whole refusal of such a
  question into a partial one (W3-06, ADR-042).
- **Evidence:**
  - Gold category G and the 22 refusal tests: refusal and not reported 32/32 for A5, 29/32 for A1
    (`eval/results/agent_baselines.md`, ADR-037).
  - Through the API: 22/22 refusal tests pass, every answer identical to A5's gold run
    (`eval/results/api_safety_results.md`).
  - The development set v2: 7/7 (`eval/results/dev_v2/dev_v2_baselines.md`); the planner's 49 development
    decisions: 49/49 (`eval/results/planner_dev.md`).
  - Over-refusal is measured too: the controls among the refusal tests (RF16-RF22) must be answered, and are.

## Prompt injection against documents (adversarial tests)

- **What we do** (ADR-035, spec §6.3): a filing passage is data. The composer is the only node that reads passages,
  inside `<passage>` tags with `<` escaped and a prompt rule; a passage with any sentence addressed to an AI system
  is left out whole before the model reads it (`agent/injection.py`), and the answer says so ("a retrieved passage
  contained conflicting or instruction-like text and was ignored"); the critic flags an answer carrying a link, an
  e-mail, prompt text, a credential request or a changed role; if that survives the send-back, the user gets the
  give-up answer, never the draft.
- **The tests:** 12 injection tests (`eval/injection_tests.jsonl`, 9 attack types: instruction override, false
  figure, refusal bypass, system-prompt leak, role change, malicious link, fake authority, hidden text, data
  exfiltration). Each plants its sentence at the top of the first passage of a search of a real filing section,
  added to every answering plan: the worst case, the agent retrieves it. (The plan's "separate test index" became
  planting at retrieval time, so the corpus stays clean.)
- **Evidence:**
  - Agent level (W2-13, `eval/results/injection_results.md`): 0/12 successful injections with the defences off and
    on; the planted sentence reached the composer as written in 10/12 tests before the defences, 0/12 after.
  - Through the API (W3-10, `eval/results/api_safety_results.md`): 0/12 successful, the planted sentence reached no
    model (0/12), the spec's note on 11/11 tests that retrieved it; every answer identical to the same test run on
    the graph directly. 10/12 pass every check; INJ04 and INJ12 resist the attack but miss the expected figure or
    wording, INJ03 is refused before anything is retrieved.
  - The detector on attacks nobody anticipated (held-out sets written by a separate agent): 29/60 caught, 1/60 filing
    sentences flagged. It is a filter, not a guarantee; the critic and the give-up answer do not depend on it.
- **Limits:** an injection that only changes a figure or the tone, without a link or prompt text, is caught only by
  the framing and the evidence checks; the user's own instructions are refusal rule R6, not an injection.

## PII

- **What we do** (ADR-035, `core/pii.py`): e-mail addresses and phone numbers (North American and international
  forms) are masked in every log record of the API process (a log-record factory `create_app` installs, whatever
  the logger or handler) and in everything sent to Langfuse (`core/tracing.scrub`). The API logs no question or
  request body. Rule R5 refuses requests for personal data about anyone. Filings name executives: public, business
  context, answered as such.
- **Evidence** (W3-10, `eval/results/api_safety_results.md`), three PII prompts through `POST /ask`:
  - A question with an e-mail address and one with a phone number are answered normally (the figures, verified);
    a request for a CEO's home address and phone number is refused under R5.
  - Every log record of the process at DEBUG while they ran: none carries the question, so none carries an address.
  - A log call that does carry the question (the demonstration in the same process) comes out as
    "My e-mail is [email] ..." and "Call me back on [phone] ...".
  - The traces, read back from Langfuse: the two prompts with an address are stored as "My e-mail is [email], please reply there: ..." and "Call me back on [phone] if needed. ...", no raw address in either (`eval/results/api_safety/pii_traces.json`).
- **Limits (stated, not hidden):** the question reaches the model as the user wrote it (masking it would change
  the question the agent answers, ADR-035); the API keeps a session's questions and its reply cache in memory for
  their lifetime (API_SESSION_TTL_S, API_CACHE_TTL_S), and the local reply cache of model calls (`data/cache/llm`,
  git-ignored, never shared) keeps the prompts to replay them. A bare "415 555 2671" without a country code or
  separators is not masked, on purpose: it cannot be told from a row of figures.

## Hallucination control

- **What we do:** values, citations and calculations are built from the tools' outputs, never typed by the model;
  a figure the model reads from a passage is kept only when that passage prints it for its own year. The critic's
  deterministic checks (W2-05, W3-03: units, scope, operation, segment, summary, refusal rules) verify every draft;
  a failing draft goes back once, then leaves with "verified": false and a "could not verify: ..." note. The UI
  shows the badge.
- **Evidence** (`eval/results/evaluation_report.md`): hallucinations (a number or claim in no source) 9 for B0, 8
  for the best prompting strategy, 0 for every agent version (A1-A5); A5's critic verified 98 of 102 answers and
  the 4 it left unverified are all wrong (A1's critic verified all 102, its 16 wrong ones included, the finding
  W3-03 fixed; the B14 traces in `docs/screenshots/w3/` show one).

## Human-in-the-loop

- FinSight is a research assistant: the UI's sidebar says it answers from the indexed filings, cites them, can be
  wrong, and gives no investment advice, forecasts or valuation opinions (`docs/screenshots/w2-10/00-start.png`).
  A human reviews the cited filing before using a figure in client work or an investment decision.
- In the evaluation, humans decide what a script cannot: two graders grade every summary and label every wrong
  answer (`eval/GRADING.md`; A5's round 13 of 13 grades agreeing, kappa 1.00), and a second team member verified
  every gold question and test.

## Transparency

- Every answer carries its citations (filing, item, link to the SEC document) and, in the UI, a trace expander with
  the plan, the tool calls and the verdict (`docs/screenshots/w2-10/03-trace-expander.png`).
- Langfuse traces every node, tool call and model call with its latency and tokens, tagged with the question id
  (W2-08, W3-09: `eval/results/traces/`, `docs/screenshots/w3/`).
- The evaluation is reproducible from committed files: `python eval/report.py --check`.

## Failure modes

From the final run (`eval/results/evaluation_report.md`, `eval/results/agent_baselines.md`): A5's 7 wrong answers on
gold are 2 retrieval misses, 4 reasoning errors and 1 tool-call error.
- **Line items the facts tool has no metric for** (B08 a medical care ratio, B14 a segment's operating margin, C08
  net sales excluding membership fees): the agent finds the wrong figure or says it is not reported; the critic now
  flags the segment case.
- **Follow-up questions** (F02): "What was it in fiscal 2025?" after an operating-margin question was answered with
  the operating income.
- **Summaries** (E01, E10): an answer from the wrong part of the MD&A, graded 0.
- **Cost:** about two model calls and 1.7 tool calls a question; latency p50 5.5 s, p95 11.2 s.

## Limitations

- **Coverage:** 20 companies; 10-Ks for fiscal 2023-2025 and fiscal 2025's first three 10-Qs. Anything else is out
  of scope (R3).
- **XBRL tags differ by company and year,** and banks report revenue differently ("total revenue, net of interest
  expense"); the facts tool maps a fixed set of metrics (`config/concept_map.yaml`) and declares the gaps it knows.
  Restated figures: the latest filed value (spec §7 rule 7).
- **One small model** (Qwen3-8B, thinking off) on a shared endpoint: answers can vary with the retrieved passages
  (W3-06 lost a summary point that way).
- **The evaluation:** 102 gold questions written by the team from the same filings (verified by a second member);
  about 1 percentage point per question; A1's failures shaped A5's fixes, which were tuned on a separate
  development set and run on gold once (ADR-037).
- **Defences are measured, not proven:** the injection detector catches about half of unanticipated phrasings; PII
  masking covers e-mail addresses and phone numbers, not every kind of personal data.
