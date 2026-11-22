# Mentor meetings

Calls with the mentor, Dr. Gunjan Balde: what was shared, the feedback, what the team changed because of it,
and the actions. A decision that changes the design also goes into `docs/decisions.md` as an ADR. Only what is
on record is written here; anything not recorded says so.

## 2026-11-07 · Week 0 review

- **When:** Sat 7 Nov 2026 (the date of the mentor's feedback, `docs/week1_plan.txt`). The team's Week 0
  submission notes (not in the repo) planned it as a 45-minute call, 11:00–11:45 IST, booked through the course
  portal.
- **Who:** Dr. Gunjan Balde and HexAgents_Squad; the attendees were not recorded.
- **Prepared for the call** (submission notes): the Week 0 Results Note and the Week 0 Detailed Results for the
  portal's Document Library, and on screen `eval/results/retrieval_summary.md`, `docs/decisions.md` and the PR
  list.
- **Week 0 exit gate** (in the results note):
  - the gold set frozen: 80 questions, tag `gold-v1`, plus 22 refusal and 12 prompt-injection tests;
  - retrieval measured: the best setting, C3 hybrid with company and year filters, had recall@5 0.747 and
    recall@10 0.904 at the time (0.774 and 0.870 since the W1-04 re-index, ADR-023).

### Questions prepared for the mentor
The submission notes list these for the session request:
1. Scope: keep the 10-Qs at fiscal 2025 Q1–Q3, or add the fiscal-2026 quarters?
2. The official endpoint from 8 Nov: JSON and tool-calling support, the token quota, and the embedding model (a
   different embedding model means re-indexing).
3. Can the same endpoint model be the LLM-as-judge for answer quality?
4. Is an AI-drafted gold set acceptable when every item is verified by a team member?
5. Our Week 1 plan: baselines, a reranker, answer-level evaluation.

### Feedback
- **The mentor asked for these by Wed 11 Nov:**
  - accuracy numbers;
  - a model comparison: which model gives which type of answer, and how accurately;
  - accuracy split into short ("one-word") answers and summary answers;
  - where the models are less accurate, fail or hallucinate.
- **Answers to questions 1–5:** not recorded.

### What changed because of it
- **Week 1 was replanned on 10 Nov** into two sprints (`docs/week1_plan.txt`):
  - Sprint A: the model comparison, due Wed 11 Nov 15:00;
  - Sprint B: the rest of Week 1.
- **The model comparison was built and delivered as W1-12 to W1-15** (ADR-019, `eval/results/model_comparison.md`):
  - Qwen3-8B with retrieved passages, Llama-3.2-3B with the same passages, and Qwen3-8B closed-book;
  - accuracy reported for short and summary answers;
  - refusals, and failure and hallucination examples.
- **Question 2 was settled by the course's endpoints:**
  - chat: Qwen3-8B (ADR-016);
  - embeddings: bge-small-en-v1.5, with the index rebuilt in W1-04 (ADR-021, ADR-023).
- **Question 3:** the summary answers were graded by two people. The Qwen3-8B judge is a second opinion only, and it was more lenient than the people in every disagreement (ADR-019).
- **Question 1:** no answer recorded; the corpus scope is unchanged (`docs/spec.md` §3), and spec §12 still lists
  the question as open.

### Actions
Carried over from Week 0 (Jira UPGRAD-41), into W1-10 unless noted:

| Action | Owner | Where |
|---|---|---|
| Second-person verification of the 22 refusal and 12 injection tests (`validate_test_sets.py --final`) | Amresh | `eval/verify_test_sets.py` |
| Independent WP2 spot check of 10 sections against the SEC filings | Chandra Shekhar (checked by Alok, the WP2 backup; Chandra's review pending) | `docs/wp2_spot_check.md` |
| Smoke test against the endpoint on every laptop | everyone | `docs/smoke_tests.md` |
| Page furniture removed from the section text, full re-index | Chandra Shekhar, Alok | done in W1-04 (ADR-023) |

All carried actions are closed (22 Nov 2026): the test sets were verified in W1-10; the smoke test passed on all five
laptops and Chandra Shekhar's independent WP2 check was signed in W2-12 (PR #45; `docs/smoke_tests.md`,
`docs/wp2_spot_check.md`).

### Next session
Not recorded. The Week 1 exit gate to show is in `docs/week1_plan.txt`: the model comparison, the prompting
baseline table, and the retrieval table.
