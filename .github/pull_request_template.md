## Jira
UPGRAD-<key> · <!-- one-line link or key of the ticket this PR implements -->

## What & why
<!-- 2–4 sentences: what changed and why it is needed -->

## How to test
<!-- exact commands, e.g. `pytest -q`, `python eval/validate_gold.py eval/gold_set.jsonl` -->

## Evidence
<!-- output, numbers (state the gold-set version if relevant), screenshots or sample records -->

## Checklist
- [ ] Branch is `wpN/UPGRAD-<key>-<short-desc>` and every commit message starts with the Jira key
- [ ] CI checks are green (ruff + pytest + data validation)
- [ ] New/changed behaviour has tests; tests run offline in seconds
- [ ] Shared data contracts unchanged, or updated in `docs/spec.md` / `docs/week0_plan.txt` / `CLAUDE.md` and announced to the team
- [ ] Design decisions recorded in `docs/decisions.md`
- [ ] No secrets, `.env` or downloaded data committed
- [ ] Gold answers NOT edited to improve scores
- [ ] Reviewed by the WP backup before merge
