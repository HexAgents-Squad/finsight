# eval/: evaluation tooling (WP8)

| File | What it does |
|---|---|
| `schema.py` | The gold-item format and every validation rule (single source of truth for week0_plan §2.4/§3 and spec §5–§7) |
| `validate_gold.py` | Checks `gold_set.jsonl`; `--final` adds the freeze rules |
| `sheet_to_jsonl.py` | Converts the team's Google Sheet (CSV export) to `gold_set.jsonl`; writes only if every row is valid |
| `gold_sheet_template.csv` | Header row to import into Google Sheets (File → Import) |
| `retrieval_eval.py` | recall@1/5/10 and MRR of a retrieval run against the gold set (contract with WP5 in the file header) |

## Writing gold questions (everyone)
1. Use the shared Google Sheet (columns = `gold_sheet_template.csv`). One row per question.
2. Cell formats:

   | Column | Example | Notes |
   |---|---|---|
   | `id` | `B07` | Category letter + 2 digits |
   | `category` | `ratio` (or `B`) | `fact`, `ratio`, `period_comparison`, `multi_company`, `qualitative`, `multi_turn`, `refuse` |
   | `companies` | `MSFT, GOOGL` | Tickers from `config/companies.yaml` |
   | `fiscal_years` | `2024` | The company's own fiscal year (spec §4) |
   | `gold_value` | `44.6` | Numbers only; `$`, `%` and commas are ignored. Required for B and C |
   | `unit` | `%` | One of: `USD`, `USD thousands`, `USD millions`, `USD billions`, `USD per share`, `shares`, `shares millions`, `%`, `ratio`, `count` |
   | `gold_sources` | `MSFT \| 10-K \| 2024 \| Item 8` | One per line or separated by `;`. A 10-Q needs the quarter: `AAPL \| 10-Q \| 2025 \| Q2 \| Part I Item 2` |
   | `must_refuse` | `TRUE` | Only for G refusals |
   | `refusal_rule` | `R1` | R1–R7 for refusals and partial refusals; empty otherwise |
   | `previous_turn_id` | `A03` | Only for F follow-ups |
   | `written_by` / `verified_by` | `Anirudh` / `Amresh` | Must be different people |

3. G questions are either **refusals** (`must_refuse` TRUE + `refusal_rule`, no sources) or **"not reported" traps** (`must_refuse` FALSE, `gold_answer` starting with "not reported", no `gold_value`, and the source that was checked, e.g. `JPM | 10-K | 2024 | Item 8`).

## Converting and validating
```bash
pip install -r requirements-dev.txt
python eval/sheet_to_jsonl.py gold_sheet.csv --dry-run     # check the sheet; errors name the sheet row
python eval/sheet_to_jsonl.py gold_sheet.csv               # writes eval/gold_set.jsonl (only if valid)
python eval/validate_gold.py                               # draft rules (CI runs this on every PR)
python eval/validate_gold.py --final                       # before tagging gold-v1
```
`--final` also requires: every item verified, the exact category split (15/15/12/10/12/6/10), at least one refusal per rule R1–R7, and 2–3 "not reported" traps.

## Scoring retrieval (with WP5)
```bash
python eval/retrieval_eval.py --runs eval/results/runs_c2_hybrid_filtered.jsonl --out eval/results/retrieval_summary.md
```
Build queries with `retrieval_eval.build_query()` so follow-up questions include the previous turn, exactly as the agent will.
