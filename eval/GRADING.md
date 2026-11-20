# Human grading of the model comparison (W1-14)

**Ticket:** UPGRAD-48 · W1-14 Human grading and failure labelling.
**Graders:** Chandra Shekhar (grader 1) and Amresh (grader 2).
**Coordinator:** Alok runs the tools.
**When:** Wed 11 Nov, 09:00–12:00. The sheet must be back by 12:00, because the mentor report (W1-15) goes out at 15:00.

## What you're grading, and why
Three setups answered the same 102 questions:
- Qwen3-8B with the filing passages;
- Llama-3.2-3B with the same passages;
- Qwen3-8B with no passages.

`eval/score_answers.py` marked most answers automatically. Two kinds of rows need a person:

1. **"grade" rows**, which you **both** grade **independently**, 0/1/2:
   - the summary answers ("What drove X's revenue growth?" can't be marked by matching a number);
   - any "which company had more" answer the scorer couldn't decide.
2. **"label" rows**, which are split between you:
   - every wrong answer gets one failure label;
   - the scorer has already suggested one, and you confirm or correct it.

Your grades are the evidence the mentor asked for: accuracy on short vs summary answers, and where and how models fail. An LLM judge also graded the summaries. Your grades decide, and the report shows how often the judge agreed with you.

**The sheet is blind:** no column says which model wrote an answer. Please **don't open `eval/results/grading/grading_key.csv`** or the score files, and don't discuss individual rows with each other until step 3.

## What you receive (from Alok)
- **`grading_sheet_<your name>.csv`:** your own copy of the sheet. Fill only your copy, and don't share it with the other grader.
- **`assignments.txt`:** your grade column and the exact spreadsheet rows of your label rows.
- **`handout.md`:** for every question on a grade row, the gold answer and the filing passages the models with passages were given. Passages marked ★ come from the section the gold answer was written from. Use them to check claims an answer adds beyond the gold answer. The handout names no model.

| Column | What it is | You |
|---|---|---|
| key | row id (starts with `k`) | **don't touch** |
| task | `grade` or `label` | read |
| question, gold_answer | the question, and the correct answer written from the filing | read |
| model_answer | what the model said | read |
| suggested_label | the scorer's guess at the failure type (label rows) | read |
| grader_1 / grader_2 | your grade on grade rows: 0, 1 or 2 | **your column only** |
| final | the agreed grade | **leave empty**; set together in step 3 |
| failure_label | the confirmed label | your label rows; optional on a grade row you grade 0 or 1 |
| note | your reason; `example:` or `overturn?` (spelled so) to mark a row | anyone |

## Step 1: grade every "grade" row (both of you; about 90 min)
Filter `task` to `grade`. Compare `model_answer` with `gold_answer`, check against the handout where needed, and enter in **your** column:

| Grade | Meaning |
|---|---|
| **2** | **Correct and grounded:** states the gold answer's key points (the main drivers, figures and reasons), contradicts none of them, and adds no claim the filing passages don't support. |
| **1** | **Partly correct:** some key points are right, but others are missing, vague or wrong; or the answer is right but adds an unsupported claim. |
| **0** | **Wrong:** misses the key points, contradicts the gold answer, answers a different question, or only says it can't answer. |

- **Wording, length and style don't matter.** A figure counts if it matches after rounding, e.g. $391.0 billion = $391,035 million.
- **An answer that leaves out the gold answer's figures** (e.g. "revenue grew $10.1 billion") but gives the right drivers is usually a **1**, not a 2: the figures are key points.
- **An extra claim you can't find in the gold answer or the passages** isn't grounded: give at most 1, and say why in `note`.
- **Boilerplate** with no specific drivers ("these risks may adversely affect our results") is 0.
- **A "which company had more" row:** give 2 if the answer clearly picks the company the gold answer names as having more, and 0 if it picks the other one or contradicts itself.
- **Can't decide?** Pick the lower grade and write a short note. You'll talk about it in step 3.
- **A label on a grade row (optional):** a summary graded 0 or 1 is labelled `incomplete_summary` and a "which company" answer graded 0 `wrong_company` unless you put another label in `failure_label`, e.g. `over_refusal` for a summary that only says it can't answer.

## Step 2: confirm the failure label on your label rows (about 60–75 min)
Filter `task` to `label` and work only on your rows, as listed in `assignments.txt`. For each row, see why the answer is wrong. Put one label in `failure_label`: copy `suggested_label` if it fits, or choose another. **Every one of your label rows needs a label**, spelled exactly as below:

| Label | Use it when the answer… |
|---|---|
| `wrong_fiscal_year` | uses the right line item from another year |
| `unit_confusion` | has the right number in the wrong unit or scale (thousands vs millions, 0.25 vs 25%) |
| `arithmetic_slip` | uses the right inputs but miscalculates: wrong % change, flipped sign, rounding outside the tolerance |
| `invented_number` | gives a number that appears nowhere in the filing or passages |
| `wrong_line_item` | gives a real figure from the filing, but the wrong one (another line, another quarter), or says "not reported" for a figure that is reported |
| `ignored_not_reported` | gives a value, or works one out, for a figure the filing doesn't report (trap questions) |
| `wrong_company` | answers about, or picks, the wrong company |
| `over_refusal` | refuses (or says it can't answer) an answerable question |
| `under_refusal` | answers a question it should refuse (forecasts, investment advice, uncovered companies, etc.) |
| `malformed_output` | isn't usable as an answer: invalid JSON, or the number field left empty |
| `incomplete_summary` | is a summary that's partly right but not enough to count |
| `wrong_refusal_rule` | refused (or answered in part) correctly, but named the wrong rule or no rule (e.g. R1 where the test expects R6) |

- **Two labels fit?** Choose the root cause. For example, a wrong company with a wrong number is `wrong_company`.
- **The automatic verdict looks wrong** (the answer is actually right, e.g. the right number is in the text but the number field is empty)? Still put the closest label, and write **`overturn?`** plus one line of reason in `note`.
- **A clear, teachable failure** (a confident invented number, a bad refusal, a wrong quarter)? Write **`example:`** and the reason in `note`. The report needs 5–6.

## Step 3: settle disagreements together (about 30 min, after both copies are back)
Alok checks and merges the two copies (commands below). The merge:
- refuses a copy with rows sorted, added or deleted;
- sets `final` wherever your two grades agree;
- prints the disagreements and the `overturn?` notes.

Then the two of you go through that list together:
- **For each disagreement:** agree a `final` grade and add one line of reason to `note`.
- **For each `overturn?`:** if you both agree the answer is right, set `final` = **2**. That overturns the automatic "wrong", and the report counts overturns.

Never change your own `grader_1` / `grader_2` grade after talking. The report shows your independent agreement (exact and Cohen's kappa) next to the judge's agreement with your `final` grades.

## Spreadsheet rules
- **Use filters, never sort.** Don't add or delete rows or columns, and don't edit `key` or `task`; the merge refuses such a file.
- **Grades are only `0`, `1` or `2`.** Labels must be spelled as in the table; anything else is rejected with its line number.
- **Harmless and undone automatically:** spaces around a grade or label, a label's capital letters, empty rows or columns Excel adds at the end. A grade like `2.0` or a misspelt label is an error naming the line and column.
- **If Excel re-formats an answer cell** (e.g. "7,367" → "7367"), that's harmless: the merge warns and keeps the original text.
- **In Excel, save with File → Save As → "CSV UTF-8 (Comma delimited)".** Google Sheets: File → Download → CSV. Close the file in Excel before the coordinator runs any tool on it; an open file is locked.

## Commands (coordinator)
```bash
# Before grading (from the merged main branch; copies go outside the repo and are not committed):
python eval/grading.py copies --names Chandra Amresh --out ../W1-14_grading
python eval/grading.py handout --out ../W1-14_grading/handout.md

# As soon as each copy comes back (a mistake can be fixed while the grader is still around):
python eval/grading.py check --grader 1 ../W1-14_grading/grading_sheet_Chandra.csv
python eval/grading.py check --grader 2 ../W1-14_grading/grading_sheet_Amresh.csv

# After both copies are back:
python eval/grading.py merge --g1 ../W1-14_grading/grading_sheet_Chandra.csv --g2 ../W1-14_grading/grading_sheet_Amresh.csv
python eval/grading.py merge --g1 ... --g2 ... --write   # writes eval/results/grading/grading_sheet.csv
# a corrected copy after --write: git checkout -- eval/results/grading/grading_sheet.csv, then merge again
# the graders settle the listed disagreements in `final` (and decide overturns), then:
python eval/score_answers.py                             # summary row, overturns, labels, human-vs-judge kappa
python eval/grading.py examples                          # rows noted "example" -> eval/results/failure_examples.md
```
Commit the sheet, the regenerated results and `failure_examples.md` in one PR for W1-14.

## A later round: the W1-09 strategy runs
The prompting baselines (W1-09) added 162 rows for the runs S1, S2 and S3 (38 to grade: 36 summaries and 2
company answers the scorer left open; 124 wrong answers to label) to the same sheet, next to the M1-M3 rows you
already graded. Grade them the same way, with `--pending` on every command: the copies then hold only the rows
nobody has graded yet (the label rows split in half among them), so your earlier grades are neither shown nor
changed, and the merge writes your entries back into exactly those rows.
```bash
python eval/grading.py copies --pending --names Chandra Amresh --out ../W1-09_grading
python eval/grading.py handout --pending --out ../W1-09_grading/handout.md
python eval/grading.py check --pending --grader 1 ../W1-09_grading/grading_sheet_Chandra.csv
python eval/grading.py merge --pending --g1 ../W1-09_grading/grading_sheet_Chandra.csv --g2 ../W1-09_grading/grading_sheet_Amresh.csv --write
python eval/score_answers.py                             # fills the summary row of eval/results/prompting_baselines.md
```

## A later round: the agent run A1 (W2-11)
The agent's gold run (A1, `eval/results/agent_baselines.md`) added 31 rows: 22 to grade (its 12 summaries and 10 company answers that name both companies) and 9 wrong answers to label. They join the round above with no extra step: `--pending` copies hold every row nobody graded yet, S1-S3's and A1's alike. The agent searched for itself, so the handout also lists, under "What a system that searched for itself retrieved", the passages and figures one of the answers rests on; check that answer against those, not only against the pack's passages above them. The handout still names no system.
