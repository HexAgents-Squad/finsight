# Failure examples (W1-14)

Rows the graders marked `example` in eval/results/grading/grading_sheet.csv, in sheet order.

## D05: wrong_company

- **Run:** M1_qwen3-8b_rag_think-off_b0-v1
- **Question:** Which spent more on research and development in fiscal 2024, Johnson & Johnson or Pfizer?
- **Model answer:** Pfizer spent more on research and development in fiscal 2024.
- **Gold answer:** Johnson & Johnson: $17,232 million (year ended December 29, 2024) vs Pfizer: $10,822 million (year ended December 31, 2024). Johnson & Johnson spent more.
- **Why:** example: Picks Pfizer, but the gold figures are $17,232m for J&J versus $10,822m for Pfizer; J&J spent more.

## G09: ignored_not_reported

- **Run:** M3_qwen3-8b_closed_think-off_b0-v1
- **Question:** What were Amazon's total liabilities at the end of fiscal 2024?
- **Model answer:** Amazon's total liabilities at the end of fiscal 2024 were reported as $135,550 million.
- **Gold answer:** not reported: Amazon's balance sheet has no total liabilities line. It reports total current liabilities ($179,431 million) and total stockholders' equity ($285,970 million) as of December 31, 2024 (Item 8).
- **Why:** example: Constructs total liabilities from component liabilities even though the filing has no total-liabilities line; the question is a not-reported trap.

## A12: wrong_line_item

- **Run:** M1_qwen3-8b_rag_think-off_b0-v1
- **Question:** What were Coca-Cola's net operating revenues in the third quarter of fiscal 2025?
- **Model answer:** Coca-Cola's net operating revenues in the third quarter of fiscal 2025 were $12,535 million.
- **Gold answer:** Coca-Cola's net operating revenues were $12,455 million for the three months ended September 26, 2025 (fiscal 2025 Q3).
- **Why:** example: Gives $12,535 million, the Q2 figure (three months ended June 27, 2025), for a Q3 question; gold $12,455 million. Only the Q2 table was retrieved, so the model took the nearest figure on offer: a wrong period, 0.6% away.

## C02: malformed_output

- **Run:** M2_llama3.2-3b_rag_think-off_b0-v1
- **Question:** How much did Meta's net income grow from fiscal 2023 to fiscal 2024, in percent?
- **Model answer:** {"answer": "24.1", "value": "24.1", "unit": "%", "company": "META PLATFORMS, INC.", "fiscal_year": 2024, "citations": ["P8"], "refused": false, "refusal_rule": null}
- **Gold answer:** Meta's net income increased 59.5%, from $39,098 million in fiscal 2023 to $62,360 million in fiscal 2024.
- **Why:** example: Invalid JSON even after the retry (the value is the quoted string "24.1"), and 24.1% is not the gold 59.5% increase in net income.

## C04: over_refusal

- **Run:** M1_qwen3-8b_rag_think-off_b0-v1
- **Question:** By what percentage did JPMorgan Chase's net income change from fiscal 2023 to fiscal 2024?
- **Model answer:** The percentage change in JPMorgan Chase's net income from fiscal 2023 to fiscal 2024 is not reported in the provided passages.
- **Gold answer:** JPMorgan Chase's net income increased 18.0%, from $49,552 million in fiscal 2023 to $58,471 million in fiscal 2024.
- **Why:** example: Says the change is not in the passages, and it is right: the retrieved passages lack JPMorgan's net income lines ($49,552 million and $58,471 million; gold +18.0%). A retrieval miss that is scored as an over-refusal.

## E01: over_refusal

- **Run:** M3_qwen3-8b_closed_think-off_b0-v1
- **Question:** What drove the growth in Google Cloud revenues in fiscal 2024?
- **Model answer:** The question asks for an analysis of factors driving growth, which is not directly reported in the filings.
- **Gold answer:** Google Cloud revenues increased $10.1 billion from 2023 to 2024. Alphabet attributes the increase primarily to growth in Google Cloud Platform, largely from infrastructure services (Item 7, MD&A).
- **Why:** example: Closed-book: refuses to summarise the drivers of Google Cloud growth as not reported, although the 10-K MD&A explains them (growth in Google Cloud Platform, infrastructure services).

## C01: arithmetic_slip

- **Run:** M1_qwen3-8b_rag_think-off_b0-v1
- **Question:** By what percentage did NVIDIA's revenue change from fiscal 2024 to fiscal 2025?
- **Model answer:** NVIDIA's revenue increased by 12% from fiscal 2024 to fiscal 2025.
- **Gold answer:** NVIDIA's revenue increased 114.2%, from $60,922 million in fiscal 2024 to $130,497 million in fiscal 2025 (years ended January 28, 2024 and January 26, 2025).
- **Why:** example: Says NVIDIA revenue grew 12%; the gold is 114.2% ($60,922 million to $130,497 million). Neither revenue figure was in the retrieved passages, yet the model gave a precise growth rate instead of saying it could not compute it.
