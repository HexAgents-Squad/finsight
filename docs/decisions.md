# Decision log
Every significant design choice gets one entry: context → decision → alternatives → evidence. This log becomes the "design decisions" content of the final report, so record choices as you make them (one PR, one entry).

| ID | Date | Owner | Decision | Why | Alternatives considered | Status |
|---|---|---|---|---|---|---|
| ADR-001 | 25 Oct 2026 | Team | Track B (API-side); domain: financial-report Q&A ("FinSight") | Measurable numeric answers; strong industry relevance; work splits cleanly across 6 members | Track A fine-tuning; procurement assistant; incident-postmortem drafter | Accepted (mentor call 26 Oct) |
| ADR-002 | 29 Oct 2026 | Alok (WP1) | Corpus scope = 20 companies in 4 sectors; 10-K FY2023–2025 + fiscal-2025 10-Qs (`config/companies.yaml`) | Covers different fiscal year-ends and statement layouts (tech, banks, pharma, retail); ~1,300 documents fits the 500–2,000 requirement | More companies (exceeds the cap); fewer sectors (less diverse evaluation) | Proposed (PR review) |
| ADR-003 | 29 Oct 2026 | Alok (WP1) | 1 document = 1 Item of 1 filing; exact Item naming shared by corpus and gold set | Natural citation unit ("MSFT 10-K FY2024 · Item 7"); enables recall@k against gold sources | Whole filing (too coarse); fixed-size pages (no meaningful citation) | Proposed (PR review) |
| ADR-004 | 29 Oct 2026 | Alok (WP1) | Fiscal year = the company's own fiscal-year label, never the calendar year | Apple, Microsoft, P&G, NVIDIA, Walmart and Costco have non-December year-ends; calendar-year matching returns the wrong filing | Calendar year of the period end | Proposed (PR review) |
| ADR-005 | 29 Oct 2026 | Alok (WP1) | Numbers only from tools or cited text; "not reported" instead of substitute line items; answer = validated JSON contract | Prevents hallucinated or mismatched figures; makes answers machine-checkable for evaluation | Free-text answers checked manually | Proposed (PR review) |
| ADR-006 | 29 Oct 2026 | Alok (WP1) | Refusal rules R1–R6 incl. instructions inside documents (prompt injection) | Regulated domain: no advice/forecasts; documents are untrusted input | Model's default safety behaviour only | Proposed (PR review) |

## Template
```
| ADR-0xx | <date> | <owner (WPn)> | <decision> | <why> | <alternatives> | Proposed / Accepted / Superseded by ADR-0yy |
```
Add numbers (recall@k, accuracy, timings) to the "Why" column whenever a decision is based on measurements.
