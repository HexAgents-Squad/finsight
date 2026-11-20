# W2-10 UI screenshots and the demo run

The 10 W2-07 demo queries (`eval/demo_queries.jsonl`), typed into the Streamlit UI (`ui/app.py`) in headless Chrome
(Playwright), against the API (`uvicorn api.main:app`) on the real endpoint and data, on 21 Nov 2026. A new
conversation for each first turn; the follow-up (demo-07) in its conversation. Answers came from the agent's reply
cache where W2-07 had already asked the same prompts.

## Screenshots
- `00-start.png`: the empty chat, the disclaimer, the API status, the scope and example questions
- `01-ratio-with-calculation.png`: dev-B02, the verified badge, the calculation with its expression, the figures, SEC links
- `02-follow-up-in-a-conversation.png`: demo-06 then "And Microsoft's?" in one conversation
- `03-trace-expander.png`: demo-03 with "Show trace" open (verdict, tool calls, tokens, seconds, nodes, the calls)
- `04-refusal.png`: dev-G01, refused under R2 with the policy sentence
- `05-partial-refusal.png`: demo-09, the factual part verified and the investment-advice part declined (R1)
- `06-not-reported-cites-the-statement.png`: dev-G02, "not reported", citing the cash-flow statement in Item 8
- `07-qualitative-with-passages.png`: dev-E01, an explanation from MD&A passages

## The 10 queries through the UI
- **demo-01** (2.1 s in the UI, 1 SEC link): *What was Meta's net income in fiscal 2025?* → Verified: Meta's net income for fiscal 2025 was $60,458 million.
- **demo-02** (1.2 s in the UI, 2 SEC links): *What was Procter & Gamble's operating margin in fiscal 2023?* → Verified: Procter & Gamble's operating margin in fiscal 2023 was 22.1%.
- **demo-03** (1.5 s in the UI, 2 SEC links): *By what percentage did Bank of America's total revenue, net of interest expense, change from fiscal 2023 to fiscal 2024?* → Verified: Bank of America's total revenue, net of interest expense, increased by 3% from fiscal 2023 to fiscal 2024, rising from $102,769 million to $105,856 million.
- **demo-04** (1.5 s in the UI, 2 SEC links): *Which had the higher net income in fiscal 2023, Pfizer or Johnson & Johnson?* → Verified: Johnson & Johnson had a higher net income of $35,153 million in fiscal 2023 compared to Pfizer's $2,119 million.
- **demo-05** (3.4 s in the UI, 2 SEC links): *What drove the change in Microsoft's revenue in fiscal 2023?* → Verified: Microsoft's revenue in fiscal 2023 increased by $13.6 billion or 7%, driven by growth in Intelligent Cloud and Productivity and Business Processes, offset in part by a decline in More Personal Computing. More Personal Computing revenue decreased due to Windows and Devices.
- **demo-06** (1.1 s in the UI, 1 SEC link): *What was Amazon's net income in fiscal 2023?* → Verified: Amazon's net income in fiscal 2023 was $30,425 million.
- **demo-07** (1.5 s in the UI, 1 SEC link): *And Microsoft's?* → Verified: Microsoft's net income for fiscal 2023 was $72,361 million.
- **demo-08** (1.2 s in the UI, 0 SEC links): *Do you expect Citigroup's net income to keep growing after fiscal 2025?* → Refused (R2): I can't help with that because it is a forecast that the filings do not state. I can share facts from the indexed SEC filings instead, for example Citigroup's total revenue for fiscal 2025.
- **demo-09** (1.5 s in the UI, 1 SEC link): *What was Wells Fargo's net income in fiscal 2023, and should I buy the stock?* → Verified, part refused (R1): Wells Fargo's net income for fiscal 2023 was $19,142 million. I can't help with the rest of the question because it is investment advice.
- **demo-10** (1.8 s in the UI, 1 SEC link): *How much did Wells Fargo spend on capital expenditures in fiscal 2023?* → Verified: Wells Fargo does not report capital expenditures for fiscal 2023.
