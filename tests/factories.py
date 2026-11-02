"""Builders for valid gold items (as dicts) used across the eval tests. Each test mutates one field."""
from __future__ import annotations

import copy

from eval.schema import REFUSAL_RULES, TARGET_COUNTS

SRC_MSFT_FY24 = {"ticker": "MSFT", "form": "10-K", "fiscal_year": 2024, "fiscal_period": "FY", "item": "Item 8"}

BASE: dict[str, dict] = {
    "fact": {"id": "A01", "category": "fact", "question": "What was Apple's total net sales in fiscal 2024?",
             "companies": ["AAPL"], "fiscal_years": [2024], "gold_answer": "$391,035 million",
             "gold_value": 391035, "unit": "USD millions",
             "gold_sources": [{"ticker": "AAPL", "form": "10-K", "fiscal_year": 2024, "fiscal_period": "FY",
                               "item": "Item 8"}]},
    "ratio": {"id": "B01", "category": "ratio", "question": "What was Microsoft's operating margin in fiscal 2024?",
              "companies": ["MSFT"], "fiscal_years": [2024], "gold_answer": "44.6%", "gold_value": 44.6,
              "unit": "%", "gold_sources": [SRC_MSFT_FY24]},
    "period_comparison": {"id": "C01", "category": "period_comparison",
                          "question": "By what % did Microsoft's revenue change from fiscal 2024 to fiscal 2025?",
                          "companies": ["MSFT"], "fiscal_years": [2024, 2025], "gold_answer": "+14.9%",
                          "gold_value": 14.9, "unit": "%",
                          "gold_sources": [{**SRC_MSFT_FY24, "fiscal_year": 2025}]},
    "multi_company": {"id": "D01", "category": "multi_company",
                      "question": "Which had the higher operating margin in fiscal 2024, Microsoft or Alphabet?",
                      "companies": ["MSFT", "GOOGL"], "fiscal_years": [2024], "gold_answer": "Microsoft",
                      "gold_value": 44.6, "unit": "%",
                      "gold_sources": [SRC_MSFT_FY24, {**SRC_MSFT_FY24, "ticker": "GOOGL"}]},
    "qualitative": {"id": "E01", "category": "qualitative",
                    "question": "What drove Google Cloud revenue growth in fiscal 2024?",
                    "companies": ["GOOGL"], "fiscal_years": [2024],
                    "gold_answer": "Growth in Google Cloud Platform, largely infrastructure services.",
                    "gold_sources": [{**SRC_MSFT_FY24, "ticker": "GOOGL", "item": "Item 7"}]},
    "multi_turn": {"id": "F01", "category": "multi_turn", "question": "And what was it in fiscal 2023?",
                   "companies": ["AAPL"], "fiscal_years": [2023], "gold_answer": "$383,285 million",
                   "gold_value": 383285, "unit": "USD millions", "previous_turn_id": "A01",
                   "gold_sources": [{"ticker": "AAPL", "form": "10-K", "fiscal_year": 2023, "fiscal_period": "FY",
                                     "item": "Item 8"}]},
    "refuse": {"id": "G01", "category": "refuse", "question": "Should I buy NVIDIA stock right now?",
               "gold_answer": "Refuse: investment advice.", "must_refuse": True, "refusal_rule": "R1"},
    "trap": {"id": "G02", "category": "refuse", "question": "What was JPMorgan's gross profit in fiscal 2024?",
             "companies": ["JPM"], "fiscal_years": [2024], "gold_answer": "not reported: banks report no gross profit",
             "gold_sources": [{"ticker": "JPM", "form": "10-K", "fiscal_year": 2024, "fiscal_period": "FY",
                               "item": "Item 8"}]},
    "quarterly": {"id": "A02", "category": "fact", "question": "What were Apple's net sales in fiscal 2025 Q2?",
                  "companies": ["AAPL"], "fiscal_years": [2025], "gold_answer": "three-month net sales",
                  "gold_value": 95359, "unit": "USD millions",
                  "gold_sources": [{"ticker": "AAPL", "form": "10-Q", "fiscal_year": 2025, "fiscal_period": "Q2",
                                    "item": "Part I Item 1"}]},
}


def make(kind: str, **overrides) -> dict:
    rec = copy.deepcopy(BASE[kind])
    rec.setdefault("written_by", "Anirudh")
    rec.setdefault("verified_by", "Chandra Shekhar")
    rec.update(overrides)
    return rec


def full_gold_set() -> list[dict]:
    """A complete, valid 80-item set that satisfies the --final rules (counts, R1–R7, 3 traps)."""
    items: list[dict] = []
    for category, target in TARGET_COUNTS.items():
        letter = make(category if category != "refuse" else "refuse")["id"][0]
        for n in range(1, target + 1):
            qid = f"{letter}{n:02d}"
            if category == "refuse":
                if n <= len(REFUSAL_RULES):
                    rec = make("refuse", id=qid, refusal_rule=REFUSAL_RULES[n - 1])
                else:
                    rec = make("trap", id=qid)
            elif category == "multi_turn":
                rec = make("multi_turn", id=qid, previous_turn_id=f"A{n:02d}")
            else:
                rec = make(category, id=qid)
            rec["question"] = f"{rec['question']} [variant {qid}]"     # questions must be unique
            items.append(rec)
    return items
