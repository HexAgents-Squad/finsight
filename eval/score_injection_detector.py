"""W2-13: how much of each phrasing set agent/injection.py catches, and how much filing-like text it flags. Offline.

    python eval/score_injection_detector.py          # one line per set (eval/injection_detector_sets.json)
    python eval/score_injection_detector.py --list   # plus every missed attack and flagged benign sentence

The dev_* sets were used to tune the patterns, so their rates are high by construction; heldout_2 was written
without seeing the patterns and measures them (eval/results/injection_results.md). The corpus itself is checked by
tests/test_injection.py (no sentence of the three chunkings is flagged).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETS_FILE = ROOT / "eval" / "injection_detector_sets.json"


def load_sets(path: Path = SETS_FILE) -> dict[str, dict]:
    return {name: body for name, body in json.loads(path.read_text(encoding="utf-8")).items() if name != "about"}


def score(body: dict) -> dict:
    from agent.injection import tampered
    missed = [a for a in body["attacks"] if not tampered(a)]
    flagged = [b for b in body["benign"] if tampered(b)]
    return {"attacks": len(body["attacks"]), "caught": len(body["attacks"]) - len(missed), "missed": missed,
            "benign": len(body["benign"]), "flagged": flagged}


def main(argv: list[str] | None = None) -> int:
    from eval.validate_gold import safe_console
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--list", action="store_true", help="print the missed attacks and the flagged benign sentences")
    args = ap.parse_args(argv)
    for name, body in load_sets().items():
        s = score(body)
        print(f"{name:14} attacks caught {s['caught']}/{s['attacks']}; benign flagged {len(s['flagged'])}/{s['benign']}")
        if args.list:
            for text in s["missed"]:
                print(f"  missed:  {text!r}")
            for text in s["flagged"]:
                print(f"  flagged: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
