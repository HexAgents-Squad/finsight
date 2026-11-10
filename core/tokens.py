"""Token counts in Qwen3-8B's own tokenizer, the model behind the hosted endpoint (8,192 tokens per call).

Why the real tokenizer and not a characters-per-token estimate: measured on C3 chunks (10 Nov 2026), filing
text runs from about 4.6 characters per token (prose) down to 2.2 (statement tables: digits and "|"), so any
single ratio either wastes half the budget on prose or overruns it on tables. This tokenizer reproduced the
endpoint's prompt_tokens exactly on three sample batches (1,048 to 3,365 tokens, difference 0).

The tokenizer (a few MB from Hugging Face) is pinned to one revision so a count never changes between runs; it
downloads once into the Hugging Face cache, like the embedding model. `tokenizers` comes with
sentence-transformers. Offline tests pass their own counter instead of calling count_tokens.
"""
from __future__ import annotations

from functools import cache
from typing import Any

from core.exceptions import FinSightError

QWEN_TOKENIZER = "Qwen/Qwen3-8B"
QWEN_TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"      # Hugging Face commit, 10 Nov 2026


@cache
def qwen_tokenizer() -> Any:
    """The pinned Qwen3 tokenizer, loaded once. Failures are not cached, so a retry after going online works."""
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_pretrained(QWEN_TOKENIZER, revision=QWEN_TOKENIZER_REVISION)
    # tokenizers raises a plain Exception for a failed download or a missing revision, so nothing narrower works
    except Exception as err:
        raise FinSightError(f"Qwen tokenizer {QWEN_TOKENIZER}@{QWEN_TOKENIZER_REVISION[:12]} unavailable "
                            f"(needs the tokenizers package and internet once): {err}",
                            error_type="TokenizerUnavailable") from err


def count_tokens(text: str) -> int:
    """Tokens `text` takes in a Qwen3 prompt, without special tokens (the chat template adds its own)."""
    return len(qwen_tokenizer().encode(text, add_special_tokens=False).ids)
