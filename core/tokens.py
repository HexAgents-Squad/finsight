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
# The hosted embedding model's tokenizer (W1-16): it counts exactly as the endpoint does, [CLS] and [SEP]
# included (eval/results/embedding_probe.md: a text of 512 tokens by this count is accepted and billed as 512
# prompt tokens, 513 is refused). BERT uncased WordPiece, the same vocabulary all-MiniLM-L6-v2 uses.
EMBED_TOKENIZER = "BAAI/bge-small-en-v1.5"
EMBED_TOKENIZER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"    # Hugging Face commit, 14 Nov 2026


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


@cache
def embed_tokenizer() -> Any:
    """The pinned bge-small-en-v1.5 tokenizer, loaded once (failures are not cached)."""
    try:
        from tokenizers import Tokenizer
        return Tokenizer.from_pretrained(EMBED_TOKENIZER, revision=EMBED_TOKENIZER_REVISION)
    except Exception as err:            # tokenizers raises a plain Exception for a failed download
        raise FinSightError(f"embedding tokenizer {EMBED_TOKENIZER}@{EMBED_TOKENIZER_REVISION[:12]} unavailable "
                            f"(needs the tokenizers package and internet once): {err}",
                            error_type="TokenizerUnavailable") from err


def cut_to_embedding_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """`text`, or its longest prefix that takes at most `max_tokens` embedding tokens including [CLS] and [SEP],
    cut where a token ends in the original string (no re-spelling: the uncased tokenizer would lower-case a
    decoded text). Returns (text, whether it was cut)."""
    enc = embed_tokenizer().encode(text)
    if len(enc.ids) <= max_tokens:
        return text, False
    keep = max_tokens - 2       # content tokens; [CLS] and [SEP] take the other two (EMBED_MAX_INPUT_TOKENS > 2)
    while True:
        cut = text[:enc.offsets[keep][1]]           # offsets[0] is [CLS]: content token n is offsets[n]
        # A word split by the cut can re-tokenise into more pieces: back off one token at a time (never to an
        # empty text; measured on every C2/C3 chunk over 512 tokens, the first cut always fits)
        if keep == 1 or len(embed_tokenizer().encode(cut).ids) <= max_tokens:
            return cut, True
        keep -= 1
