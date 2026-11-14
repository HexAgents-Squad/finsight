"""core.tokens.cut_to_embedding_tokens with a stand-in tokenizer (tests never download one): word tokens plus
[CLS] and [SEP], with offsets into the original text, as the bge-small tokenizer reports them."""
import re
from types import SimpleNamespace

import pytest

import core.tokens as tk


class WordTokenizer:
    """One token per word or punctuation mark; offsets index the original string; [CLS]/[SEP] get (0, 0)."""

    def encode(self, text):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        return SimpleNamespace(ids=[0] * (len(spans) + 2), offsets=[(0, 0), *spans, (0, 0)])


@pytest.fixture(autouse=True)
def words(monkeypatch):
    monkeypatch.setattr(tk, "embed_tokenizer", lambda: WordTokenizer())


def test_a_text_that_fits_is_unchanged():
    assert tk.cut_to_embedding_tokens("Net sales rose 6%.", 8) == ("Net sales rose 6%.", False)   # 6 + 2 = 8


def test_a_long_text_is_cut_where_a_token_ends_and_fits_with_the_special_tokens():
    text = "Revenue | 391,035 | 383,285 Net income | 93,736"
    cut, was = tk.cut_to_embedding_tokens(text, 7)                       # 5 content tokens + [CLS] + [SEP]
    assert was and cut == "Revenue | 391,035" and text.startswith(cut)
    assert len(WordTokenizer().encode(cut).ids) <= 7


def test_the_original_spelling_is_kept():
    cut, _ = tk.cut_to_embedding_tokens("Apple's TOTAL Net Sales were high", 5)
    assert cut == "Apple's"                                              # case and punctuation as written


def test_the_pinned_revision_is_a_full_commit_hash():
    assert re.fullmatch(r"[0-9a-f]{40}", tk.EMBED_TOKENIZER_REVISION) and tk.EMBED_TOKENIZER == "BAAI/bge-small-en-v1.5"


def test_text_without_spaces_is_cut_at_the_token_not_emptied():
    # table rows often have no spaces near the limit: the cut must follow the tokens, not fall back to a space
    assert tk.cut_to_embedding_tokens("Revenue|391,035|383,285", 5) == ("Revenue|391", True)   # 3 + [CLS] + [SEP]


class SplitWordTokenizer(WordTokenizer):
    """A prefix that ends in a number takes two more tokens, as a word cut mid-way re-tokenises into pieces."""

    def encode(self, text):
        enc = super().encode(text)
        extra = 2 if text[-1:].isdigit() else 0
        return SimpleNamespace(ids=enc.ids + [0] * extra, offsets=enc.offsets)


@pytest.mark.parametrize("text, cut", [
    ("Revenue 391 383 net income", "Revenue"),          # 3 content tokens, then 2, re-tokenise too long
    ("Revenue|391|383|net", "Revenue|"),                # no space to fall back to: still not emptied
])
def test_a_cut_that_re_tokenises_longer_backs_off_a_token_at_a_time(monkeypatch, text, cut):
    monkeypatch.setattr(tk, "embed_tokenizer", lambda: SplitWordTokenizer())
    assert tk.cut_to_embedding_tokens(text, 5) == (cut, True)
    assert len(SplitWordTokenizer().encode(cut).ids) <= 5


def test_the_back_off_stops_at_one_token(monkeypatch):
    class AlwaysLonger(WordTokenizer):
        def encode(self, text):
            enc = super().encode(text)
            return SimpleNamespace(ids=enc.ids * 3, offsets=enc.offsets)

    monkeypatch.setattr(tk, "embed_tokenizer", lambda: AlwaysLonger())
    assert tk.cut_to_embedding_tokens("alpha beta gamma delta", 4) == ("alpha", True)
