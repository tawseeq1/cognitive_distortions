"""Tests for tokenisation and n-gram generation."""
from src.preprocessing.tokenize import generate_ngrams, ngrams, sentence_tokens, word_tokens


def test_word_tokens_lowercases_and_splits():
    toks = word_tokens("I WILL Fail Today")
    assert "will" in toks and "fail" in toks
    assert all(t == t.lower() for t in toks)


def test_word_tokens_preserves_contractions():
    toks = word_tokens("he won't know")
    # The distortion dictionary contains "won't"; the tokenizer must keep it intact.
    assert "won't" in toks


def test_ngrams_orders():
    toks = ["a", "b", "c"]
    assert ngrams(toks, 1) == ["a", "b", "c"]
    assert ngrams(toks, 2) == ["a b", "b c"]
    assert ngrams(toks, 4) == []  # not enough tokens


def test_generate_ngrams_dict_keys():
    out = generate_ngrams(["a", "b", "c"], [1, 2, 3])
    assert set(out.keys()) == {1, 2, 3}
    assert out[3] == ["a b c"]


def test_sentence_tokens_splits_on_terminators():
    sents = sentence_tokens("First one. Second one! Third?")
    assert len(sents) == 3
