"""Tests for the character-level tokenizer."""

import pytest

from pytorch_gpt.tokenizer import CharTokenizer

# Capitals, spaces, punctuation and a newline: the characters most likely to go wrong
TEXT = "First Citizen:\nBefore we proceed any further, hear me speak."


def test_round_trip_returns_original_text():
    tokenizer = CharTokenizer(TEXT)
    assert tokenizer.decode(tokenizer.encode(TEXT)) == TEXT


def test_ids_follow_sorted_character_order():
    # Sorted distinct characters: e -> 0, h -> 1, l -> 2, o -> 3
    tokenizer = CharTokenizer("hello")
    assert tokenizer.vocab_size == 4
    assert tokenizer.encode("hello") == [1, 0, 2, 2, 3]


def test_vocabulary_ignores_character_order():
    assert CharTokenizer("abc").encode("abc") == CharTokenizer("cba").encode("abc")


def test_encode_rejects_unknown_character():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tokenizer.encode("d")
