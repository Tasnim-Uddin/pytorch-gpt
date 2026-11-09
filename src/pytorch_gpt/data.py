"""
Data pipeline: character-level tokenizer, train/validation split, and random batches.
"""

from pathlib import Path

import torch


class CharTokenizer:
    """
    Character-level tokenizer: every distinct character in the text is one token.
    IDs are 0 .. V-1, assigned in sorted order so they are the same on every run.
    """

    def __init__(self, text):
        # Vocabulary: every distinct character, sorted (a set has no fixed order)
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)  # V

        # Lookup tables in both directions
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}  # for encode
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}  # for decode

    def encode(self, text):
        """
        text:    string
        returns: list of token IDs, one per character
        """
        return [self.char_to_id[char] for char in text]

    def decode(self, ids):
        """
        ids:     sequence of token IDs (list, or a 1D tensor converted with .tolist())
        returns: the string they represent
        """
        return "".join(self.id_to_char[i] for i in ids)
