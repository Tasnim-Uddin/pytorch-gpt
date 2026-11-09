"""Tokenizers that convert between text and integer token IDs."""

from collections.abc import Iterable


class CharTokenizer:
    """Character-level tokenizer: every distinct character in the text is one token.

    IDs run from 0 to V-1 in sorted character order, so they are the same on every run.

    Args:
        text: Text whose distinct characters form the vocabulary.
    """

    def __init__(self, text: str) -> None:
        # Vocabulary: every distinct character, sorted (a set has no fixed order)
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)  # V

        # Lookup tables in both directions
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}  # for encode
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}  # for decode

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs, one per character.

        Raises:
            KeyError: If `text` contains a character outside the vocabulary.
        """
        return [self.char_to_id[char] for char in text]

    def decode(self, ids: Iterable[int]) -> str:
        """Convert token IDs back to text.

        Args:
            ids: Token IDs as Python ints; convert a tensor with `.tolist()` first.
        """
        return "".join(self.id_to_char[i] for i in ids)
