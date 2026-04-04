"""Character-level tokenizer implementing BaseTokenizer.

Suitable for TinyShakespeare and other character-granularity experiments.
Vocabulary = sorted unique chars in training text + PAD/BOS/EOS specials.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from nanochat.tokenizer.base import BaseTokenizer


_SPECIALS = {"<pad>": 0, "<bos>": 1, "<eos>": 2}


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer.

    Special tokens occupy the first three IDs:
    - 0 = ``<pad>``
    - 1 = ``<bos>``
    - 2 = ``<eos>``

    All other IDs map to individual characters from the training corpus.

    Example::

        tok = CharTokenizer.from_text(open("data/tinyshakespeare.txt").read())
        ids = tok.encode("Hello")        # [10, 11, 18, 18, 21] (example)
        text = tok.decode(ids)           # "Hello"
        tok.save("data/char_vocab.json")
        tok2 = CharTokenizer.load("data/char_vocab.json")
    """

    def __init__(self, ch2id: dict[str, int]):
        self._ch2id = dict(ch2id)
        self._id2ch = {v: k for k, v in ch2id.items()}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build vocabulary from all unique characters in *text*."""
        ch2id = dict(_SPECIALS)
        for c in sorted(set(text)):
            if c not in ch2id:
                ch2id[c] = len(ch2id)
        return cls(ch2id)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load vocabulary from a JSON file saved by :meth:`save`."""
        with open(path) as f:
            ch2id = json.load(f)
        return cls(ch2id)

    def save(self, path: str | Path) -> None:
        """Persist vocabulary to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._ch2id, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # BaseTokenizer interface
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._ch2id)

    @property
    def bos_id(self) -> int:
        return self._ch2id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._ch2id["<eos>"]

    @property
    def pad_id(self) -> int:
        return self._ch2id["<pad>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode *text* to a list of token IDs."""
        ids = [self._ch2id.get(c, 0) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(
        self,
        ids: list[int] | np.ndarray,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to a string."""
        chars = []
        for i in ids:
            ch = self._id2ch.get(int(i), "")
            if skip_special_tokens and ch in ("<pad>", "<bos>", "<eos>"):
                continue
            chars.append(ch)
        return "".join(chars)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Minimal chat template: concatenate role and content."""
        parts = []
        for msg in messages:
            parts.append(f"[{msg['role']}] {msg['content']}\n")
        return "".join(parts)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
