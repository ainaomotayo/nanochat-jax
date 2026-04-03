import structlog
import numpy as np
from nanochat.tokenizer.base import BaseTokenizer

logger = structlog.get_logger()


class BPETokenizer(BaseTokenizer):
    """BPE tokenizer using tiktoken (cl100k_base encoding by default).

    Provides GPT-4 compatible tokenization with added special tokens
    for chat formatting.
    """

    # Define special tokens with IDs starting after the base vocab
    SPECIAL_TOKENS = {
        "<|bos|>": 100277,
        "<|eos|>": 100278,
        "<|pad|>": 100279,
        "<|user|>": 100280,
        "<|assistant|>": 100281,
        "<|system|>": 100282,
        "<|end|>": 100283,
    }

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for BPETokenizer. "
                "Install it with: pip install tiktoken"
            ) from e

        base_enc = tiktoken.get_encoding(encoding_name)
        self._enc = tiktoken.Encoding(
            name=f"{encoding_name}_chat",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens={**base_enc._special_tokens, **self.SPECIAL_TOKENS},
        )
        self._encoding_name = encoding_name

        # Cache special token IDs
        self._bos_id = self.SPECIAL_TOKENS["<|bos|>"]
        self._eos_id = self.SPECIAL_TOKENS["<|eos|>"]
        self._pad_id = self.SPECIAL_TOKENS["<|pad|>"]
        self._vocab_size = self._enc.n_vocab

        logger.info("tokenizer_initialized", encoding=encoding_name, vocab_size=self._vocab_size)

    @classmethod
    def from_pretrained(cls, name: str = "cl100k_base") -> "BPETokenizer":
        return cls(encoding_name=name)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._enc.encode(text, allowed_special="all")
        if add_bos:
            ids = [self._bos_id] + ids
        if add_eos:
            ids = ids + [self._eos_id]
        return ids

    def decode(self, ids, *, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        if skip_special_tokens:
            special_ids = set(self.SPECIAL_TOKENS.values())
            ids = [i for i in ids if i not in special_ids]
        return self._enc.decode(ids)
