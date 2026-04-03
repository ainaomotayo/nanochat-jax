from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseTokenizer(ABC):
    """Abstract tokenizer interface for nanochat-jax.

    All tokenizer implementations must provide:
    - encode/decode for single strings
    - encode_batch for batched encoding with padding
    - apply_chat_template for conversation formatting
    - Special token IDs (bos, eos, pad)
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def bos_id(self) -> int: ...

    @property
    @abstractmethod
    def eos_id(self) -> int: ...

    @property
    @abstractmethod
    def pad_id(self) -> int: ...

    @abstractmethod
    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int] | np.ndarray, *, skip_special_tokens: bool = True) -> str: ...

    def encode_batch(self, texts: list[str], max_length: int, *,
                     padding: bool = True, truncation: bool = True,
                     add_bos: bool = False, add_eos: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Encode batch with right-padding. Returns (input_ids [B, max_len], attention_mask [B, max_len])."""
        batch_ids = []
        batch_mask = []
        for text in texts:
            ids = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding and len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_mask.append(mask)
        return np.array(batch_ids, dtype=np.int32), np.array(batch_mask, dtype=np.int32)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Apply ChatML-style template. Override in subclass for custom formats."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}\n<|end|>")
        parts.append("<|assistant|>\n")
        return "".join(parts)
