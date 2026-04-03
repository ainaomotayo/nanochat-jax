"""Multi-turn chat session management."""
from __future__ import annotations
import json
import structlog
from pathlib import Path
from typing import Any
from nanochat.inference.engine import InferenceEngine

logger = structlog.get_logger()


class ChatSession:
    """Stateful multi-turn chat session.

    Manages conversation history, context window, and chat template rendering.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        system_prompt: str = "You are a helpful assistant.",
        max_context_len: int = 2048,
    ):
        self.engine = engine
        self.system_prompt = system_prompt
        self.max_context_len = max_context_len
        self.history: list[dict[str, str]] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def chat(self, user_message: str, **kwargs: Any) -> str:
        """Send user message and get assistant response.

        Args:
            user_message: User's input text
            **kwargs: Passed to InferenceEngine.generate()

        Returns:
            Assistant's response text
        """
        self.history.append({"role": "user", "content": user_message})

        # Render conversation using chat template
        prompt = self.engine.tokenizer.apply_chat_template(self.history)

        # Truncate if too long (remove oldest turns, keep system prompt)
        while len(self.engine.tokenizer.encode(prompt)) > self.max_context_len and len(self.history) > 2:
            # Remove oldest user-assistant pair (keep system prompt at index 0)
            self.history.pop(1)
            if self.history[1]["role"] == "assistant":
                self.history.pop(1)
            prompt = self.engine.tokenizer.apply_chat_template(self.history)

        # Generate response
        response = self.engine.generate(prompt, **kwargs)
        assert isinstance(response, str)

        self.history.append({"role": "assistant", "content": response})
        logger.info("chat_turn", user_len=len(user_message), response_len=len(response),
                    history_turns=len(self.history))
        return response

    def reset(self) -> None:
        """Clear history, keep system prompt."""
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    def export_history(self, path: str | Path) -> None:
        """Save conversation history as JSON."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("history_exported", path=str(path), turns=len(self.history))

    @property
    def token_count(self) -> int:
        """Current total tokens in conversation context."""
        text = self.engine.tokenizer.apply_chat_template(self.history)
        return len(self.engine.tokenizer.encode(text))
