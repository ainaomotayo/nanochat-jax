#!/usr/bin/env python3
"""Interactive chat REPL for nanochat-jax.

Usage:
    python scripts/chat.py --model-size nano
    python scripts/chat.py --checkpoint checkpoints/latest

Commands:
    /reset    - Clear conversation history
    /history  - Show full conversation
    /export   - Save conversation to JSON
    /quit     - Exit
    /help     - Show commands
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
from flax import nnx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from nanochat.config import ModelConfig
from nanochat.model.transformer import TransformerLM
from nanochat.tokenizer.bpe import BPETokenizer
from nanochat.inference.engine import InferenceEngine
from nanochat.inference.chat import ChatSession

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoChat-JAX Interactive Chat")
    parser.add_argument("--model-size", type=str, default="nano")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.")
    args = parser.parse_args()

    console.print(Panel("[bold blue]NanoChat-JAX[/bold blue] Interactive Chat", expand=False))
    console.print(f"Model: {args.model_size} | Temp: {args.temperature} | Top-k: {args.top_k}")
    console.print("Type /help for commands\n")

    # Build model and engine
    cfg = ModelConfig.for_scale(args.model_size)
    model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))
    tokenizer = BPETokenizer.from_pretrained()
    engine = InferenceEngine(model, tokenizer, cfg)
    session = ChatSession(engine, system_prompt=args.system_prompt, max_context_len=cfg.max_seq_len)

    console.print("[dim]Model loaded. Note: untrained model will produce random output.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.strip().lower()
            if cmd == "/quit" or cmd == "/exit":
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/reset":
                session.reset()
                console.print("[dim]Conversation reset.[/dim]")
            elif cmd == "/history":
                for msg in session.history:
                    role_color = {"system": "gray", "user": "blue", "assistant": "green"}.get(msg["role"], "white")
                    console.print(f"[{role_color}][{msg['role']}][/{role_color}]: {msg['content']}")
            elif cmd == "/export":
                session.export_history("chat_history.json")
                console.print("[dim]Exported to chat_history.json[/dim]")
            elif cmd == "/help":
                console.print("/reset - Clear history | /history - Show history | /export - Save JSON | /quit - Exit")
            else:
                console.print(f"[dim]Unknown command: {cmd}[/dim]")
            continue

        # Generate response
        try:
            response = session.chat(
                user_input,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=args.max_tokens,
            )
            console.print(f"[bold green]Assistant:[/bold green] {response}\n")
            console.print(f"[dim]({session.token_count} tokens in context)[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
