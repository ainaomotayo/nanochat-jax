#!/usr/bin/env python3
"""Serve nanochat-jax with a ChatGPT-style web UI.

Usage::

    python -m scripts.chat_web                          # random weights, CPU
    python -m scripts.chat_web --device gpu              # random weights, GPU
    python -m scripts.chat_web --checkpoint checkpoints/latest --device gpu

Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import structlog

logger = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoChat-JAX Web Chat UI")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "tpu"], default="cpu")
    parser.add_argument("--checkpoint", type=str, default="RANDOM")
    parser.add_argument("--model-size", type=str, default="nano")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # 1. Setup device
    from nanochat.core.device import setup_device
    setup_device(args.device)

    # 2. Build model
    from flax import nnx
    from nanochat.config import ModelConfig
    from nanochat.model.transformer import TransformerLM
    from nanochat.inference.engine import InferenceEngine

    cfg = ModelConfig.for_scale(args.model_size)

    # Pick tokenizer to match model vocab size:
    #   vocab_size=256  → char-level (nano/small char presets)
    #   vocab_size>256  → BPE (medium/large/xlarge presets)
    if cfg.vocab_size == 256:
        from nanochat.tokenizer.char import CharTokenizer
        # Look for a saved vocab alongside the data dir or checkpoint
        script_dir = Path(__file__).resolve().parent.parent
        vocab_candidates = [
            script_dir / "data" / f"{args.model_size}_vocab.json",
            script_dir / "data" / "tinystories_vocab.json",
            script_dir / "data" / "shakespeare_vocab.json",
        ]
        vocab_file = next((p for p in vocab_candidates if p.exists()), None)
        if vocab_file:
            tokenizer = CharTokenizer.load(vocab_file)
            logger.info("tokenizer.char", source=str(vocab_file), vocab_size=tokenizer.vocab_size)
        else:
            import string
            tokenizer = CharTokenizer.from_text(string.printable)
            logger.warning("tokenizer.char.fallback", note="no vocab file found, using string.printable — output may be garbled if checkpoint was trained with a different vocab")
    else:
        from nanochat.tokenizer.bpe import BPETokenizer
        tokenizer = BPETokenizer.from_pretrained()
        logger.info("tokenizer.bpe", vocab_size=cfg.vocab_size)

    random_weights = args.checkpoint == "RANDOM"
    if random_weights:
        logger.info("model.random_init", model_size=args.model_size)
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))
    else:
        from nanochat.training.checkpoint import CheckpointManager
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error("checkpoint_not_found", path=str(ckpt_path))
            sys.exit(1)
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))
        ckpt_mgr = CheckpointManager(ckpt_path.parent)
        step = ckpt_mgr.load(ckpt_path, model)
        logger.info("model.checkpoint_loaded", path=str(ckpt_path), step=step)

    engine = InferenceEngine(model, tokenizer, cfg)

    # 3. Create app with web UI
    from nanochat.server.app import create_app
    from fastapi.responses import HTMLResponse

    app = create_app(engine)

    # Expose random_weights status for UI banner
    @app.get("/status")
    def status():
        return {"random_weights": random_weights, "model_size": args.model_size}

    # Serve the chat UI at root
    ui_path = Path(__file__).resolve().parent.parent / "src" / "nanochat" / "server" / "ui.html"
    ui_html = ui_path.read_text()

    @app.get("/", response_class=HTMLResponse)
    def chat_ui():
        return ui_html

    logger.info(
        "web_ui.starting",
        url=f"http://localhost:{args.port}",
        device=args.device,
        model_size=args.model_size,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
