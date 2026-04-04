#!/usr/bin/env python3
"""Serve nanochat-jax as an OpenAI-compatible API.

Usage::

    python scripts/serve.py --device cpu --checkpoint RANDOM --port 8000
    python scripts/serve.py --device gpu --checkpoint checkpoints/latest
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
    parser = argparse.ArgumentParser(description="NanoChat-JAX API Server")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        default="cpu",
        help="Device backend (default: cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="RANDOM",
        help="Path to checkpoint directory or RANDOM for random weights (default: RANDOM)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="nano",
        help="Model scale preset when using RANDOM checkpoint (default: nano)",
    )
    args = parser.parse_args()

    # 1. Setup device
    from nanochat.core.device import setup_device

    setup_device(args.device)

    # 2. Build model
    from flax import nnx

    from nanochat.config import ModelConfig
    from nanochat.model.transformer import TransformerLM
    from nanochat.tokenizer.bpe import BPETokenizer
    from nanochat.inference.engine import InferenceEngine

    cfg = ModelConfig.for_scale(args.model_size)
    tokenizer = BPETokenizer.from_pretrained()

    if args.checkpoint == "RANDOM":
        logger.info("model.random_init", model_size=args.model_size)
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))
    else:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error("checkpoint_not_found", path=str(ckpt_path))
            sys.exit(1)
        # TODO: implement checkpoint loading when checkpoint module is available
        logger.info("model.loading_checkpoint", path=str(ckpt_path))
        model = TransformerLM(cfg, rngs=nnx.Rngs(params=42, dropout=43))
        logger.warning("checkpoint_loading_not_implemented", fallback="random_init")

    engine = InferenceEngine(model, tokenizer, cfg)
    logger.info(
        "server.starting",
        host=args.host,
        port=args.port,
        device=args.device,
        checkpoint=args.checkpoint,
    )

    # 3. Create app and serve
    from nanochat.server.app import create_app

    app = create_app(engine)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
