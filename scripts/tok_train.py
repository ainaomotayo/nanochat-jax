#!/usr/bin/env python3
"""Train a BPE tokenizer from text files using HuggingFace tokenizers.

Usage:
    python scripts/tok_train.py --input_files "data/*.txt" --vocab_size 8000 --output_dir data/tokenizer
    python scripts/tok_train.py --input_text "Hello world. This is a test." --vocab_size 256 --output_dir data/tokenizer
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# Ensure project src is importable when run from repo root
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import structlog

logger = structlog.get_logger()


def train_tokenizer(
    input_files: list[str],
    input_text: str | None,
    vocab_size: int,
    output_dir: Path,
) -> Path:
    """Train a BPE tokenizer and save it to output_dir.

    Args:
        input_files: List of resolved text file paths to train on.
        input_text: Direct text string to train on (alternative to files).
        vocab_size: Target vocabulary size.
        output_dir: Directory to save the trained tokenizer.

    Returns:
        Path to the saved tokenizer JSON file.
    """
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError:
        print(
            "Error: The 'tokenizers' library is required for BPE training.\n"
            "Install it with:\n"
            "  pip install tokenizers\n"
            "Or:\n"
            "  pip install 'nanochat-jax[tokenizers]'",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    special_tokens = [
        "<|unk|>",
        "<|bos|>",
        "<|eos|>",
        "<|pad|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "<|end|>",
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    if input_text is not None:
        # Train from an in-memory text string via iterator
        logger.info("training_from_text", text_len=len(input_text))
        tokenizer.train_from_iterator([input_text], trainer=trainer)
    elif input_files:
        logger.info("training_from_files", n_files=len(input_files))
        tokenizer.train(input_files, trainer=trainer)
    else:
        print("Error: No input provided. Use --input_files or --input_text.", file=sys.stderr)
        sys.exit(1)

    # Save tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Write a metadata sidecar for compatibility
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in special_tokens},
        "type": "bpe",
        "library": "tokenizers",
    }
    meta_path = output_dir / "tokenizer_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        "tokenizer_saved",
        path=str(tokenizer_path),
        vocab_size=tokenizer.get_vocab_size(),
    )

    print(f"\nTokenizer trained successfully.")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")
    print(f"  Saved to:   {tokenizer_path}")
    print(f"  Metadata:   {meta_path}")

    return tokenizer_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer from text files or inline text."
    )
    parser.add_argument(
        "--input_files",
        type=str,
        default=None,
        help="Glob pattern for input text files (e.g. 'data/*.txt').",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default=None,
        help="Direct text string to train on (for quick tests).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="Target vocabulary size (default: 8000).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/tokenizer",
        help="Directory to save the trained tokenizer (default: data/tokenizer).",
    )
    args = parser.parse_args()

    resolved_files: list[str] = []
    if args.input_files is not None:
        resolved_files = sorted(glob.glob(args.input_files))
        if not resolved_files:
            print(f"Warning: no files matched pattern '{args.input_files}'.", file=sys.stderr)

    if not resolved_files and args.input_text is None:
        parser.error("At least one of --input_files or --input_text is required.")

    train_tokenizer(
        input_files=resolved_files,
        input_text=args.input_text,
        vocab_size=args.vocab_size,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
