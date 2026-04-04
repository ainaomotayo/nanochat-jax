#!/usr/bin/env python3
"""Preprocess TinyShakespeare text → HDF5 token file using char-level tokenizer.

Usage:
    python scripts/preprocess_shakespeare.py
    python scripts/preprocess_shakespeare.py --text data/tinyshakespeare.txt \
        --output data/shakespeare_char.h5 --vocab data/shakespeare_vocab.json
"""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import structlog
from nanochat.tokenizer.char import CharTokenizer
from nanochat.data.preprocessing import preprocess_and_tokenize

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess TinyShakespeare → HDF5")
    parser.add_argument("--text", default="data/tinyshakespeare.txt")
    parser.add_argument("--output", default="data/shakespeare_char.h5")
    parser.add_argument("--vocab", default="data/shakespeare_vocab.json")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    text_path = Path(args.text)
    if not text_path.exists():
        raise FileNotFoundError(
            f"TinyShakespeare not found at {text_path}. "
            "Download with: curl -o data/tinyshakespeare.txt "
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    # Remove old output so preprocessing.py re-runs
    if args.force and Path(args.output).exists():
        Path(args.output).unlink()

    # Build vocabulary from full text
    log.info("building_vocab", text=args.text)
    text = text_path.read_text()
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(args.vocab)
    log.info("vocab_saved",
             vocab_size=tokenizer.vocab_size,
             path=args.vocab)

    # Tokenize and write HDF5
    result = preprocess_and_tokenize(
        source=args.text,
        tokenizer=tokenizer,
        output_path=args.output,
    )

    log.info("done",
             n_tokens=f"{result['n_tokens']:,}",
             output=result["output_path"],
             vocab_size=tokenizer.vocab_size)

    print(f"\nDone. Saved {result['n_tokens']:,} tokens to {result['output_path']}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Vocab saved to: {args.vocab}")
    print(f"\nTo train on this data:")
    print(f"  python scripts/train.py --data-path {args.output} "
          f"--model-size nano --total-steps 2000 --no-use-synthetic")


if __name__ == "__main__":
    main()
