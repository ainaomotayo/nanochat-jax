#!/usr/bin/env python3
"""Universal preprocessing script for nanochat-jax.

Downloads datasets from HuggingFace (or uses local files), tokenizes them,
and writes flat HDF5 token arrays ready for training.

Usage:
    python scripts/preprocess.py --dataset shakespeare_char --output_dir data/
    python scripts/preprocess.py --dataset tinystories --output_dir data/ --max_samples 10000
    python scripts/preprocess.py --dataset openwebtext --output_dir data/ --tokenizer bpe
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import h5py
import numpy as np
import structlog

from nanochat.data.registry import get_dataset, DATASET_REGISTRY

logger = structlog.get_logger()


def _build_tokenizer(tokenizer_type: str, text_for_char: str | None = None):
    """Construct the appropriate tokenizer.

    Args:
        tokenizer_type: Either "char" or "bpe".
        text_for_char: Full text corpus needed to build char vocabulary.

    Returns:
        A BaseTokenizer instance.
    """
    if tokenizer_type == "char":
        from nanochat.tokenizer.char import CharTokenizer
        if text_for_char is None:
            raise ValueError("Character tokenizer requires text to build vocabulary.")
        return CharTokenizer.from_text(text_for_char)
    else:
        from nanochat.tokenizer.bpe import BPETokenizer
        return BPETokenizer.from_pretrained("cl100k_base")


def _preprocess_shakespeare(output_dir: Path, max_samples: int | None) -> dict:
    """Handle shakespeare_char using existing CharTokenizer pipeline."""
    spec = get_dataset("shakespeare_char")

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: 'datasets' library required. Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    ds = load_dataset(spec.hf_path, split=spec.split_map["train"], trust_remote_code=True)
    all_text = "\n".join(row[spec.text_column] for row in ds)
    if max_samples is not None:
        all_text = all_text[:max_samples * 100]

    tokenizer = _build_tokenizer("char", text_for_char=all_text)

    output_path = output_dir / "shakespeare_char.h5"
    if output_path.exists():
        logger.info("dataset_exists", path=str(output_path))
        with h5py.File(output_path, "r") as f:
            return {"n_tokens": len(f["tokens"]), "output_path": str(output_path)}

    from nanochat.data.preprocessing import preprocess_and_tokenize

    # Write text to a temp file for the preprocessing pipeline
    tmp_text_path = output_dir / "shakespeare_raw.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_text_path.write_text(all_text)

    result = preprocess_and_tokenize(
        source=str(tmp_text_path),
        tokenizer=tokenizer,
        output_path=output_path,
    )
    # Save vocab alongside
    vocab_path = output_dir / "shakespeare_vocab.json"
    tokenizer.save(str(vocab_path))
    logger.info("vocab_saved", path=str(vocab_path), vocab_size=tokenizer.vocab_size)

    return result


def _preprocess_generic(
    dataset_name: str,
    output_dir: Path,
    tokenizer_type: str,
    max_samples: int | None,
) -> dict:
    """Preprocess a generic HuggingFace dataset."""
    spec = get_dataset(dataset_name)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Error: 'datasets' library required. Install with: pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    split = spec.split_map.get("train", "train")
    logger.info("downloading_dataset", name=spec.name, hf_path=spec.hf_path, split=split)

    load_kwargs: dict = {"path": spec.hf_path, "split": split, "trust_remote_code": True}
    if spec.hf_name is not None:
        load_kwargs["name"] = spec.hf_name

    ds = load_dataset(**load_kwargs)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    logger.info("dataset_loaded", n_samples=len(ds))

    # Build tokenizer
    effective_tokenizer_type = tokenizer_type or spec.tokenizer
    if effective_tokenizer_type == "char":
        sample_text = "\n".join(
            str(row[spec.text_column]) for row in ds.select(range(min(10000, len(ds))))
        )
        tokenizer = _build_tokenizer("char", text_for_char=sample_text)
    else:
        tokenizer = _build_tokenizer("bpe")

    # Tokenize all documents
    output_path = output_dir / f"{spec.name}.h5"
    if output_path.exists():
        logger.info("dataset_exists", path=str(output_path))
        with h5py.File(output_path, "r") as f:
            return {"n_tokens": len(f["tokens"]), "output_path": str(output_path)}

    all_tokens: list[int] = []
    n_docs = 0
    eos_id = tokenizer.eos_id

    for row in ds:
        text = row[spec.text_column]
        # Handle message-format datasets (list of dicts)
        if isinstance(text, list):
            text = " ".join(
                msg.get("content", str(msg)) if isinstance(msg, dict) else str(msg)
                for msg in text
            )
        text = str(text).strip()
        if not text:
            continue
        ids = tokenizer.encode(text)
        ids.append(eos_id)
        all_tokens.extend(ids)
        n_docs += 1
        if n_docs % 10000 == 0:
            logger.info("tokenizing_progress", docs=n_docs, tokens=len(all_tokens))

    token_array = np.array(all_tokens, dtype=np.int32)
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tokens", data=token_array, chunks=True, compression="gzip")
        f.attrs["n_tokens"] = len(token_array)
        f.attrs["n_documents"] = n_docs
        f.attrs["source"] = spec.hf_path
        f.attrs["vocab_size"] = tokenizer.vocab_size
        f.attrs["dataset_name"] = spec.name

    logger.info(
        "preprocessing_complete",
        dataset=spec.name,
        n_tokens=len(token_array),
        n_docs=n_docs,
        output=str(output_path),
    )
    return {"n_tokens": len(token_array), "n_documents": n_docs, "output_path": str(output_path)}


def main() -> None:
    available = ", ".join(sorted(DATASET_REGISTRY.keys()))
    parser = argparse.ArgumentParser(
        description="Universal dataset preprocessing for nanochat-jax."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Dataset name. Available: {available}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/",
        help="Output directory for HDF5 files (default: data/).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        choices=["char", "bpe"],
        default=None,
        help="Tokenizer type. If not set, uses the dataset's default.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.dataset.lower() == "shakespeare_char":
        result = _preprocess_shakespeare(output_dir, args.max_samples)
    else:
        result = _preprocess_generic(
            args.dataset,
            output_dir,
            args.tokenizer,
            args.max_samples,
        )

    print(f"\nPreprocessing complete.")
    print(f"  Tokens:  {result['n_tokens']:,}")
    print(f"  Output:  {result['output_path']}")


if __name__ == "__main__":
    main()
