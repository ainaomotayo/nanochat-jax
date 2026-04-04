#!/usr/bin/env python3
"""Verify all nanochat-jax gap implementations (G1-G14).

Attempts to import each module and key class, printing a status table.
Exit code 0 if all pass, 1 if any fail.

Usage:
    python scripts/verify_all_gaps.py
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Each gap: (label, module_path, attribute_names_to_check)
GAPS: list[tuple[str, str, list[str]]] = [
    # G1: Model architecture
    ("G1  Model (TransformerLM)", "nanochat.model.transformer", ["TransformerLM"]),
    # G2: Attention
    ("G2  Attention", "nanochat.model.attention", ["MultiHeadAttention"]),
    # G3: Feed-forward
    ("G3  FFN", "nanochat.model.feedforward", ["FeedForward"]),
    # G4: Normalization
    ("G4  Norms", "nanochat.model.norms", ["RMSNorm"]),
    # G5: Embeddings + value embeddings
    ("G5  Embeddings", "nanochat.model.embeddings", ["TokenEmbedding"]),
    # G6: Token mixing (smear/backout)
    ("G6  Token Mixing", "nanochat.model.token_mixing", ["SmearBackout"]),
    # G7: Loss function
    ("G7  Loss", "nanochat.training.loss", ["cross_entropy_loss"]),
    # G8: Optimizer (Muon + AdamW)
    ("G8  Optimizer", "nanochat.training.optimizer", ["build_optimizer"]),
    # G9: LR Scheduler
    ("G9  Scheduler", "nanochat.training.scheduler", ["build_schedule"]),
    # G10: Data pipeline
    ("G10 Data Pipeline", "nanochat.data", ["TokenDataset", "build_dataloader", "preprocess_and_tokenize"]),
    # G11: Config system
    ("G11 Config", "nanochat.config", ["ModelConfig", "TrainingConfig"]),
    # G12: Scaling runner
    ("G12 Scaling Runner", "nanochat.scaling.runner", ["ScalingRunner", "ScalingRunResult"]),
    # G13: Scaling analysis
    ("G13 Scaling Analysis", "nanochat.scaling.analysis", ["fit_power_law", "chinchilla_optimal"]),
    # G14: Visualization + Report
    ("G14 Visualization", "nanochat.scaling.visualization", ["generate_full_report"]),
    # Additional modules
    ("     Auto-Scale", "nanochat.config.auto_scale", ["model_config_from_depth", "training_config_from_depth"]),
    ("     Report Gen", "nanochat.scaling.report", ["generate_report", "generate_comparison_table"]),
    ("     Dataset Registry", "nanochat.data.registry", ["DATASET_REGISTRY", "get_dataset", "DatasetSpec"]),
    ("     Device", "nanochat.core.device", ["setup_device", "DeviceType"]),
    ("     Tokenizer (BPE)", "nanochat.tokenizer.bpe", ["BPETokenizer"]),
    ("     Tokenizer (Char)", "nanochat.tokenizer.char", ["CharTokenizer"]),
    ("     Checkpointing", "nanochat.training.checkpoint", ["save_checkpoint", "load_checkpoint"]),
    ("     KV Cache", "nanochat.model.kv_cache", ["KVCache"]),
    ("     Param Count", "nanochat.model.param_count", ["count_params"]),
    ("     Sampling", "nanochat.inference.sampling", ["sample"]),
]


def verify_gap(label: str, module_path: str, attrs: list[str]) -> tuple[bool, str]:
    """Try importing the module and checking for the expected attributes.

    Returns:
        (passed, detail_message)
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        return False, f"Import failed: {type(exc).__name__}: {exc}"

    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        return False, f"Missing attributes: {', '.join(missing)}"

    return True, "OK"


def main() -> int:
    width_label = max(len(g[0]) for g in GAPS) + 2
    width_module = max(len(g[1]) for g in GAPS) + 2

    header = f"{'Gap':<{width_label}} {'Module':<{width_module}} Status   Detail"
    sep = "-" * len(header)

    print("\n" + sep)
    print("  nanochat-jax Gap Verification")
    print(sep)
    print(header)
    print(sep)

    n_pass = 0
    n_fail = 0

    for label, module_path, attrs in GAPS:
        passed, detail = verify_gap(label, module_path, attrs)
        status = "PASS" if passed else "FAIL"
        symbol = "[+]" if passed else "[-]"
        if passed:
            n_pass += 1
        else:
            n_fail += 1
        print(f"{label:<{width_label}} {module_path:<{width_module}} {symbol} {status:<6}  {detail}")

    print(sep)
    total = n_pass + n_fail
    print(f"\nTotal: {n_pass}/{total} passed, {n_fail}/{total} failed")

    if n_fail > 0:
        print("\nSome gaps are not fully implemented.")
        return 1
    else:
        print("\nAll gaps verified successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
