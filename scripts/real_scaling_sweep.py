#!/usr/bin/env python3
"""Scaling sweep on real text (TinyShakespeare) with char-level tokenization.

Downloads/loads TinyShakespeare, tokenizes char-level (vocab=67),
trains 3 model sizes for a fixed token budget, fits L = a * N^(-alpha).

Usage:
    python scripts/real_scaling_sweep.py
    python scripts/real_scaling_sweep.py --steps 600 --batch 32
"""
from __future__ import annotations
import os, sys, time, json, math
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params
from nanochat.training.optimizer import build_optimizer
from nanochat.training.loss import cross_entropy_loss
from nanochat.scaling.analysis import fit_power_law

import structlog
log = structlog.get_logger()

DATA_PATH = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


# ── Minimal char-level tokenizer ─────────────────────────────────────────────

def build_char_vocab(text: str):
    chars = sorted(set(text))
    # Reserve 0=PAD, 1=BOS, 2=EOS
    ch2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    for c in chars:
        ch2id[c] = len(ch2id)
    id2ch = {v: k for k, v in ch2id.items()}
    return ch2id, id2ch


def tokenize(text: str, ch2id: dict, eos_id: int = 2) -> np.ndarray:
    """Encode text → token array with EOS at end."""
    ids = [ch2id.get(c, 0) for c in text]
    ids.append(eos_id)
    return np.array(ids, dtype=np.int32)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(seq_len: int, batch: int):
    """Load TinyShakespeare and build train/val numpy arrays."""
    text = DATA_PATH.read_text()
    ch2id, _ = build_char_vocab(text)
    vocab_size = len(ch2id)

    tokens = tokenize(text, ch2id)
    n = len(tokens)
    split = int(0.9 * n)
    train_tok = tokens[:split]
    val_tok   = tokens[split:]

    log.info("data_ready", vocab_size=vocab_size,
             train_tokens=len(train_tok), val_tokens=len(val_tok))
    return train_tok, val_tok, vocab_size


def make_batch(tokens: np.ndarray, seq_len: int, batch: int, rng: np.random.Generator):
    """Sample a random batch from token array. Returns (input_ids, labels)."""
    max_start = len(tokens) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch)
    inp  = np.stack([tokens[s     : s + seq_len    ] for s in starts])
    lbl  = np.stack([tokens[s + 1 : s + seq_len + 1] for s in starts])
    return jnp.array(inp), jnp.array(lbl)


# ── Model sizes (all fit comfortably on 4 GB) ─────────────────────────────────

def get_scales(vocab_size: int):
    return {
        "micro": ModelConfig(vocab_size=vocab_size, d_model=64,  n_layers=3,
                             n_heads=2, n_kv_heads=2, max_seq_len=128),
        "nano":  ModelConfig(vocab_size=vocab_size, d_model=128, n_layers=5,
                             n_heads=4, n_kv_heads=4, max_seq_len=128),
        "small": ModelConfig(vocab_size=vocab_size, d_model=256, n_layers=7,
                             n_heads=8, n_kv_heads=4, max_seq_len=128),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one(name: str, cfg: ModelConfig, train_tok: np.ndarray, val_tok: np.ndarray,
              steps: int, batch: int) -> dict:
    log.info("run_start", name=name, d_model=cfg.d_model, n_layers=cfg.n_layers,
             vocab_size=cfg.vocab_size)

    rngs = nnx.Rngs(params=42, dropout=43)
    model = TransformerLM(cfg, rngs=rngs)
    n_params = count_params(model)["total"]

    train_cfg = TrainingConfig(
        batch_size=batch, total_steps=steps,
        warmup_steps=steps // 10, learning_rate=3e-3,
        muon_weight_decay=0.0, weight_decay=0.0,
    )
    tx = build_optimizer(train_cfg)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def step_fn(model, opt, inp, lbl):
        def loss_fn(m):
            logits, _ = m(inp, deterministic=False)
            # labels are pre-shifted: lbl[t] = target for logit[t]
            loss, _ = cross_entropy_loss(logits[:, :-1, :], lbl[:, :-1])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    rng = np.random.default_rng(0)
    losses, val_losses = [], []
    t0 = time.time()

    for s in range(steps):
        inp, lbl = make_batch(train_tok, cfg.max_seq_len, batch, rng)
        loss = step_fn(model, opt, inp, lbl)

        if s % max(steps // 10, 1) == 0:
            lv = float(loss)
            losses.append((s, lv))
            log.info("step", name=name, step=s, loss=round(lv, 4))

    # Final validation loss
    val_inp, val_lbl = make_batch(val_tok, cfg.max_seq_len, batch * 4, rng)
    val_logits, _ = model(val_inp, deterministic=True)
    val_loss_val, _ = cross_entropy_loss(val_logits[:, :-1, :], val_lbl[:, :-1])
    val_loss = float(val_loss_val)

    wall = time.time() - t0
    tok_per_sec = (steps * batch * cfg.max_seq_len) / wall

    log.info("run_done", name=name, n_params=n_params,
             train_loss=round(float(loss), 4), val_loss=round(val_loss, 4),
             tok_per_sec=int(tok_per_sec), wall_s=round(wall, 1))

    return {
        "name": name, "n_params": n_params,
        "final_train_loss": float(loss), "final_val_loss": val_loss,
        "tok_per_sec": tok_per_sec, "losses": losses,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out",   type=str, default="outputs/real_scaling.json")
    args = parser.parse_args()

    log.info("real_scaling_sweep_start", steps=args.steps, batch=args.batch,
             devices=str(jax.devices()))

    train_tok, val_tok, vocab_size = prepare_data(seq_len=128, batch=args.batch)
    scales = get_scales(vocab_size)

    results = []
    for name, cfg in scales.items():
        r = train_one(name, cfg, train_tok, val_tok, args.steps, args.batch)
        results.append(r)

    # ── Power law fit on val loss ─────────────────────────────────────────────
    ns = np.array([r["n_params"]       for r in results], dtype=float)
    ls = np.array([r["final_val_loss"] for r in results], dtype=float)

    fit = fit_power_law(ns, ls)
    alpha = fit["alpha"]
    a     = fit["a"]

    print("\n" + "="*65)
    print("  REAL SCALING SWEEP — TinyShakespeare (char-level)")
    print("="*65)
    print(f"  {'Model':8s}  {'Params':>10s}  {'Train loss':>11s}  "
          f"{'Val loss':>9s}  {'Tok/sec':>8s}")
    print("  " + "-"*61)
    for r in results:
        print(f"  {r['name']:8s}  {r['n_params']:>10,}  "
              f"{r['final_train_loss']:>11.4f}  "
              f"{r['final_val_loss']:>9.4f}  {int(r['tok_per_sec']):>8,}")
    print("  " + "-"*61)
    print(f"\n  Power law: L = {a:.4f} × N^(-{alpha:.4f})")
    print(f"  Scaling exponent α = {alpha:.4f}  "
          f"(Chinchilla: ~0.34, GPT-3: ~0.08)")
    if alpha > 0.01:
        print(f"  → Loss halves every {math.log(2)/alpha:.1f}× increase in params")
    print("="*65 + "\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "fit": fit,
                   "dataset": "TinyShakespeare (char-level)",
                   "steps": args.steps, "batch": args.batch}, f, indent=2)
    log.info("saved", path=args.out)


if __name__ == "__main__":
    main()
