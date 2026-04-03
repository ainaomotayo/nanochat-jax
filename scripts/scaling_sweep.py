#!/usr/bin/env python3
"""Quick 3-point scaling sweep: vary model size at fixed token budget.

Runs 3 model sizes (micro/nano/small-context) for fixed token budget on GPU.
Fits a power law L = a * N^(-alpha) and prints the scaling exponent.

Usage:
    python scripts/scaling_sweep.py
    python scripts/scaling_sweep.py --steps 300 --batch 8
"""
from __future__ import annotations
import os, sys, time, json, math
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

from nanochat.config import ModelConfig, TrainingConfig
from nanochat.model.transformer import TransformerLM
from nanochat.model.param_count import count_params
from nanochat.training.optimizer import build_optimizer
from nanochat.training.loss import cross_entropy_loss
from nanochat.scaling.analysis import fit_power_law

import structlog
log = structlog.get_logger()


# ── 3 model sizes, all fit in 4 GB ──────────────────────────────────────────
SCALES = {
    "micro": dict(vocab_size=256, d_model=64,  n_layers=2, n_heads=2, n_kv_heads=2, max_seq_len=64),
    "nano":  dict(vocab_size=256, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4, max_seq_len=64),
    "small": dict(vocab_size=256, d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, max_seq_len=64),
}


def train_one(name: str, model_cfg: ModelConfig, steps: int, batch: int) -> dict:
    log.info("run_start", name=name, d_model=model_cfg.d_model, n_layers=model_cfg.n_layers)
    rngs = nnx.Rngs(params=42, dropout=43)
    model = TransformerLM(model_cfg, rngs=rngs)
    n_params = count_params(model)["total"]

    train_cfg = TrainingConfig(
        batch_size=batch,
        total_steps=steps,
        warmup_steps=steps // 10,
        learning_rate=3e-3,
        muon_weight_decay=0.0,
        weight_decay=0.0,
    )
    tx = build_optimizer(train_cfg)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def step_fn(model, opt, ids):
        def loss_fn(m):
            logits, _ = m(ids, deterministic=False)
            loss, _ = cross_entropy_loss(logits[:, :-1, :], ids[:, 1:])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    rng = jax.random.PRNGKey(0)
    losses = []
    t0 = time.time()
    for s in range(steps):
        rng, k = jax.random.split(rng)
        ids = jax.random.randint(k, (batch, model_cfg.max_seq_len), 0, model_cfg.vocab_size)
        loss = step_fn(model, opt, ids)
        if s % max(steps // 10, 1) == 0:
            lv = float(loss)
            losses.append((s, lv))
            log.info("step", name=name, step=s, loss=round(lv, 4))

    final_loss = float(loss)
    wall = time.time() - t0
    tok_per_sec = (steps * batch * model_cfg.max_seq_len) / wall

    log.info("run_done", name=name, n_params=n_params,
             final_loss=round(final_loss, 4), tok_per_sec=int(tok_per_sec),
             wall_s=round(wall, 1))
    return {
        "name": name,
        "n_params": n_params,
        "final_loss": final_loss,
        "tok_per_sec": tok_per_sec,
        "losses": losses,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", type=str, default="outputs/scaling_sweep.json")
    args = parser.parse_args()

    log.info("scaling_sweep_start", steps=args.steps, batch=args.batch,
             devices=str(jax.devices()))

    results = []
    for name, spec in SCALES.items():
        cfg = ModelConfig(**spec)
        r = train_one(name, cfg, args.steps, args.batch)
        results.append(r)

    # ── Power law fit: L = a * N^(-alpha) ───────────────────────────────────
    ns = np.array([r["n_params"] for r in results], dtype=float)
    ls = np.array([r["final_loss"] for r in results], dtype=float)

    fit = fit_power_law(ns, ls)
    alpha = fit["alpha"]
    a     = fit["a"]

    print("\n" + "="*60)
    print("  SCALING SWEEP RESULTS")
    print("="*60)
    for r in results:
        print(f"  {r['name']:8s}  N={r['n_params']:>8,}  loss={r['final_loss']:.4f}"
              f"  tok/s={int(r['tok_per_sec']):>7,}")
    print(f"\n  Power law fit: L = {a:.4f} × N^(-{alpha:.4f})")
    print(f"  Scaling exponent α = {alpha:.4f}")
    if abs(alpha) > 0.05:
        print(f"  → Loss halves every {math.log(2)/alpha:.1f}× increase in params")
    print("="*60 + "\n")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "fit": fit}, f, indent=2, default=str)
    log.info("saved", path=args.out)


if __name__ == "__main__":
    main()
