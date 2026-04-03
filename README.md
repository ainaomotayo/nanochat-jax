# NanoChat-JAX

[![CI](https://github.com/your-org/nanochat-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/nanochat-jax/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-green.svg)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-0.10+-purple.svg)](https://github.com/google/flax)

**Enterprise-grade GPT-style transformer in JAX/Flax NNX with empirical scaling law experiments.**

Reproduces the [NanoChat](https://github.com/karpathy/nanochat) architecture (decoder-only transformer with chat interface) using idiomatic JAX + Flax NNX, then instruments it to run Chinchilla-style compute-optimal scaling experiments following [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) and [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556).

---

## What This Is

**NanoChat-JAX** is a complete, production-grade implementation of a GPT-style language model built entirely in JAX and Flax NNX. Unlike tutorial code, every component is enterprise-ready: full type annotations, structured logging, comprehensive tests, Docker support, and CI/CD pipelines.

The project serves three purposes: (1) a reference implementation of modern transformer architecture in JAX, (2) a training harness with proper distributed training support, checkpointing, and observability, and (3) a scaling law experiment framework that empirically measures the power-law relationships between model size, data, compute, and loss.

The architecture implements Grouped-Query Attention (GQA), RoPE positional encodings, SwiGLU feed-forward networks, RMSNorm, and weight-tied embeddings — matching the design choices of LLaMA, Mistral, and Gemma.

---

## Architecture Overview

```
Input IDs [B, S]
     │
     ▼
┌─────────────────────┐
│  TokenEmbedding     │  [B, S] → [B, S, D]
│  (vocab → d_model)  │
└─────────┬───────────┘
          │
          │  RoPE freqs precomputed [S, D/(2H)]
          │
          ▼
┌─────────────────────────────────────────────────┐
│              TransformerBlock × L                │
│                                                  │
│  x ──► RMSNorm ──► GQA Attention ──► (+) ──► x' │
│                         │              ▲         │
│                    RoPE, Causal        │         │
│                    Mask, KVCache    residual     │
│                                                  │
│  x'──► RMSNorm ──► SwiGLU FFN ──► (+) ──► x''  │
│                                    ▲             │
│                                 residual         │
└─────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  Final RMSNorm      │  [B, S, D]
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LM Head            │  [B, S, D] → [B, S, V]
│  (tied to embed)    │
└─────────┬───────────┘
          │
          ▼
     Logits [B, S, V]
```

### Model Size Configurations

| Config  | d_model | Layers | Heads | KV Heads | d_ff  | Params  |
|---------|---------|--------|-------|----------|-------|---------|
| nano    | 64      | 2      | 4     | 4        | 128   | ~100K   |
| small   | 512     | 8      | 8     | 8        | auto  | ~50M    |
| medium  | 1024    | 12     | 16    | 8 (GQA)  | auto  | ~150M   |
| large   | 1536    | 24     | 16    | 8 (GQA)  | auto  | ~350M   |
| xlarge  | 2048    | 24     | 16    | 8 (GQA)  | auto  | ~700M   |

---

## Scaling Law Results

The scaling experiment framework measures empirical power-law exponents:

| Relationship | Kaplan et al. | Chinchilla | This Work |
|-------------|---------------|------------|-----------|
| L(N) ∝ N^(-α) | α = 0.076 | α ≈ 0.34 | Run `make scale` |
| L(D) ∝ D^(-β) | β = 0.095 | β ≈ 0.28 | Run `make scale` |
| L(C) ∝ C^(-γ) | γ = 0.057 | — | Run `make scale` |

Run `python scripts/run_scaling.py --experiment scale_n` to reproduce.

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/your-org/nanochat-jax.git
cd nanochat-jax

# 2. Install
pip install -e ".[dev]"

# 3. Run tests
make test

# 4. Train a tiny model (smoke test, ~2 min on CPU)
make train

# 5. Chat with it
make chat
```

---

## Training Guide

### Basic Training

```bash
# Train nano model (smoke test)
python scripts/train.py --model-size nano --total-steps 200

# Train small model with custom LR
python scripts/train.py --model-size small --learning-rate 1e-4 --total-steps 10000

# Train with real data (requires preprocessing)
python scripts/train.py --model-size small --data-path data/openwebtext/tokens.h5
```

### Data Preprocessing

```python
from nanochat.tokenizer import BPETokenizer
from nanochat.data import preprocess_and_tokenize

tokenizer = BPETokenizer.from_pretrained()
stats = preprocess_and_tokenize(
    source="openwebtext",
    tokenizer=tokenizer,
    output_path="data/openwebtext/tokens.h5",
    hf_dataset="Skylion007/openwebtext",
)
print(f"Tokenized {stats['n_tokens']:,} tokens")
```

### Checkpointing

Checkpoints are saved automatically every `save_every_steps`. Resume training:

```bash
python scripts/train.py --model-size small --checkpoint-dir checkpoints/
```

---

## Scaling Experiments

### Run Experiments

```bash
# Vary model size at fixed compute
python scripts/run_scaling.py --experiment scale_n

# Vary token budget at fixed model size
python scripts/run_scaling.py --experiment scale_d

# Quick test (fewer steps)
python scripts/run_scaling.py --experiment scale_n --max-steps 50
```

### Interpret Results

Results are saved to `outputs/scaling/` including:
- `scaling_report.md` — Summary table with fitted exponents
- `loss_vs_params.png` — Log-log plot of loss vs model size
- `loss_vs_compute.png` — Log-log plot of loss vs total FLOPs
- `training_curves.png` — Training curves for all runs

### Key References

- **Kaplan et al. (2020)** "Scaling Laws for Neural Language Models" — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **Hoffmann et al. (2022)** "Training Compute-Optimal Large Language Models" — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

---

## Project Structure

```
nanochat-jax/
├── src/nanochat/
│   ├── config/          # Pydantic config dataclasses
│   ├── model/           # RMSNorm, RoPE, GQA, SwiGLU, Transformer
│   ├── tokenizer/       # tiktoken-based BPE tokenizer
│   ├── data/            # HDF5 dataset, dataloader, preprocessing
│   ├── training/        # Loss, optimizer, scheduler, trainer, checkpointing
│   ├── inference/       # KV cache, sampling, inference engine, chat
│   ├── evaluation/      # Metrics, evaluator, throughput benchmarks
│   └── scaling/         # Experiment runner, power law fitting, visualization
├── scripts/             # CLI entry points (train, chat, evaluate, run_scaling)
├── tests/               # Unit, integration, and scaling tests
├── configs/             # Hydra YAML configs for models, training, data
├── docker/              # Dockerfile (CPU + GPU) and docker-compose
├── monitoring/          # Prometheus + Grafana configs
└── .github/workflows/   # CI/CD pipelines
```

---

## Contributing

1. Install dev dependencies: `pip install -e ".[dev]"`
2. Run `make lint` and `make typecheck` before committing
3. All new code must have tests (target: >90% coverage)
4. PRs require passing CI (lint + typecheck + tests)

### Code Style

- Line length: 100 characters
- Type annotations on all functions
- Google-style docstrings
- Structured logging via `structlog` (no `print()`)
- Shape annotations as comments on tensor operations

---

## Citation

```bibtex
@article{kaplan2020scaling,
  title={Scaling Laws for Neural Language Models},
  author={Kaplan, Jared and McCandlish, Sam and Henighan, Tom and Brown, Tom B and Chess, Benjamin and Child, Rewon and Gray, Scott and Radford, Alec and Wu, Jeffrey and Amodei, Dario},
  journal={arXiv preprint arXiv:2001.08361},
  year={2020}
}

@article{hoffmann2022training,
  title={Training Compute-Optimal Large Language Models},
  author={Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and Casas, Diego de Las and Hendricks, Lisa Anne and Welbl, Johannes and Clark, Aidan and others},
  journal={arXiv preprint arXiv:2203.15556},
  year={2022}
}
```

---

## License

MIT
