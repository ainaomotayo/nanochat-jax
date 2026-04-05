<div align="center">

# nanochat-jax

**Originalgetreuer Port der nanochat-Architektur in JAX/Flax NNX mit empirischen Skalierungsgesetz-Experimenten.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-a259ff)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-0.12+-34a853)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](../../LICENSE)

<br/>

[🇺🇸 English](../../README.md) &nbsp;|&nbsp; [🇨🇳 中文](README_zh.md) &nbsp;|&nbsp; [🇯🇵 日本語](README_ja.md) &nbsp;|&nbsp; [🇰🇷 한국어](README_ko.md) &nbsp;|&nbsp; [🇫🇷 Français](README_fr.md) &nbsp;|&nbsp; 🇩🇪 Deutsch

<br/>

> **AI GDE TPU Sprint 2026** — Vielen Dank an Google für die TPU Research Cloud Credits, die dieses Projekt ermöglichen.

</div>

---

## Überblick

`nanochat-jax` verfolgt zwei Ziele:

1. **Port** — Vollständige Reproduktion aller Architekturdetails von [nanochat](https://github.com/karpathy/nanochat) in JAX + Flax NNX: parameterfreie RMSNorm, relu²-MLP, QK-L2-Normalisierung, Logit-Softcap, Value-Embeddings, schichtweise Skalare, Smear/Backout-Token-Mixing, Muon-Optimierer mit kubischem Newton-Schulz, BOS-ausgerichtetes Packing.

2. **Skalierungsexperimente** — Durchführung von Chinchilla-artigen `scale_n / scale_d / scale_c`-Experimenten und Potenzgesetz-Fitting:

```
L = 3.29 × N^(−0.027)   [TinyShakespeare, Zeichenebene, 600 Schritte]
```

---

## Architekturmerkmale

| Komponente | Beschreibung |
|-----------|--------------|
| **Parameterfreie RMSNorm** | `y = x / √(mean(x²) + ε)`, keine gelernten γ/β |
| **relu²-MLP** | `h = x · relu(x)`, 2 Projektionen, `d_ff = 4 × d_model` |
| **QK-L2-Normalisierung** | Q/K-Vektoren L2-normalisiert, dann × `1.2/√d_head` |
| **Logit Softcap** | `30 · tanh(logits/30)` — begrenzt Attention-Scores |
| **Value-Embeddings** | Schichtübergreifender Residual-Bias pro Token, `init=1e-4` |
| **Schichtweise Skalare** | Lernbarer Skalar auf Attention/FFN-Ausgaben, init = 1 |
| **Smear/Backout** | Kausales Token-Mixing vor/nach Attention, Sigmoid-Gate init -10 |
| **Muon-Optimierer** | Kubische NS-Orthogonalisierung (F-Norm), 10 Schritte, Nesterov-Momentum |

> **[Interaktives D3.js-Diagramm →](../architecture.html)**

---

## Installation

```bash
git clone https://github.com/ainaomotayo/nanochat-jax
cd nanochat-jax
pip install -e ".[dev]"

# GPU (CUDA 12)
pip install -U "jax[cuda12]"

# TPU
pip install -U "jax[tpu]" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

---

## Schnellstart

```bash
# Smoke-Test mit synthetischen Daten (kein Datensatz erforderlich)
python scripts/train.py --model-size nano --use-synthetic --total-steps 50

# Training auf TinyShakespeare
curl -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python scripts/preprocess_shakespeare.py
python scripts/train.py --model-size nano --data-path data/shakespeare_char.h5 \
  --total-steps 2000 --learning-rate 3e-3
```

---

## Modellkonfigurationen

| Preset | d\_model | Schichten | Q-Köpfe | KV-Köpfe | seq\_len | Parameter |
|--------|--------:|---------:|--------:|---------:|---------:|----------:|
| `nano` | 128 | 4 | 4 | 4 | 64 | ~886K |
| `small` | 512 | 6 | 8 | 8 | 2048 | ~68M |
| `medium` | 1024 | 12 | 16 | 8 GQA | 2048 | ~237M |
| `large` | 2048 | 24 | 32 | 8 GQA | 4096 | ~1.25B |
| `xlarge` | 4096 | 32 | 32 | 8 GQA | 4096 | ~6.03B |

---

## Gemessene Ergebnisse (TinyShakespeare, Zeichenebene)

| Modell | Parameter | Val.-Verlust | Token/Sek. |
|-------|----------:|-------------:|-----------:|
| micro | 161K | 2.390 | 37.632 |
| nano | 1,01M | 2.243 | 26.028 |
| small | 5,10M | 2.179 | 17.118 |

**`L = 3,29 × N^(−0,027)`** (600 Schritte; α → 0,07–0,12 bei vollständigem Training)

---

## Tests

```bash
python -m pytest              # 180 Tests insgesamt
python -m pytest tests/unit/  # Nur Unit-Tests
```

---

## Danksagung

Dieses Projekt ist Teil des **AI GDE TPU Sprint 2026**.  
Vielen Dank an **Google** für die [TPU Research Cloud](https://sites.research.google/trc/about/) Credits.

Basierend auf: [JAX](https://github.com/google/jax) · [Flax NNX](https://github.com/google/flax) · [Optax](https://github.com/google-deepmind/optax)  
Referenzarchitektur: [nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy)

---

## Lizenz

[MIT](../../LICENSE)
