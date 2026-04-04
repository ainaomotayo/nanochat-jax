<div align="center">

# nanochat-jax

**Portage fidèle de l'architecture nanochat en JAX/Flax NNX avec expériences empiriques de lois d'échelle.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-a259ff)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-0.10+-34a853)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](../../LICENSE)

<br/>

[🇺🇸 English](../../README.md) &nbsp;|&nbsp; [🇨🇳 中文](README_zh.md) &nbsp;|&nbsp; [🇯🇵 日本語](README_ja.md) &nbsp;|&nbsp; [🇰🇷 한국어](README_ko.md) &nbsp;|&nbsp; 🇫🇷 Français &nbsp;|&nbsp; [🇩🇪 Deutsch](README_de.md)

<br/>

> **AI GDE TPU Sprint 2026** — Merci à Google pour les crédits TPU Research Cloud qui rendent ce projet possible.

</div>

---

## Présentation

`nanochat-jax` poursuit deux objectifs :

1. **Portage** — Reproduire fidèlement chaque détail architectural de [nanochat](https://github.com/karpathy/nanochat) en JAX + Flax NNX : RMSNorm sans paramètres, MLP relu², normalisation L2 des Q/K, softcap sur les logits, embeddings de valeur, scalaires par couche, mélange de tokens Smear/Backout, optimiseur Muon avec Newton-Schulz cubique, et packing aligné BOS.

2. **Expériences de mise à l'échelle** — Instrumenter ce modèle pour des expériences de type Chinchilla `scale_n / scale_d / scale_c`, ajustement de lois de puissance :

```
L = 3.29 × N^(−0.027)   [TinyShakespeare, niveau caractère, 600 étapes]
```

---

## Caractéristiques architecturales

| Composant | Description |
|-----------|-------------|
| **RMSNorm sans paramètres** | `y = x / √(mean(x²) + ε)`, aucun γ/β appris |
| **MLP relu²** | `h = x · relu(x)`, 2 projections, `d_ff = 4 × d_model` |
| **Normalisation L2 Q/K** | Vecteurs Q/K normalisés L2, puis × `1.2/√d_head` |
| **Logit Softcap** | `30 · tanh(logits/30)` — limite les scores d'attention |
| **Embeddings de valeur** | Biais résiduel partagé par token, `init=1e-4` |
| **Scalaires par couche** | Scalaire appris sur sorties attention/FFN, init = 1 |
| **Smear/Backout** | Mélange causal de tokens avant/après attention, gate sigmoid init -10 |
| **Optimiseur Muon** | Orthogonalisation NS cubique (norme F), 10 étapes, momentum Nesterov |

> **[Diagramme D3.js interactif →](../architecture.html)**

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

## Démarrage rapide

```bash
# Test de fumée avec données synthétiques (aucun dataset requis)
python scripts/train.py --model-size nano --use-synthetic --total-steps 50

# Entraînement sur TinyShakespeare
curl -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python scripts/preprocess_shakespeare.py
python scripts/train.py --model-size nano --data-path data/shakespeare_char.h5 \
  --total-steps 2000 --learning-rate 3e-3
```

---

## Configurations de modèle

| Preset | d\_model | Couches | Têtes Q | Têtes KV | seq\_len | Paramètres |
|--------|--------:|--------:|--------:|---------:|---------:|-----------:|
| `nano` | 128 | 4 | 4 | 4 | 64 | ~886K |
| `small` | 512 | 6 | 8 | 8 | 2048 | ~68M |
| `medium` | 1024 | 12 | 16 | 8 GQA | 2048 | ~237M |
| `large` | 2048 | 24 | 32 | 8 GQA | 4096 | ~1.25B |
| `xlarge` | 4096 | 32 | 32 | 8 GQA | 4096 | ~6.03B |

---

## Résultats mesurés (TinyShakespeare, niveau caractère)

| Modèle | Paramètres | Perte val. | Tokens/sec |
|-------|----------:|----------:|-----------:|
| micro | 161K | 2.390 | 37 632 |
| nano | 1,01M | 2.243 | 26 028 |
| small | 5,10M | 2.179 | 17 118 |

**`L = 3,29 × N^(−0,027)`** (600 étapes ; α → 0,07–0,12 avec entraînement complet)

---

## Tests

```bash
python -m pytest              # 180 tests au total
python -m pytest tests/unit/  # tests unitaires uniquement
```

---

## Remerciements

Ce projet fait partie du **AI GDE TPU Sprint 2026**.  
Merci à **Google** pour les crédits [TPU Research Cloud](https://sites.research.google/trc/about/).

Basé sur : [JAX](https://github.com/google/jax) · [Flax NNX](https://github.com/google/flax) · [Optax](https://github.com/google-deepmind/optax)  
Architecture de référence : [nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy)

---

## Licence

[MIT](../../LICENSE)
