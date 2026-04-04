<div align="center">

# nanochat-jax

**JAX/Flax NNX による nanochat アーキテクチャの忠実な移植と、経験的スケーリング則実験フレームワーク。**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-a259ff)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-0.10+-34a853)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](../../LICENSE)

<br/>

[🇺🇸 English](../../README.md) &nbsp;|&nbsp; [🇨🇳 中文](README_zh.md) &nbsp;|&nbsp; 🇯🇵 日本語 &nbsp;|&nbsp; [🇰🇷 한국어](README_ko.md) &nbsp;|&nbsp; [🇫🇷 Français](README_fr.md) &nbsp;|&nbsp; [🇩🇪 Deutsch](README_de.md)

<br/>

> **AI GDE TPU Sprint 2026** — Google による TPU Research Cloud クレジット提供に感謝します。

</div>

---

## 概要

`nanochat-jax` には 2 つの目的があります：

1. **移植** — [nanochat](https://github.com/karpathy/nanochat) のすべてのアーキテクチャ詳細を JAX + Flax NNX で完全再現します。パラメータなし RMSNorm、relu² MLP、QK L2 正規化、logit softcap、値埋め込み、レイヤーごとのスカラー、Smear/Backout トークンミキシング、三次 Newton-Schulz を用いた Muon オプティマイザ、BOS アラインパッキング。

2. **スケーリング実験** — Chinchilla スタイルの `scale_n / scale_d / scale_c` 実験を実行し、べき乗則をフィットします：

```
L = 3.29 × N^(−0.027)   [TinyShakespeare、文字レベル、600 ステップ]
```

---

## アーキテクチャの特徴

| コンポーネント | 説明 |
|--------------|------|
| **パラメータなし RMSNorm** | `y = x / √(mean(x²) + ε)`、学習パラメータ γ/β なし |
| **relu² MLP** | `h = x · relu(x)`、2 つの線形変換のみ、`d_ff = 4 × d_model` |
| **QK L2 正規化** | Q/K を L2 正規化後 `1.2/√d_head` でスケール |
| **Logit Softcap** | `30 · tanh(logits/30)` でアテンションスコアを制限 |
| **値埋め込み** | 全レイヤー共有の残差バイアス、`init=1e-4` |
| **レイヤースカラー** | アテンション/FFN 出力への学習可能スカラー、初期値 1 |
| **Smear/Backout** | アテンション前後の因果トークンミキシング、sigmoid ゲート初期値 -10 |
| **Muon オプティマイザ** | 三次 NS 直交化（F ノルム正規化）、10 ステップ、Nesterov 運動量 |

> **[インタラクティブ D3.js 図 →](../architecture.html)**（ブラウザで開き、各ブロックにホバーで詳細表示）

---

## インストール

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

## クイックスタート

```bash
# 合成データでのスモークテスト（データ不要）
python scripts/train.py --model-size nano --use-synthetic --total-steps 50

# TinyShakespeare での文字レベル学習
curl -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python scripts/preprocess_shakespeare.py
python scripts/train.py --model-size nano --data-path data/shakespeare_char.h5 \
  --total-steps 2000 --learning-rate 3e-3
```

---

## モデルプリセット

| プリセット | d\_model | レイヤー | Q ヘッド | KV ヘッド | seq\_len | パラメータ数 |
|-----------|--------:|--------:|--------:|---------:|---------:|-----------:|
| `nano` | 128 | 4 | 4 | 4 | 64 | ~886K |
| `small` | 512 | 6 | 8 | 8 | 2048 | ~50M |
| `medium` | 1024 | 12 | 16 | 8 GQA | 2048 | ~210M |
| `large` | 2048 | 24 | 32 | 8 GQA | 4096 | ~1.5B |
| `xlarge` | 4096 | 32 | 32 | 8 GQA | 4096 | ~7B |

---

## スケーリング実験

```bash
python scripts/run_scaling.py --experiment scale_n   # モデルサイズを変化
python scripts/run_scaling.py --experiment scale_d   # トークン数を変化
python scripts/run_scaling.py --experiment scale_c   # Chinchilla 最適フロンティア
```

**実測結果（TinyShakespeare、文字レベル）：**

| モデル | パラメータ数 | 検証損失 | トークン/秒 |
|-------|----------:|--------:|---------:|
| micro | 161K | 2.390 | 37,632 |
| nano | 1.01M | 2.243 | 26,028 |
| small | 5.10M | 2.179 | 17,118 |

---

## テスト

```bash
python -m pytest              # 全 180 テスト
python -m pytest tests/unit/  # 単体テストのみ
```

---

## 謝辞

本プロジェクトは **AI GDE TPU Sprint 2026** の一環です。  
**Google** による [TPU Research Cloud](https://sites.research.google/trc/about/) クレジットの提供に感謝します。

使用ライブラリ：[JAX](https://github.com/google/jax) · [Flax NNX](https://github.com/google/flax) · [Optax](https://github.com/google-deepmind/optax)  
参照アーキテクチャ：[nanochat](https://github.com/karpathy/nanochat)（Andrej Karpathy）

---

## ライセンス

[MIT](../../LICENSE)
