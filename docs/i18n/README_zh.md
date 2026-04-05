<div align="center">

# nanochat-jax

**基于 JAX/Flax NNX 的 nanochat 架构完整复现，支持 Chinchilla 风格缩放定律实验。**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-a259ff)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-0.12+-34a853)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](../../LICENSE)

<br/>

[🇺🇸 English](../../README.md) &nbsp;|&nbsp; 🇨🇳 中文 &nbsp;|&nbsp; [🇯🇵 日本語](README_ja.md) &nbsp;|&nbsp; [🇰🇷 한국어](README_ko.md) &nbsp;|&nbsp; [🇫🇷 Français](README_fr.md) &nbsp;|&nbsp; [🇩🇪 Deutsch](README_de.md)

<br/>

> **AI GDE TPU Sprint 2026** — 感谢 Google 提供 TPU Research Cloud 算力支持本项目。

</div>

---

## 项目简介

`nanochat-jax` 有两个核心目标：

1. **移植** — 用 JAX + Flax NNX 完整复现 [nanochat](https://github.com/karpathy/nanochat) 的每个架构细节：无参数 RMSNorm、relu² MLP、QK L2 归一化、logit softcap、值嵌入、逐层标量、Smear/Backout 令牌混合、基于三次 Newton-Schulz 的 Muon 优化器，以及 BOS 对齐打包。

2. **缩放实验** — 对上述模型进行 Chinchilla 风格的 `scale_n / scale_d / scale_c` 实验、幂律拟合，实测结果示例：

```
L = 3.29 × N^(−0.027)   [TinyShakespeare，字符级，600 步]
```

---

## 架构要点

| 组件 | 描述 |
|------|------|
| **无参数 RMSNorm** | `y = x / √(mean(x²) + ε)`，无学习参数 γ/β |
| **relu² MLP** | `h = x · relu(x)`，仅 2 个投影，`d_ff = 4 × d_model` |
| **QK L2 归一化** | Q/K 向量 L2 归一化后乘以 `1.2/√d_head` |
| **Logit Softcap** | `30 · tanh(logits/30)`，防止注意力分数爆炸 |
| **值嵌入** | 每 token 的残差偏置，跨层共享，`init=1e-4` |
| **逐层标量** | 注意力/FFN 输出各有一个可学习标量，初始化为 1 |
| **Smear/Backout** | 注意力前/后的因果令牌混合，sigmoid 门初始化为 -10 |
| **Muon 优化器** | 三次 NS 正交化（F-范数归一化），10 步，Nesterov 动量 |

> **[查看交互式架构图 →](../architecture.html)**（在浏览器中打开，悬停可查看详情）

---

## 安装

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

## 快速开始

```bash
# 合成数据冒烟测试（无需下载数据）
python scripts/train.py --model-size nano --use-synthetic --total-steps 50

# TinyShakespeare 字符级训练
curl -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python scripts/preprocess_shakespeare.py
python scripts/train.py --model-size nano --data-path data/shakespeare_char.h5 \
  --total-steps 2000 --learning-rate 3e-3
```

---

## 模型规模预设

| 预设 | d\_model | 层数 | Q 头数 | KV 头数 | seq\_len | 参数量 |
|------|--------:|-----:|-------:|--------:|---------:|-------:|
| `nano` | 128 | 4 | 4 | 4 | 64 | ~886K |
| `small` | 512 | 6 | 8 | 8 | 2048 | ~68M |
| `medium` | 1024 | 12 | 16 | 8 GQA | 2048 | ~237M |
| `large` | 2048 | 24 | 32 | 8 GQA | 4096 | ~1.25B |
| `xlarge` | 4096 | 32 | 32 | 8 GQA | 4096 | ~6.03B |

---

## 缩放实验

```bash
python scripts/run_scaling.py --experiment scale_n   # 固定计算量，变化模型大小
python scripts/run_scaling.py --experiment scale_d   # 固定模型，变化 token 数量
python scripts/run_scaling.py --experiment scale_c   # Chinchilla 最优计算边界
```

```python
from nanochat.scaling.analysis import fit_power_law
import numpy as np

fit = fit_power_law(
    xs=np.array([r.n_params for r in results]),
    ys=np.array([r.final_val_loss for r in results]),
)
print(f"L = {fit['a']:.3f} × N^(-{fit['alpha']:.4f})")
```

---

## 测试

```bash
python -m pytest          # 全部测试（共 180 个）
python -m pytest tests/unit/  # 仅单元测试
```

---

## 致谢

本项目是 **AI GDE TPU Sprint 2026** 的组成部分。  
特别感谢 **Google** 通过 [TPU Research Cloud](https://sites.research.google/trc/about/) 提供算力支持。

基础框架：[JAX](https://github.com/google/jax) · [Flax NNX](https://github.com/google/flax) · [Optax](https://github.com/google-deepmind/optax)  
参考架构：[nanochat](https://github.com/karpathy/nanochat)（Andrej Karpathy）

---

## 许可证

[MIT](../../LICENSE)
