<div align="center">

# nanochat-jax

**JAX/Flax NNX로 구현한 nanochat 아키텍처 완전 이식 및 경험적 스케일링 법칙 실험 프레임워크.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.35+-a259ff)](https://github.com/google/jax)
[![Flax NNX](https://img.shields.io/badge/Flax_NNX-0.12+-34a853)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](../../LICENSE)

<br/>

[🇺🇸 English](../../README.md) &nbsp;|&nbsp; [🇨🇳 中文](README_zh.md) &nbsp;|&nbsp; [🇯🇵 日本語](README_ja.md) &nbsp;|&nbsp; 🇰🇷 한국어 &nbsp;|&nbsp; [🇫🇷 Français](README_fr.md) &nbsp;|&nbsp; [🇩🇪 Deutsch](README_de.md)

<br/>

> **AI GDE TPU Sprint 2026** — 이 프로젝트를 위한 TPU Research Cloud 크레딧을 제공해 주신 Google에 감사드립니다.

</div>

---

## 소개

`nanochat-jax`는 두 가지 목적을 가집니다:

1. **이식** — [nanochat](https://github.com/karpathy/nanochat)의 모든 아키텍처 세부 사항을 JAX + Flax NNX로 완전 재현합니다: 파라미터 없는 RMSNorm, relu² MLP, QK L2 정규화, logit softcap, 값 임베딩, 레이어별 스칼라, Smear/Backout 토큰 믹싱, 3차 Newton-Schulz 기반 Muon 옵티마이저, BOS 정렬 패킹.

2. **스케일링 실험** — Chinchilla 스타일의 `scale_n / scale_d / scale_c` 실험 실행 및 거듭제곱 법칙 피팅:

```
L = 3.29 × N^(−0.027)   [TinyShakespeare, 문자 수준, 600 스텝]
```

---

## 아키텍처 특징

| 컴포넌트 | 설명 |
|---------|------|
| **파라미터 없는 RMSNorm** | `y = x / √(mean(x²) + ε)`, 학습 파라미터 γ/β 없음 |
| **relu² MLP** | `h = x · relu(x)`, 2개의 선형 변환, `d_ff = 4 × d_model` |
| **QK L2 정규화** | Q/K를 L2 정규화 후 `1.2/√d_head` 스케일링 |
| **Logit Softcap** | `30 · tanh(logits/30)` — 어텐션 점수 폭발 방지 |
| **값 임베딩** | 레이어 공유 잔차 바이어스, `init=1e-4` |
| **레이어 스칼라** | 어텐션/FFN 출력에 학습 가능한 스칼라, 초기값 1 |
| **Smear/Backout** | 어텐션 전후 인과적 토큰 믹싱, sigmoid 게이트 초기값 -10 |
| **Muon 옵티마이저** | 3차 NS 직교화 (F-norm 정규화), 10 스텝, Nesterov 모멘텀 |

> **[인터랙티브 D3.js 다이어그램 →](../architecture.html)**

---

## 설치

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

## 빠른 시작

```bash
# 합성 데이터 스모크 테스트 (데이터 불필요)
python scripts/train.py --model-size nano --use-synthetic --total-steps 50

# TinyShakespeare 문자 수준 학습
curl -o data/tinyshakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python scripts/preprocess_shakespeare.py
python scripts/train.py --model-size nano --data-path data/shakespeare_char.h5 \
  --total-steps 2000 --learning-rate 3e-3
```

---

## 모델 프리셋

| 프리셋 | d\_model | 레이어 | Q 헤드 | KV 헤드 | seq\_len | 파라미터 수 |
|-------|--------:|-------:|-------:|--------:|---------:|-----------:|
| `nano` | 128 | 4 | 4 | 4 | 64 | ~886K |
| `small` | 512 | 6 | 8 | 8 | 2048 | ~68M |
| `medium` | 1024 | 12 | 16 | 8 GQA | 2048 | ~237M |
| `large` | 2048 | 24 | 32 | 8 GQA | 4096 | ~1.25B |
| `xlarge` | 4096 | 32 | 32 | 8 GQA | 4096 | ~6.03B |

---

## 측정 결과 (TinyShakespeare, 문자 수준)

| 모델 | 파라미터 수 | 검증 손실 | 토큰/초 |
|-----|----------:|--------:|-------:|
| micro | 161K | 2.390 | 37,632 |
| nano | 1.01M | 2.243 | 26,028 |
| small | 5.10M | 2.179 | 17,118 |

**`L = 3.29 × N^(−0.027)`** (600 스텝; 완전 학습 시 α → 0.07–0.12)

---

## 테스트

```bash
python -m pytest              # 전체 180개 테스트
python -m pytest tests/unit/  # 단위 테스트만
```

---

## 감사의 말

이 프로젝트는 **AI GDE TPU Sprint 2026**의 일부입니다.  
[TPU Research Cloud](https://sites.research.google/trc/about/) 크레딧을 제공해 주신 **Google**에 감사드립니다.

사용 라이브러리: [JAX](https://github.com/google/jax) · [Flax NNX](https://github.com/google/flax) · [Optax](https://github.com/google-deepmind/optax)  
참고 아키텍처: [nanochat](https://github.com/karpathy/nanochat) (Andrej Karpathy)

---

## 라이선스

[MIT](../../LICENSE)
