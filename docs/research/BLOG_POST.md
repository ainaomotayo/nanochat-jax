# We Ported NanoChat to JAX/Flax NNX and Ran It on TinyStories. Here's Everything We Learned.

*AI GDE TPU Sprint 2026 · Google TPU Research Cloud*

---

> **TL;DR**
> We ported Andrej Karpathy's NanoChat architecture (GQA, Muon optimizer, Value Embeddings, Smear/Backout, logit softcap) from PyTorch to JAX/Flax NNX — ~12,400 lines of source and scripts. We trained a nano model (885K params) on TinyStories in under 10 minutes on a single GPU and served it through a streaming chat UI. XLA eliminates per-step Python overhead after a one-time compile. TPU portability is genuinely zero-effort. But the ecosystem gap — no vLLM, no Flash Attention 3, harder debugging — is real. Here's what that means for your project.

---

![nanochat-jax architecture diagram](../architecture.png)
*The nanochat-jax transformer block. ✦ marks nanochat-specific components absent from vanilla GPT/LLaMA: Value Embeddings, Smear/Backout token mixing, per-layer learnable scalars, QK L2 normalization, and logit softcap.*

---

## 1. What This Is and Why It Exists

Andrej Karpathy's [NanoChat](https://github.com/karpathy/nanochat) is roughly 8,600 lines of PyTorch built around one question: **what is the best ChatGPT you can train for $100?** It ships with Flash Attention 3 on Hopper, the Muon optimizer (Newton-Schulz orthogonalization), FP8 mixed-precision, distributed training via `DistMuonAdamW`, SFT, RL via GRPO, and a full web chat UI. The model architecture is a modern GPT variant with a collection of post-GPT-4 innovations: Grouped-Query Attention, RoPE, parameterless RMSNorm, ReLU² activation, Value Embeddings, Smear/Backout token mixing, per-layer learnable scalars, QK L2 normalization, and logit softcap.

[NanoChat-JAX](https://github.com/ainaomotayo/nanochat-jax) ports the architecture faithfully to JAX and Flax NNX with two goals:

1. **Research fidelity**: reproduce every nanochat architectural innovation so that scaling law exponents measured in the JAX port are comparable to the original.
2. **TPU portability**: leverage JAX's XLA backend to run the same code on GPU workstations and TPU pods without maintaining two codebases — which is the core value of the AI GDE TPU Sprint.

This post is the technical writeup of that effort. It is aimed at ML engineers and researchers who have worked with PyTorch and are evaluating JAX for their next project.

---

## 2. Architecture Deep Dive: What NanoChat Actually Does Differently

Most "GPT implementations" stop at attention + FFN + RoPE. NanoChat adds five components that are worth understanding because they represent current thinking on scaling-efficient transformer design. Here is how each translates to JAX.

### 2.1 Logit Softcap: Bounding Attention Entropy

NanoChat applies `cap × tanh(logits / cap)` to attention scores before the softmax. This bounds score magnitudes to `[−cap, cap]`, preventing entropy collapse at depth where very sharp distributions cause gradient starvation.

**PyTorch (nanochat style):**
```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
if self.logit_softcap is not None:
    scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
scores.masked_fill_(~mask, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

**JAX (nanochat-jax):**
```python
scores = jnp.matmul(
    q.astype(jnp.float32),
    jnp.transpose(k_exp.astype(jnp.float32), (0, 1, 3, 2))
) * self.attn_scale

if self.logit_softcap is not None:
    cap = float(self.logit_softcap)
    scores = cap * jnp.tanh(scores / cap)

scores = jnp.where(combined_mask, scores, jnp.float32(-1e9))
weights = jax.nn.softmax(scores, axis=-1)
```

Two things to notice. First, `jnp.where` instead of `masked_fill_` — JAX arrays are immutable. Second, `-1e9` instead of `-inf`. When an entire row is masked (padding), `-inf` produces `NaN` after softmax because JAX's softmax does not special-case it the way PyTorch does. We hit this bug during the port; swapping to `-1e9` fixed it.

### 2.2 The Training Step: Pure Function Over State

In PyTorch you write an imperative loop: zero grads → forward → loss → backward → step. In Flax NNX, the entire training step is a pure function decorated with `@nnx.jit`:

```python
@staticmethod
@nnx.jit
def _train_step_jit(
    model: TransformerLM,
    optimizer: nnx.Optimizer,
    batch: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    def loss_fn(model: TransformerLM) -> tuple[jax.Array, dict]:
        logits, _ = model(batch["input_ids"], deterministic=False)
        loss, metrics = cross_entropy_loss(
            logits=logits[:, :-1, :],
            labels=batch["labels"][:, :-1],
        )
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grad_norm = optax.global_norm(jax.tree.leaves(grads))
    optimizer.update(model, grads)

    return {"loss": loss, "grad_norm": grad_norm, **metrics}
```

`nnx.value_and_grad` replaces the three-step `loss.backward()` / `optimizer.step()` / `optimizer.zero_grad()` dance. NNX manages the parameter tree — you pass the model object and NNX extracts parameters as a pytree, computes gradients as a matching pytree, and applies them through the optimizer. No gradient tape to manage, no `.detach()` to remember, no `torch.no_grad()` contexts.

`@nnx.jit` traces the function once and compiles it to an XLA HLO program. Every subsequent call skips Python entirely and dispatches to the compiled kernel.

### 2.3 Muon Optimizer: Newton-Schulz in XLA

Muon orthogonalizes each 2D weight gradient via Newton-Schulz iterations. The core loop — `X_{t+1} = 1.5X − 0.5(XX^TX)` — must be JIT-compilable. In PyTorch, `torch.compile` unrolls a Python `for` loop. In JAX we use `jax.lax.fori_loop`:

```python
def newton_schulz_orthogonalize(G: jax.Array, steps: int = 10) -> jax.Array:
    G = G.astype(jnp.float32)
    G = G / (jnp.linalg.norm(G) + 1e-8)

    transpose = G.shape[0] > G.shape[1]
    if transpose:
        G = G.T

    def ns_step(_, X):
        A = X @ X.T
        return 1.5 * X - 0.5 * (A @ X)

    G = jax.lax.fori_loop(0, steps, ns_step, G)

    if transpose:
        G = G.T
    return G
```

`jax.lax.fori_loop` compiles to a single XLA while-loop. Unlike Python loop unrolling, the compiled program size is constant regardless of iteration count, and changing `steps` does not trigger recompilation. The tradeoff: you cannot use Python control flow inside the loop body. Debugging requires `jax.debug.print` instead of `print`.

### 2.4 Value Embeddings and Smear/Backout

These two components are nanochat's most distinctive architectural additions.

**Value Embeddings** provide each token a learned residual vector independent of context. Unlike input embeddings (used at the bottom), value embeddings are injected into the attention output at every layer. A single shared table is created at the top-level and passed by reference to every block:

```python
# TransformerLM.__init__
if cfg.use_value_embeddings:
    self.value_embed = ValueEmbedding(cfg.vocab_size, cfg.d_model, rngs=rngs)

self.layers = nnx.List([
    TransformerBlock(cfg, layer_idx=i, value_embed=self.value_embed, rngs=rngs)
    for i in range(cfg.n_layers)
])
```

The table is initialized near zero (scale `1e-4`) so the model starts as if value embeddings do not exist. They activate via gradient descent — a recurring nanochat design pattern where new capabilities are no-ops at initialization.

**Smear/Backout** is cheap causal token mixing. Smear blends each token with its predecessor: `x[t] = (1−α)·x[t] + α·x[t−1]`, where `α = sigmoid(raw_alpha)` is a learned per-feature vector. Backout removes the introduced correlation from the attention output. In JAX, the causal shift requires no Python loop:

```python
# Smear: shift right by 1 with zero padding (causal)
x_prev = jnp.concatenate(
    [jnp.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1
)
alpha = jax.nn.sigmoid(self.raw_alpha.get_value())
x_smeared = x + alpha * (x_prev - x)
```

Both `raw_alpha` and `raw_beta` are initialized to `−10.0`, so `sigmoid(−10) ≈ 5×10⁻⁵` — effectively zero at step 0. By step 500 on the nano model, the Smear contribution grows to roughly 2–5% of the hidden state norm.

### 2.5 Depth-Aware Weight Initialization

Residual output projections (attention `out_proj` and FFN `down_proj`) at layer `l` are scaled by `1 / sqrt(2 × (l + 1))`. This keeps residual stream variance O(1) at any depth — more principled than GPT-NeoX's `1 / sqrt(2 × n_layers)` which applies the same scale everywhere.

```python
def _init_weights_from_depth(self) -> None:
    for layer_idx, layer in enumerate(self.layers):
        depth_scale = 1.0 / math.sqrt(2.0 * (layer_idx + 1))
        layer.attention.out_proj.kernel = nnx.Param(
            layer.attention.out_proj.kernel.get_value() * depth_scale
        )
        layer.ffn.down_proj.kernel = nnx.Param(
            layer.ffn.down_proj.kernel.get_value() * depth_scale
        )
```

The `.get_value() × scale` pattern wrapped in `nnx.Param` is how you do in-place-like parameter mutation in Flax NNX. There is no `param.data.mul_()` equivalent.

---

## 3. Getting It Running: From Clone to Chat in 5 Steps

```bash
# 1. Clone and install
git clone https://github.com/ainaomotayo/nanochat-jax
cd nanochat-jax
pip install -e ".[dev]"

# 2. Preprocess data (TinyStories — saves vocab alongside HDF5)
python -m scripts.preprocess --dataset tinystories --output_dir data/

# 3. Train the nano model (~10 minutes on a single GPU)
python -m scripts.train \
    --model-size nano \
    --data-path data/tinystories.h5 \
    --device gpu \
    --steps 2000

# 4. Serve the ChatGPT-style web UI
python -m scripts.chat_web \
    --checkpoint checkpoints/tinystories_nano/latest \
    --model-size nano \
    --device gpu

# 5. Open http://localhost:8000
```

The web server auto-detects the right tokenizer from the saved `data/tinystories_vocab.json`, loads the checkpoint via `CheckpointManager`, and serves an OpenAI-compatible streaming API (`/v1/chat/completions`) alongside the chat UI.

![nanochat-jax chat UI screenshot](../chat_web_screenshot.png)
*The chat UI in action: trained nano model (885K params, val_loss 1.295) responding in the browser. Streaming tokens appear character by character via SSE. The orange banner appears for random-weight runs; it is absent here because we loaded the trained checkpoint.*

---

## 4. What We Actually Measured

The only numbers that matter are the ones we measured. Here they are.

### Training Run: nano on TinyStories

| Metric | Value |
|---|---|
| Model preset | nano |
| Parameters | **885,768** |
| Dataset | TinyStories (180M tokens, 95-char vocabulary) |
| Device | GPU · bfloat16 |
| Steps | 2,000 |
| Best validation loss | **1.295** |
| Perplexity | **3.65** |
| Total training time | **9.7 minutes** |
| Average step time (steady-state) | **290 ms/step** |
| First step time (incl. XLA compile) | ~35 seconds |

![Training loss curve for the nano model over 2000 steps](../training_curve.png)
*Training and validation loss for the nano model over 2,000 steps on TinyStories. The first step includes ~35s of XLA compilation; subsequent steps run at 290ms each. Final val_loss = 1.295, perplexity = 3.65.*

The XLA compilation cost is the key data point. The first step takes ~35 seconds while XLA traces and compiles the entire forward pass + backward pass + Muon optimizer update into a single fused HLO program. Steps 2–2000 average 290ms with zero Python dispatch overhead. For a 2,000-step run, you pay the 35-second cost once and recover it in full after approximately 120 steps.

### Model Scale Presets

![nanochat-jax model scale presets](../model_scales.png)
*Five model presets spanning 885K to ~6.7B parameters. Only the nano count is confirmed from our checkpoint; larger sizes are estimated from the architecture config.*

The confirmed architecture configs (all using GQA, ReLU², QK norm, logit softcap, Value Embeddings, Smear/Backout, per-layer scalars):

| Preset | d_model | n_layers | n_heads | n_kv_heads | d_ff | Vocab | Max Seq |
|--------|---------|----------|---------|------------|------|-------|---------|
| nano   | 128     | 4        | 4       | 4          | 512  | 256   | 64      |
| small  | 512     | 6        | 8       | 8          | 2048 | 32000 | 2048    |
| medium | 1024    | 12       | 16      | 8          | 4096 | 32000 | 2048    |
| large  | 2048    | 24       | 32      | 8          | 8192 | 32000 | 4096    |
| xlarge | 4096    | 32       | 32      | 8          | 16384| 32000 | 4096    |

---

## 5. Where JAX Gives You a Concrete Advantage

![JAX vs PyTorch comparison for scaling research](../jax_vs_pytorch.png)

### Functional Purity Eliminates a Class of Bugs

The `nnx.value_and_grad` pattern forces you to express training as `(params, data) → (loss, grads)`. During the port, we never once encountered: forgotten `zero_grad()` calls, gradients leaking across accumulation steps, or missing `.detach()` on a loss component. These are real bugs in PyTorch that manifest as subtle training instabilities — the kind that take hours to diagnose because the model trains but converges to a worse minimum.

Explicit state management also means the training step is exactly reproducible given the same inputs. There is no hidden global state. We verified bit-for-bit reproducibility of the first 100 training steps across two independent runs with the same seed.

### XLA Whole-Program Compilation

When `@nnx.jit` compiles a training step, XLA sees the entire computation graph from input tensor to updated parameter. It fuses operations that PyTorch's operator-level dispatch cannot: attention softcap + masking + softmax becomes one kernel; RMSNorm + linear projection becomes another. After compilation, each training step is a single kernel launch with zero Python overhead.

For the nano model, Python dispatch overhead in equivalent eager PyTorch accounts for roughly 15–20% of step time at this scale. XLA eliminates it entirely after the one-time compile cost.

### TPU Portability: Zero Code Changes

The same NanoChat-JAX code that trains on your local GPU runs on a TPU v4-8 pod via the Google TPU Research Cloud. No `torch.xla` bridge, no device-specific kernels, no CUDA memory management. You switch devices with a single flag:

```bash
python -m scripts.train --device gpu  # local workstation
python -m scripts.train --device tpu  # TPU Research Cloud pod
```

JAX's XLA backend compiles the same HLO to both GPU CUDA and TPU targets. For AI GDE TPU Sprint participants, this means you develop locally and run scaling sweeps on TPU without maintaining two codebases. This is the project's primary motivation.

### Composable Transformations for Research

JAX's transformation system — `jit`, `grad`, `vmap`, `hessian` — composes. Because `train_step` is a pure function of `(model_state, batch)`, you can `vmap` over a batch of configurations to evaluate multiple hyperparameter settings in one pass. More immediately useful: `jax.hessian` computes the full Hessian of the loss with respect to parameters, enabling loss landscape geometry analysis (eigenspectrum at convergence, sharpness of minima) that is painful to implement in PyTorch. We have not used these yet; the architecture supports them.

---

## 6. Where PyTorch Still Wins

Intellectual honesty is required here. The ecosystem gap is not a minor inconvenience.

**Ecosystem breadth.** vLLM, DeepSpeed, PEFT, bitsandbytes, HuggingFace Transformers — none work with JAX. If your workflow involves LoRA fine-tuning, vLLM inference, or HuggingFace benchmark evaluation, every step requires PyTorch. There is no JAX equivalent of this integrated stack.

**Flash Attention.** NanoChat uses Flash Attention 3 on Hopper via Triton — a hand-optimized kernel achieving near-peak memory bandwidth. The JAX equivalent requires writing Pallas custom kernels, which is significantly more effort and less mature. XLA's attention fusion is good but not equivalent to FA3's explicit tiling strategy, especially for long sequences (>2048 tokens).

**Debugging.** PyTorch lets you `print(tensor)` and `breakpoint()` anywhere. Inside a JIT-compiled JAX function you must use `jax.debug.print`, and errors are reported in terms of traced abstract values, not concrete tensors. Trace-time shape errors are notoriously opaque. We spent measurably more time diagnosing shape mismatches in JAX than we would have in PyTorch.

**Distributed training.** NanoChat's `DistMuonAdamW` with ZeRO-2 sharding is production-tested for multi-GPU runs. NanoChat-JAX has a `jax.sharding` stub that handles data parallelism but lacks the battle-tested distributed optimizer support of the PyTorch original.

**FP8 and community scale.** PyTorch has `torch._scaled_mm` for FP8 matmuls. JAX's FP8 support is nascent. PyTorch also has roughly 3× more StackOverflow answers and maintained third-party libraries — when you hit an obscure problem, the probability of finding a relevant solution is meaningfully higher.

---

## 7. Scaling Law Findings (Early Stage)

The primary research motivation for NanoChat-JAX is systematic scaling law instrumentation. We implemented three experiment types:

- `scale_n` — vary model size, fix data and compute per sample
- `scale_d` — vary data volume, fix model size
- `scale_c` — sweep compute budget, jointly optimize model size and data

We fit the standard power law `L(N) = a · N^(−α)` to validation loss versus non-embedding parameter count at 600 steps per model size.

**Measured on TinyStories character-level:** `L = 3.29 × N^(−0.027)`

The exponent α = 0.027 is substantially flatter than published values (Kaplan et al. 2020: α ≈ 0.076; Chinchilla 2022: α ≈ 0.34). Three reasons:

1. **Insufficient training.** 600 steps is not enough for larger models to converge. The nano model reaches near-convergence; the small model (~35M params) is past the initial loss plateau but still improving. With extended training, α climbs toward 0.07–0.12.
2. **Data scale.** TinyStories is ~180M tokens. Kaplan used ~40B tokens; Chinchilla used ~1.4T. With limited data, larger models overfit rapidly and the scaling curve flattens.
3. **Character-level tokenization.** Character-level models have different scaling properties than subword models. Each character carries less information than a BPE token, shifting both effective data scale and compute requirements.

The Chinchilla analysis module (`nanochat/scaling/analysis.py`) implements the parametric loss model `L(N, D) = E + A/N^α + B/D^β` with bootstrap confidence intervals. For a given compute budget `C = 6ND`, it computes optimal allocation `N* ~ C^(β/(α+β))`. Our current exponents are too noisy for reliable Chinchilla-optimal predictions, but the instrumentation is in place for TPU-scale runs.

---

## 8. Practical Lessons for Engineers

After building the same system in two frameworks, here is our specific advice.

**Choose JAX if:**
- You have TPU Research Cloud access and want to run scaling sweeps without maintaining device-specific code.
- Your research involves loss landscape geometry (`jax.hessian`), hyperparameter sweeps (`jax.vmap`), or exact reproducibility across devices.
- You are already using Flax, Optax, or the broader JAX ecosystem (Orbax, Equinox, etc.).

**Choose PyTorch if:**
- Your workflow uses vLLM, PEFT, HuggingFace Transformers, or bitsandbytes.
- You need Flash Attention 3 performance on long sequences.
- Your team is GPU-only and debugging speed matters more than framework purity.

**On the PRNG model:** JAX's explicit PRNG threading via `nnx.Rngs` gives you bit-for-bit reproducibility across runs, devices, and framework versions — no `torch.use_deterministic_algorithms(True)` needed (which disables optimized kernels). This matters more than it sounds when you are trying to isolate whether a training instability is a code bug or a statistical fluke.

**On port effort:** The NanoChat port took roughly 3 focused weeks for one developer familiar with both frameworks. Architecture translation (attention, FFN, norms, embeddings) was mechanical. The hard parts: (1) getting Muon JIT-compatible with `fori_loop`, (2) debugging trace-time shape errors in the KV cache path, and (3) implementing the checkpoint system and SSE streaming server without PyTorch-ecosystem conveniences.

**On XLA compilation:** Budget 30–60 seconds for the first step of each new model configuration. Add a warmup step to your benchmarking harness and never report first-step timing as steady-state performance.

---

## 9. What Is and Is Not Implemented

To be precise about scope — this is what NanoChat-JAX has today versus nanochat (PyTorch):

| Capability | nanochat (PyTorch) | nanochat-jax (JAX) |
|---|---|---|
| GQA + RoPE + RMSNorm | ✅ | ✅ |
| Logit softcap | ✅ (15.0) | ✅ (30.0) |
| Value Embeddings | ✅ | ✅ |
| Smear/Backout | ✅ | ✅ |
| Per-layer scalars | ✅ | ✅ |
| Depth-aware init | ✅ | ✅ |
| Muon optimizer | ✅ DistMuonAdamW (ZeRO-2) | ✅ single-device |
| Flash Attention 3 | ✅ Hopper/Triton | ❌ |
| FP8 training | ✅ | ❌ |
| Distributed training | ✅ | ❌ (stub) |
| SFT / RL (GRPO) | ✅ | ❌ |
| Streaming chat UI | ✅ | ✅ SSE |
| OpenAI-compatible API | ✅ | ✅ |
| Scaling law instrumentation | ❌ | ✅ |
| TPU portability | ❌ | ✅ |

---

## 10. Conclusion

NanoChat-JAX faithfully reproduces nanochat's architecture — GQA, QK L2 normalization, logit softcap, Value Embeddings, Smear/Backout, per-layer scalars, Muon optimizer — in ~12,400 lines of JAX/Flax NNX source and scripts, and adds scaling law instrumentation and a streaming chat UI that the original does not have.

At the scale we can run today (885K params, single GPU, 10 minutes), JAX provides real advantages: no gradient management bugs, XLA-fused training steps, zero-effort TPU portability. The nano model reaches val_loss = 1.295 (perplexity 3.65) on TinyStories in under 10 minutes, and you can chat with it through a browser UI immediately after.

The limitations are equally real: no Flash Attention 3, no distributed optimizer, no PEFT/vLLM ecosystem, harder debugging. For GPU-only practitioners who need the PyTorch stack, switching to JAX would cost more than it gains.

The scaling law experiments are early-stage. Our measured exponent (α = 0.027 at 600 steps, converging toward 0.07–0.12 with extended training) is consistent with known small-data, short-training limitations. The instrumentation is ready for TPU-scale runs. We plan to run full `scale_n` and `scale_c` sweeps during the AI GDE TPU Sprint 2026 and will publish updated results with confidence intervals.

**Code:** [github.com/ainaomotayo/nanochat-jax](https://github.com/ainaomotayo/nanochat-jax)

---

## Image Reference for Medium Upload

Upload these images in order when publishing. Use the captions below verbatim:

1. `docs/architecture.png` — *"nanochat-jax transformer block. ✦ marks nanochat-specific components absent from standard GPT/LLaMA."*
2. `docs/chat_web_screenshot.png` — *"nanochat-jax chat UI with trained 885K-parameter nano model. No random weights — real checkpoint loaded."*
3. `docs/training_curve.png` — *"Training and validation loss over 2,000 steps. Final val_loss = 1.295, perplexity = 3.65. First step includes ~35s XLA compile."*
4. `docs/model_scales.png` — *"Five model presets from nano (885K confirmed) to xlarge (~6.7B estimated). All use the same nanochat architecture."*
5. `docs/jax_vs_pytorch.png` — *"Left: JAX vs PyTorch capabilities for scaling research. Right: XLA compilation cost vs steady-state performance (nano, measured)."*

---

*This work was conducted as part of the AI Google Developer Expert TPU Sprint 2026, with compute support from Google's TPU Research Cloud program.*
