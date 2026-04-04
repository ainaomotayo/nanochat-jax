# NanoChat vs NanoChat-JAX: A Full-Spectrum Comparative Study
## Software Engineering · Systems Design · ML Research · Publication & Dissemination
---

**Abstract.** This document presents a comprehensive comparative analysis of nanochat (PyTorch, Andrej Karpathy) and nanochat-jax (JAX/Flax NNX), a faithful port designed for scaling law research. We examine architectural fidelity, optimizer semantics, systems design, numerical behavior, and research capability across both codebases, grounding every claim in file-level source evidence. The analysis reveals that nanochat-jax achieves high fidelity on model architecture and novel features (value embeddings, Smear/Backout, per-layer scalars, logit softcap) while making deliberate divergences in optimizer implementation, LR scheduling, distributed training, and attention kernels. These divergences have measurable consequences for scaling law exponents, training efficiency, and the reproducibility of nanochat's published results.

**Scope.** Sections 1-6 of a 12-section study. Covers: Executive Summary, Architecture, Optimizer & Training Loop, Systems & Infrastructure, Scaling Law Methodology, and Numerical Fidelity.

**Codebases analyzed.**
- **nanochat**: `github.com/karpathy/nanochat`, PyTorch 2.9.1, ~8,600 LOC
- **nanochat-jax**: `/mnt/c/Users/ainao/Desktop/aigde/nanochat-jax/`, JAX 0.4.35+ / Flax NNX 0.10+, ~6,664 LOC source + ~2,629 LOC tests across 76 files

---

## Table of Contents (Sections 1-6)

1. [Executive Summary](#1-executive-summary) -- Motivation, key findings, quantitative summary
2. [Architecture Deep Dive](#2-architecture-deep-dive) -- GQA, QK norm, RoPE, RMSNorm, FFN, value embeddings, Smear/Backout, init, sliding window, KV cache
3. [Optimizer and Training Loop](#3-optimizer-and-training-loop) -- Muon, LR schedule, training step, weight decay, checkpointing
4. [Systems and Infrastructure](#4-systems-and-infrastructure) -- Module org, config, JIT, distributed, inference, data pipeline, tokenizer, throughput, observability, dependencies
5. [Scaling Law Methodology](#5-scaling-law-methodology) -- Experiment framework, power law fitting, Chinchilla, FLOP estimation, results analysis
6. [Numerical Fidelity and Precision](#6-numerical-fidelity-and-precision) -- Precision strategy, attention numerics, softcap, NS precision, determinism, memory layout

---

## 1. Executive Summary

### 1.1 Motivation

Andrej Karpathy's nanochat project demonstrates that a competitive ChatGPT-class model can be trained for approximately $100 of compute, leveraging a carefully tuned stack of modern architectural innovations. The project's subtitle -- "The best ChatGPT that $100 can buy" -- encapsulates a research agenda: given a fixed compute budget, what combination of architecture, optimizer, and data pipeline yields the best language model?

Nanochat-jax ports this architecture to JAX/Flax NNX with two goals: (1) reproduce the architectural innovations faithfully enough that scaling law exponents measured in the JAX port are comparable to the original, and (2) leverage JAX's functional transformation ecosystem (JIT compilation, automatic differentiation, device mesh sharding) for research on compute-optimal training.

### 1.2 Key Findings

**High-fidelity reproductions:**
- Grouped Query Attention with configurable n_heads/n_kv_heads ratio (`attention.py:104-162`)
- Parameterless RMSNorm without learned scale (`norms.py:31-101`)
- ReluSquared MLP as default FFN (`feedforward.py:44-114`)
- RoPE with base=100,000 (`embeddings.py:170-341`, `model_config.py:132`)
- Value embeddings with near-zero init (`value_embeddings.py:37-102`)
- Per-layer alpha scalars on attention and FFN outputs (`block.py:137-139`)
- Smear/Backout token mixing with sigmoid-gated interpolation (`token_mixing.py:58-168`)
- Logit softcap before softmax (`attention.py:282-284`)
- QK L2 normalization with 1.2x scale factor (`attention.py:141`, `model_config.py:163`)
- Depth-aware weight initialization (`transformer.py:201-236`)
- Untied input/output embeddings (`transformer.py:102-107`)

**Deliberate divergences:**

| Feature | nanochat (PyTorch) | nanochat-jax (JAX) |
|---|---|---|
| Attention kernel | Flash Attention 3 + SDPA fallback | JAX SDPA via manual matmul |
| Optimizer | MuonAdamW (Polar Express + NorMuon) | Muon (cubic NS, 5 iter) + AdamW fallback |
| LR schedule | Linear warmup (40 steps) -> constant -> linear warmdown | Linear warmup -> cosine decay -> constant floor |
| Grad clipping | None (Muon normalizes) | `optax.clip_by_global_norm(1.0)` |
| Distributed | Custom DistMuonAdamW with ZeRO-2 | Stub only (`distributed.py:14-87`) |
| FP8 | ~150 LOC torch._scaled_mm | Not implemented |
| Softcap value | 15.0 | 30.0 |
| Vocab size | 32,768 (2^15, custom BPE) | 100,284 (tiktoken cl100k_base) |
| Scaling dial | `--depth` auto-computes everything | Manual presets + `from_depth` factory |
| Mixed precision | Custom Linear bf16 cast, fp16+GradScaler | dtype config field, param_dtype always float32 |

**Missing capabilities in nanochat-jax:** SFT, RL (GRPO/REINFORCE), tokenizer training, Flash Attention, FP8, distributed training, tool use, web UI, DCLM CORE evaluation, auto-scaling hyperparameters.

### 1.3 Quantitative Summary

| Metric | nanochat | nanochat-jax |
|---|---|---|
| Source LOC | ~8,600 | ~6,664 |
| Test LOC | (integrated) | ~2,629 |
| Files | ~12 core | 46 source + 30 test |
| Model presets | 1 (auto via --depth) | 5 (nano through xlarge) |
| Scaling exponent (reported) | Power Lines reference | L = 3.29 x N^(-0.027) at 600 steps |
| Max tested scale | Multi-GPU, 2048 ctx | Single-device, 4096 ctx (xlarge preset) |

---

## 2. Architecture Deep Dive

### 2.1 Transformer Backbone

Both codebases implement a decoder-only GPT architecture with pre-norm residual connections. The forward pass follows the same pipeline:

```
input_ids -> TokenEmbedding -> [TransformerBlock x N] -> RMSNorm -> LM Head -> logits
```

This is documented in nanochat-jax at `transformer.py:1-6` and realized in `TransformerLM.__call__` (`transformer.py:135-195`). The model initializes an embedding layer, a stack of transformer blocks via `nnx.List`, a final parameterless RMSNorm, and optionally an untied LM head:

```python
# transformer.py:88-96
self.layers = nnx.List([
    TransformerBlock(cfg, layer_idx=i, value_embed=self.value_embed, rngs=rngs)
    for i in range(cfg.n_layers)
])
```

The use of `nnx.List` rather than a Python list is significant: Flax NNX's `nnx.List` ensures that each block's parameters are registered in the module tree, making them visible to `nnx.state()`, `nnx.jit`, and `nnx.value_and_grad`. This is the NNX equivalent of PyTorch's `nn.ModuleList`.

**Divergence note:** nanochat uses `meta device` initialization (constructing model parameters on a meta device to avoid memory allocation until sharding is applied). Nanochat-jax has no equivalent -- all parameters are materialized immediately on the default JAX device. This limits single-device model scale to available memory. [HYPOTHESIS: For models beyond ~1B parameters, nanochat-jax would need to integrate `jax.sharding` constraints during initialization to match nanochat's memory efficiency.]

### 2.2 Grouped Query Attention

The `GroupedQueryAttention` class (`attention.py:104-338`) implements the full MHA/GQA/MQA spectrum through the `n_heads`/`n_kv_heads` ratio:

```python
# attention.py:134-137
self.n_heads = cfg.n_heads
self.n_kv_heads = cfg.n_kv_heads
self.d_head = cfg.d_head
self.n_groups = cfg.n_heads // cfg.n_kv_heads
```

KV head expansion for GQA uses `jnp.repeat` (`attention.py:181-185`):

```python
def _expand_kv(self, x: jax.Array) -> jax.Array:
    if self.n_groups == 1:
        return x
    return jnp.repeat(x, repeats=self.n_groups, axis=1)
```

This is functionally identical to nanochat's approach but has a critical performance implication: `jnp.repeat` materializes the expanded tensor in memory, whereas Flash Attention 3 in nanochat handles the GQA expansion implicitly inside the fused kernel without materializing the repeated KV heads. For a model with 32 query heads and 8 KV heads (the `large` preset at `model_config.py:348-353`), this means nanochat-jax allocates 4x the KV memory that nanochat does during the attention computation.

**Attention score computation** proceeds through explicit matmul (`attention.py:272-275`):

```python
scores = jnp.matmul(
    q.astype(jnp.float32),
    jnp.transpose(k_exp.astype(jnp.float32), (0, 1, 3, 2))
) * self.attn_scale
```

The explicit cast to float32 before matmul ensures numerical stability in attention scores, matching nanochat's behavior of computing attention in higher precision. However, this also means the full `[B, n_heads, S, S]` attention matrix is materialized, giving O(S^2) memory scaling versus Flash Attention's O(S) tiled approach.

### 2.3 QK Normalization and Softcap

Nanochat-jax faithfully implements both QK L2 normalization and logit softcap. QK normalization (`attention.py:241-243`) applies L2 normalization to Q and K **before** RoPE:

```python
if self.use_qk_norm:
    q = _l2_normalize(q.astype(jnp.float32)).astype(q.dtype)
    k = _l2_normalize(k.astype(jnp.float32)).astype(k.dtype)
```

The `_l2_normalize` function (`attention.py:48-59`) divides by the L2 norm along the last axis with eps=1e-8. Applying normalization before RoPE is deliberate: it ensures that the rotation operates on unit vectors, so the dot product magnitudes after rotation remain bounded. This ordering matches the Gemma-2 technical report (2024) and Wortsman et al. (2023).

The attention scale factor (`attention.py:141`) uses `qk_scale_factor / sqrt(d_head)` instead of the standard `1/sqrt(d_head)`, with a default of 1.2 (`model_config.py:163`). This 20% boost compensates for the variance reduction caused by L2 normalization.

**Logit softcap** (`attention.py:282-284`) applies `cap * tanh(scores / cap)` before masking:

```python
if self.logit_softcap is not None:
    cap = float(self.logit_softcap)
    scores = cap * jnp.tanh(scores / cap)
```

**Critical divergence:** Nanochat uses `cap=15.0` while nanochat-jax defaults to `cap=30.0` (`model_config.py:176`). A softcap of 15 bounds attention logits to [-15, 15] with aggressive compression starting around magnitude 10. A softcap of 30 provides a wider linear regime before compression begins. This difference affects gradient flow through the attention mechanism: the smaller cap creates stronger gradient clipping on extreme attention patterns, which can be either beneficial (stability) or harmful (capacity). [HYPOTHESIS: The 2x difference in softcap was likely a deliberate conservative choice during the port, since the JAX implementation lacks Flash Attention's numerical advantages and needed a wider dynamic range to maintain training stability.]

### 2.4 Rotary Position Embeddings

The `RotaryEmbedding` class (`embeddings.py:145-341`) is implemented as a pure-JAX utility class (not an `nnx.Module`) that precomputes sin/cos frequency tables. The frequency computation (`embeddings.py:232-252`) follows the standard RoPE formulation:

```python
inv_freq = 1.0 / (base ** (2.0 * freq_indices / d_head))
angles = jnp.outer(positions, inv_freq)
cos_table = jnp.cos(angles)
sin_table = jnp.sin(angles)
```

Both codebases use `rope_base=100,000` (`model_config.py:132`, matching Su et al. 2022's extended base for longer contexts). The `apply` static method (`embeddings.py:288-335`) uses the split-and-rotate approach:

```python
x1, x2 = x[..., :half_d], x[..., half_d:]
rotated_x1 = x1 * cos - x2 * sin
rotated_x2 = x1 * sin + x2 * cos
rotated = jnp.concatenate([rotated_x1, rotated_x2], axis=-1)
```

This is numerically equivalent to nanochat's implementation. The `get_freqs` method (`embeddings.py:254-286`) supports positional offsets for incremental decoding, which is essential for KV cache compatibility.

### 2.5 Parameterless RMSNorm

This is one of nanochat's distinctive architectural choices, faithfully replicated. The `RMSNorm` class (`norms.py:31-101`) deliberately omits the learned scale parameter (gamma):

```python
# norms.py:70-71
# No gamma parameter -- this is intentional for nanochat fidelity.
# Do NOT add self.gamma here.
```

The forward pass (`norms.py:75-98`) promotes to float32, computes mean-squared, and normalizes:

```python
x_f32 = x.astype(jnp.float32)
mean_sq = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
rms_inv = jax.lax.rsqrt(mean_sq + self.eps)
normed = x_f32 * rms_inv
return normed.astype(input_dtype)
```

The use of `jax.lax.rsqrt` rather than `1.0 / jnp.sqrt(...)` is a subtle optimization: rsqrt compiles to a single hardware instruction on both GPU (MUFU.RSQ) and TPU, avoiding the division.

**Design rationale** (documented in `norms.py:1-18`): The residual stream magnitude is controlled by depth-aware weight initialization rather than learned norms. This reduces parameter count, eliminates a potential source of training instability at depth, and is a deliberate departure from LLaMA/Gemma which use learned gamma.

A `LayerNorm` class with learned affine parameters is provided (`norms.py:104-142`) for ablation experiments but is not used in the default configuration.

### 2.6 Feed-Forward Networks

Nanochat-jax provides four FFN variants in a registry (`block.py:64-69`):

```python
_FFN_REGISTRY: dict[str, type] = {
    "relu2": ReLUSquaredMLP,
    "swiglu": SwiGLUFFN,
    "geglu": GeGLUFFN,
    "gelu": StandardMLP,
}
```

The nanochat-faithful default is `ReLUSquaredMLP` (`feedforward.py:44-114`), which implements relu^2(x) = x * relu(x):

```python
# feedforward.py:106-107
h = self.gate_proj(x)          # [B, S, d_ff]
h = h * jax.nn.relu(h)         # relu^2: x * max(0, x)
```

This uses `d_ff = 4 * d_model` (two-projection architecture), unlike gated variants (SwiGLU, GeGLU) which use three projections with `d_ff = round_up(2/3 * 4 * d_model, 256)` to equalize parameter count (`feedforward.py:140-141`).

**Implementation note:** The relu^2 formulation `h * jax.nn.relu(h)` is numerically equivalent to `jnp.square(jax.nn.relu(h))` for positive inputs but avoids a potential numerical issue: `jnp.square(relu(x))` computes relu then squares, while `x * relu(x)` keeps the original (possibly negative) value multiplied by zero, avoiding any gradient through the squared path when x < 0. Both formulations produce the same forward result, but `x * relu(x)` has better-behaved gradients under XLA's automatic differentiation.

### 2.7 Value Embeddings

The `ValueEmbedding` class (`value_embeddings.py:37-102`) implements per-token learned residual vectors injected at each transformer block's attention output. A single shared table is created in `TransformerLM.__init__` (`transformer.py:80-85`) and passed by reference to all blocks:

```python
# transformer.py:80-83
if cfg.use_value_embeddings:
    self.value_embed = ValueEmbedding(cfg.vocab_size, cfg.d_model, rngs=rngs)
```

The table is initialized near-zero (`value_embeddings.py:77-79`):

```python
self.table = nnx.Param(
    jax.random.normal(key, (vocab_size, d_model)) * init_scale
)  # init_scale = 1e-4
```

Injection occurs in the block's forward pass (`block.py:237-239`):

```python
if self._value_embed is not None and token_ids is not None:
    v_embed = self._value_embed(token_ids)  # [B, S, d_model]
    attn_out = attn_out + v_embed
```

**Comparison with nanochat:** The original uses ResFormer-style value embeddings with alternating layers and a learned gating mechanism. Nanochat-jax simplifies this to injection at every layer without gating. The near-zero initialization (1e-4) ensures value embeddings start as no-ops and grow during training, preventing them from disrupting the attention mechanism early on.

**Parameter cost:** For vocab_size=32,000 and d_model=512 (small preset), the value embedding table adds 16.4M parameters -- roughly 24% overhead on a ~68M model. This cost is amortized across all layers since the table is shared.

### 2.8 Per-Layer Scalars

Learnable scalar weights on attention and FFN outputs (`block.py:137-139`):

```python
if cfg.use_per_layer_scalars:
    self.alpha_attn = nnx.Param(jnp.ones(()))  # scalar
    self.alpha_ffn = nnx.Param(jnp.ones(()))   # scalar
```

These are applied during the residual addition (`block.py:242-245`):

```python
if self.alpha_attn is not None:
    x = residual + self.alpha_attn.get_value() * attn_out
else:
    x = residual + attn_out
```

Initialization to 1.0 means the model starts with standard residual behavior. The scalars learn to modulate each sublayer's contribution independently.

**Comparison with nanochat:** The original uses per-layer residual lambdas that decay from 1.15 to 1.05 across depth, plus separate input embedding blending lambdas (x0_lambdas) that decay from 0.20 to 0.05. Nanochat-jax's uniform initialization at 1.0 is simpler but may converge to different dynamics. The nanochat lambdas encode a prior that deeper layers should contribute less to the residual stream -- a form of implicit depth regularization. [HYPOTHESIS: The uniform 1.0 initialization combined with depth-aware weight init (Section 2.10) may achieve a similar effect, since the weight scaling already reduces deeper layers' contributions.]

### 2.9 Smear and Backout Token Mixing

The `Smear` class (`token_mixing.py:58-113`) implements causal predecessor blending:

```python
# token_mixing.py:99-108
alpha = jax.nn.sigmoid(self.raw_alpha.get_value())  # [d_model]
x_prev = jnp.concatenate(
    [jnp.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1
)
x_smeared = x + alpha * (x_prev - x)
```

The `raw_alpha` parameter is initialized to -10.0 (`token_mixing.py:84`), giving `sigmoid(-10) = 4.5e-5`, effectively a no-op at initialization. The per-feature (not per-head) design gives the model fine-grained control over which feature dimensions benefit from predecessor blending.

The `Backout` class (`token_mixing.py:116-168`) applies a corrective subtraction after attention:

```python
# token_mixing.py:162-163
beta = jax.nn.sigmoid(self.raw_beta.get_value())
return attn_out - beta * x_prev
```

**Computational cost:** Both operations are O(B * S * d_model) -- strictly cheaper than any attention variant. The concatenation-based shift (`jnp.concatenate([zeros, x[:, :-1, :]], axis=1)`) is JIT-compilable and avoids Python loops.

**Relation to RWKV:** The documentation (`token_mixing.py:44`) notes a conceptual similarity to RWKV-style token mixing (Peng et al., 2023). However, Smear/Backout is simpler: RWKV uses exponential decay over all previous positions, while Smear only blends with the immediate predecessor (window size 1).

### 2.10 Depth-Aware Weight Initialization

The `_init_weights_from_depth` method (`transformer.py:201-236`) scales residual output projections by `1/sqrt(2*(layer_idx+1))`:

```python
# transformer.py:222-230
for layer_idx, layer in enumerate(self.layers):
    depth_scale = 1.0 / math.sqrt(2.0 * (layer_idx + 1))
    if hasattr(layer.attention, "out_proj"):
        _scale_linear(layer.attention.out_proj, depth_scale)
    if hasattr(layer.ffn, "down_proj"):
        _scale_linear(layer.ffn.down_proj, depth_scale)
```

The scaling helper (`transformer.py:285-293`) multiplies the kernel in-place:

```python
def _scale_linear(linear: nnx.Linear, scale: float) -> None:
    if hasattr(linear, "kernel"):
        linear.kernel = nnx.Param(linear.kernel.get_value() * scale)
```

**Comparison with GPT-NeoX:** Standard practice (GPT-NeoX, GPT-3) uses a global `1/sqrt(2*n_layers)` factor. The nanochat/nanochat-jax approach adapts to actual depth at each layer. Early layers (small `layer_idx`) get larger scales since the residual stream is still small. Deep layers get smaller scales to prevent variance blowup. As documented in `transformer.py:19-25`, this is "strictly more principled" because it accounts for actual depth rather than total depth.

### 2.11 Sliding Window Attention

The `_build_sliding_window_mask` function (`attention.py:62-101`) constructs a combined causal + window + global mask:

```python
# attention.py:89-99
causal = k_idx <= q_idx
in_window = k_idx > (q_idx - window_size)
is_global = k_idx < n_global_tokens
mask = (causal & in_window) | is_global
```

This supports nanochat's SSSL (Short-Short-Short-Long) attention pattern: three local-window layers followed by one full-attention layer. In nanochat-jax, this is configured per-model via `sliding_window_size` and `n_global_tokens` in `ModelConfig` (`model_config.py:226-243`).

**Comparison with nanochat:** The original uses a tiled SSSL pattern compiled into Flash Attention's mask. Nanochat-jax materializes the full `[1, 1, S, S]` boolean mask and applies it via `jnp.where` (`attention.py:304`), which is memory-equivalent to full attention for the mask tensor itself. The savings come from sparsity in the softmax output, not from avoiding materialization.

### 2.12 KV Cache

The `KVCache` dataclass (`kv_cache.py:9-53`) uses pre-allocated arrays with `jax.lax.dynamic_update_slice` for XLA-compatible updates:

```python
# kv_cache.py:37-43
keys = jax.lax.dynamic_update_slice(self.keys, new_k, (0, 0, self.position, 0))
values = jax.lax.dynamic_update_slice(self.values, new_v, (0, 0, self.position, 0))
return KVCache(keys=keys, values=values, position=self.position + seq_new)
```

Layout is `[batch, n_kv_heads, max_len, d_head]` (`kv_cache.py:16-17`), matching nanochat's `(n_layers, B, T_max, H, D)` layout at the per-layer level.

**Divergence:** Nanochat's KV cache is FA3-compatible (contiguous memory layout optimized for Flash Attention's tiled access pattern). Nanochat-jax's cache works with the explicit attention implementation but would need layout changes for Pallas-based Flash Attention kernels.

---

## 3. Optimizer and Training Loop

### 3.1 Muon Optimizer

Muon (Momentum + Orthogonalization by Newton-Schulz) is implemented in `muon.py:1-273`. The core algorithm for 2D weight matrices:

1. **Momentum accumulation** (`muon.py:191`): `m_new = momentum * m + g`
2. **Nesterov look-ahead** (`muon.py:194`): `g_eff = g + momentum * m_new`
3. **Newton-Schulz orthogonalization** (`muon.py:200`): `g_orth = newton_schulz_orthogonalize(g_eff)`
4. **Shape-aware scaling** (`muon.py:204`): `scale = max(1, sqrt(m/n))`
5. **Decoupled weight decay** (`muon.py:211-215`): `update = update + lr * wd * p`

The Newton-Schulz iteration (`muon.py:63-109`) uses the cubic polynomial:

```python
# muon.py:100-103
def ns_step(_, X):
    A = X @ X.T        # (m_eff, m_eff)
    return 1.5 * X - 0.5 * (A @ X)

G = jax.lax.fori_loop(0, steps, ns_step, G)
```

The coefficients (1.5, -0.5) satisfy the cubic iteration for the polar decomposition: given `X_0 = G/||G||_F`, the iteration converges to the unitary polar factor `U` such that `G = U * P` where `P` is positive semidefinite.

**Critical divergence from nanochat:**

| Aspect | nanochat | nanochat-jax |
|---|---|---|
| NS variant | Polar Express (quintic) | Cubic (standard) |
| NS coefficients | Higher-order polynomial for faster convergence | (1.5, -0.5) |
| Variance reduction | NorMuon for 2D params | None |
| 1D fallback | Fused compiled AdamW | Raw SGD with momentum |
| Distributed | DistMuonAdamW with ZeRO-2 async comms | Not implemented |

The nanochat Polar Express uses a quintic polynomial that achieves the same convergence quality in fewer iterations than the cubic variant. NorMuon adds variance reduction to the orthogonalization, which stabilizes training for large weight matrices. Nanochat-jax's simpler cubic iteration (`muon.py:100-103`) requires more iterations (default 5 via `jax.lax.fori_loop`) to achieve comparable orthogonality.

The `jax.lax.fori_loop` usage (`muon.py:104`) is significant: it compiles the iteration into a single XLA while-loop, avoiding Python-level loop overhead and enabling the iteration to run entirely on-device. This is one of the few places where nanochat-jax leverages JAX's functional loop primitives.

**1D parameter handling:** Nanochat uses a fused compiled AdamW for 1D parameters (biases, embedding rows, normalization params). Nanochat-jax falls back to plain SGD with momentum for 1D params (`muon.py:207-208`): `update = lr * g_eff`. This is a significant degradation for parameters like embedding tables, where Adam's adaptive learning rates per-element are important for handling the varying gradient magnitudes across vocabulary entries. [HYPOTHESIS: This 1D fallback difference may partially explain the weak scaling exponent (alpha=0.027) observed in nanochat-jax's scaling experiments, as embedding parameters dominate at small model scales.]

### 3.2 Optimizer Construction

The `build_optimizer` function (`optimizer.py:36-111`) dispatches between Muon and AdamW:

```python
# optimizer.py:64-81
if cfg.optimizer == "muon":
    optimizer = build_muon_optimizer(
        learning_rate=schedule,
        momentum=cfg.muon_momentum,      # default: 0.95
        nesterov=cfg.muon_nesterov,      # default: True
        ns_steps=cfg.muon_ns_steps,      # default: 5
        weight_decay=cfg.muon_weight_decay,  # default: 0.01
        grad_clip_norm=cfg.grad_clip_norm,   # default: 1.0
    )
```

Weight decay masking (`optimizer.py:20-34`) excludes normalization params, biases, embeddings, and nanochat-specific parameters (alpha_attn, alpha_ffn, raw_alpha, raw_beta):

```python
# optimizer.py:28-29
exclude = ("norm", "bias", "embed", "gamma", "beta", "table",
           "alpha_attn", "alpha_ffn", "raw_alpha", "raw_beta")
```

The complete optimizer chain (`muon.py:263-272`) is:

```python
return optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    muon(learning_rate=learning_rate, ...),
)
```

**Gradient clipping divergence:** Nanochat does not use gradient clipping -- Muon's orthogonalization inherently normalizes gradient magnitudes since the NS iteration produces a matrix with unit singular values. Nanochat-jax adds `optax.clip_by_global_norm(1.0)` (`muon.py:264`), which clips before orthogonalization. This can interfere with the Newton-Schulz convergence: if the input gradient is already clipped to a small norm, the NS iteration's initial normalization (`G / ||G||_F`) may amplify noise. [HYPOTHESIS: Removing gradient clipping in nanochat-jax would better match nanochat's training dynamics and potentially improve scaling behavior.]

### 3.3 Learning Rate Schedule

The `build_lr_schedule` function (`scheduler.py:5-46`) constructs a three-phase schedule:

```python
# scheduler.py:31-46
warmup = optax.linear_schedule(init_value=0.0, end_value=learning_rate,
                                transition_steps=warmup_steps)
cosine = optax.cosine_decay_schedule(init_value=learning_rate,
                                      decay_steps=max(decay_steps, 1),
                                      alpha=min_lr_ratio)
return optax.join_schedules(schedules=[warmup, cosine], boundaries=[warmup_steps])
```

**Divergence from nanochat:** The original uses linear warmup (40 steps) -> constant peak -> linear warmdown (65% of training) with a floor at 5% of peak. Nanochat-jax uses cosine decay (the standard post-GPT-3 schedule) with `min_lr_ratio=0.1` (10% floor) (`training_config.py:66`) and `warmup_steps=2000` (`training_config.py:53`).

The shape difference matters for scaling law experiments:

- **nanochat:** Constant-then-linear produces a flat learning rate for the majority of training, with a sharp linear decline. This is optimal for fixed-budget training where the model trains for exactly one "epoch" of the compute budget.
- **nanochat-jax:** Cosine decay starts declining immediately after warmup, spending more time at lower learning rates. For short training runs (typical in scaling experiments), this means the effective average learning rate is lower than nanochat's.

The LR scaling rule also differs: nanochat scales LR as `sqrt(B/B_ref)` where B is batch size. Nanochat-jax has no batch-size-aware LR scaling.

### 3.4 Training Loop

The `Trainer` class (`trainer.py:32-246`) orchestrates training with JIT-compiled steps:

```python
# trainer.py:103-141
@staticmethod
@nnx.jit
def _train_step_jit(model, optimizer, batch):
    def loss_fn(model):
        logits, _ = model(batch["input_ids"], deterministic=False)
        labels_src = batch["labels"] if "labels" in batch else batch["input_ids"]
        loss, metrics = cross_entropy_loss(
            logits=logits[:, :-1, :], labels=labels_src[:, :-1])
        return loss, metrics

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    grad_norm = optax.global_norm(jax.tree.leaves(grads))
    optimizer.update(model, grads)
    return { ... }
```

Key design choices:
- `@nnx.jit` on the static method enables full XLA compilation of the train step, including loss computation, gradient computation, and optimizer update
- `nnx.value_and_grad` with `has_aux=True` returns both the loss metrics and gradients in a single backward pass
- Grad norm computation happens after the backward pass but before the optimizer update, matching standard monitoring practice

**Loss function** (`loss.py:13-74`) uses `optax.softmax_cross_entropy_with_integer_labels` for the standard case and supports label smoothing and z-loss (PaLM-style logit regularization):

```python
# loss.py:45-48
ce = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits.reshape(-1, vocab_size),
    labels=safe_labels.reshape(-1),
).reshape(logits.shape[:2])
```

The `IGNORE_INDEX = -100` convention (`loss.py:10`) matches PyTorch's `CrossEntropyLoss` default, ensuring compatibility with the data pipeline.

### 3.5 Weight Decay Masking

The weight decay mask (`optimizer.py:20-34`) is a significant design element that determines which parameters are regularized:

```python
# optimizer.py:28-29
exclude = ("norm", "bias", "embed", "gamma", "beta", "table",
           "alpha_attn", "alpha_ffn", "raw_alpha", "raw_beta")
```

This excludes:
- **Normalization parameters**: RMSNorm is parameterless so this is a no-op, but provides safety if LayerNorm is substituted
- **Bias terms**: Not used in the default config (`use_bias=False`, `model_config.py:147`), but protected
- **Embedding tables**: Both token embeddings and value embeddings are excluded from decay
- **Per-layer scalars**: `alpha_attn`, `alpha_ffn` are not decayed, allowing them to grow freely
- **Smear/Backout gates**: `raw_alpha`, `raw_beta` are not decayed, preserving the sigmoid-gated initialization

**Comparison with nanochat:** The original applies weight decay only through the Muon optimizer's decoupled mechanism, which operates uniformly on all parameters passed to it. The explicit exclusion list in nanochat-jax is more conservative and matches the LLaMA/GPT best practice of only decaying weight matrices.

For AdamW, the mask is applied via optax's `mask` parameter (`optimizer.py:92-94`):

```python
mask=lambda params: jax.tree_util.tree_map_with_path(
    _weight_decay_mask, params
),
```

This uses `jax.tree_util.tree_map_with_path` to inspect each parameter's path in the pytree and return a boolean mask. The path-based inspection is the JAX equivalent of iterating over PyTorch's `named_parameters()`.

### 3.6 Checkpointing

The `CheckpointManager` class (`checkpoint.py:18-249`) uses pickle-based serialization rather than Orbax (despite Orbax being listed in dependencies):

```python
# checkpoint.py:127-128
model_state = nnx.state(model)
model_np = self._state_to_numpy(model_state)
with open(step_dir / "model.pkl", "wb") as f:
    pickle.dump(model_np, f, protocol=4)
```

State conversion handles JAX's typed PRNG keys (`checkpoint.py:80-88`):

```python
if "key" in dtype_str:
    return np.asarray(jax.random.key_data(x))
return np.asarray(x)
```

Restoration uses `nnx.update` for in-place state replacement (`checkpoint.py:182`):

```python
nnx.update(model, model_state)
```

The checkpoint rotation policy (`checkpoint.py:64-71`) keeps the last `keep_last_n` checkpoints (default 3, `training_config.py:137`) and creates a `latest` symlink.

---

## 4. Systems and Infrastructure

### 4.1 Module Organization

Nanochat-jax is organized into 8 top-level packages under `src/nanochat/`:

| Package | Files | Responsibility |
|---|---|---|
| `model/` | 9 | Transformer architecture (attention, block, embeddings, FFN, norms, value_embeddings, token_mixing, param_count, transformer) |
| `training/` | 7 | Optimizer, scheduler, loss, trainer, checkpoint, distributed, muon |
| `inference/` | 5 | Engine, KV cache, sampling, chat session |
| `config/` | 5 | Pydantic configs (model, training, data, scaling) |
| `data/` | 5 | Dataset, loader, packing, preprocessing |
| `scaling/` | 4 | Runner, analysis (power law fitting, Chinchilla), visualization |
| `tokenizer/` | 4 | Base, BPE (tiktoken), char tokenizer |
| `evaluation/` | 4 | Evaluator, metrics, throughput |

This is substantially more modular than nanochat, which concentrates the model in `gpt.py`, the optimizer in `optim.py`, and inference in `engine.py`. The modular structure in nanochat-jax enables independent testing (30 test files covering 180 tests) but adds import complexity and cross-module coupling.

### 4.2 Configuration System

All configuration uses Pydantic v2 models (`model_config.py:25-420`, `training_config.py:15-234`):

```python
# model_config.py:25-38
class ModelConfig(BaseModel):
    vocab_size: int = Field(default=32000, ge=1, le=1_000_000)
    d_model: int = Field(default=512, ge=1, le=65536)
    n_layers: int = Field(default=6, ge=1, le=1024)
    # ...
```

Key validators enforce architectural constraints:
- `n_heads % n_kv_heads == 0` (`model_config.py:250-257`)
- `d_model % n_heads == 0` (`model_config.py:259-267`)
- Auto-computed `d_ff` based on FFN type (`model_config.py:269-284`)
- Computed fields for `d_head`, `n_groups`, `is_gqa` (`model_config.py:290-306`)

Model presets (`model_config.py:327-372`):

| Preset | d_model | n_layers | n_heads | n_kv_heads | max_seq_len | ~Params |
|---|---|---|---|---|---|---|
| nano | 128 | 4 | 4 | 4 | 64 | ~886K |
| small | 512 | 6 | 8 | 8 | 2048 | ~68M |
| medium | 1024 | 12 | 16 | 8 | 2048 | ~237M |
| large | 2048 | 24 | 32 | 8 | 4096 | ~1.25B |
| xlarge | 4096 | 32 | 32 | 8 | 4096 | ~6.03B |

**Comparison with nanochat:** The original uses a single `--depth` parameter that auto-computes width, heads, LR, batch size, and decay via muP-style transfer and Power Lines scaling. Nanochat-jax requires explicit preset selection or manual specification via `from_depth` (`model_config.py:374-420`), which accepts `n_layers` and `d_model` but does not auto-tune hyperparameters.

### 4.3 JIT Compilation Strategy

The nanochat-jax codebase uses `@nnx.jit` at three key boundary points:

1. **Trainer step** (`trainer.py:104`): `@nnx.jit` on `_train_step_jit` -- compiles the full forward pass, loss computation, backward pass, grad norm computation, and optimizer update into a single XLA program.

2. **Scaling runner step** (`runner.py:145`): `@nnx.jit` on the inner `train_step` function -- same scope as the trainer.

3. **Throughput benchmark step** (`throughput.py:58`): `@nnx.jit` on the benchmark `step` function.

The decision to JIT at the train-step level (rather than per-layer or per-module) has important implications:

- **Compilation cost:** The first call triggers XLA compilation of the entire train step, which can take 30-120 seconds for medium-sized models. Subsequent calls with the same input shapes reuse the cached compiled program.
- **Optimization scope:** XLA can fuse operations across the entire forward + backward pass. This enables optimizations like operator fusion (combining matmul + bias + activation), memory layout optimization, and cross-layer buffer reuse.
- **Retracing risk:** `@nnx.jit` on a static method avoids common retracing issues where Python-level state changes trigger recompilation. The batch dict's array shapes must remain constant across steps.

**Comparison with nanochat:** PyTorch 2.x uses `torch.compile()` which applies TorchInductor-based compilation. Nanochat compiles optimizer steps individually (`compiled=True` in optimizer creation) and relies on Flash Attention's handwritten CUDA kernels rather than compiler-generated code. The JAX approach of compiling the entire train step gives XLA more optimization scope but incurs a higher upfront compilation cost.

Notably absent from nanochat-jax: `jax.vmap`, `jax.pmap`, and `jax.lax.scan`. The lack of `jax.lax.scan` for the transformer layer stack is a missed optimization: scanning over layers with a carried state (the hidden representation) would allow XLA to overlap computation and memory operations across layers, potentially improving throughput by 5-15% for deep models.

### 4.4 Distributed Training

The `distributed.py` module (`distributed.py:1-87`) provides mesh creation and partition specs but no functional distributed training:

```python
# distributed.py:14-62
def create_device_mesh(mesh_shape=None, axis_names=("data", "model")):
    devices = jax.devices()
    n_devices = len(devices)
    # Auto-factor: prefer more data parallelism
    # ...
    return Mesh(device_array, axis_names=axis_names)
```

Partition specs (`distributed.py:65-87`) follow Megatron-LM conventions:
- Embeddings: shard along vocab dimension
- Q/K/V projections: shard along head dimension (output)
- Out projection: shard along head dimension (input)
- FFN gate/up: shard along d_ff (output)
- FFN down: shard along d_ff (input)
- Norms: replicated

**Gap analysis:** These specs are defined but never applied. The `Trainer` class does not use `jax.sharding`, `jax.experimental.shard_map`, or any multi-device primitives. The training loop runs entirely on the default device. This is the single largest capability gap versus nanochat, which implements custom `DistMuonAdamW` with ZeRO-2-style gradient sharding across multiple GPUs.

### 4.5 Inference Engine

The `InferenceEngine` class (`engine.py:17-226`) supports both batch and streaming generation:

- **Prefill phase** (`engine.py:131`): Process full prompt in parallel, populate KV caches
- **Decode phase** (`engine.py:136-159`): Generate one token at a time using cached K/V

Sampling strategies (`sampling.py:1-106`) include greedy, temperature, top-k, top-p, and a combined sampler with repetition penalty:

```python
# sampling.py:61-106
def combined_sample(logits, rng, temperature=1.0, top_k=0, top_p=1.0,
                    repetition_penalty=1.0, generated_ids=None):
    # repetition penalty -> top-k -> top-p -> temperature -> sample
```

Streaming generation (`engine.py:177-226`) yields decoded text fragments via a Python generator:

```python
# engine.py:218-219
fragment = self.tokenizer.decode([tok_id], skip_special_tokens=True)
if fragment:
    yield fragment
```

**Comparison with nanochat:** The original provides a full chat system with CLI (`chat_cli.py`), web UI with FastAPI + streaming (`chat_web.py`, `ui.html`), and tool use (calculator). Nanochat-jax provides the engine and a `ChatSession` class (`chat.py:11-79`) for multi-turn conversation management, but no web interface or tool use.

### 4.6 Data Pipeline

Two loader modes (`loader.py:1-200`):

1. **Standard loader** (`loader.py:37-92`): Sliding-window batches from HDF5 `TokenDataset`, with epoch-based shuffling
2. **Packed loader** (`loader.py:99-200`): BOS-aligned best-fit packing with 2D document-aware attention masks for ~100% token utilization

#### 4.6.1 TokenDataset and HDF5 Memory Mapping

The `TokenDataset` class (`dataset.py:11-77`) uses HDF5 memory mapping for efficient access to large corpora:

```python
# dataset.py:28-32
self._file = h5py.File(self.path, "r")
self._tokens = self._file["tokens"]
total_tokens = len(self._tokens)
```

Windows are sliced without copying (`dataset.py:59-67`):

```python
window = self._tokens[start:end].astype(np.int32)
input_ids = window[:-1]      # [seq_len]
labels = window[1:]          # [seq_len] shifted right by 1
```

The label convention -- `labels[t] = tokens[t+1]` -- means labels are pre-shifted at data loading time. This is critical for the loss computation in the trainer, which pairs `logits[:, :-1, :]` with `labels[:, :-1]` (`trainer.py:122-125`). The data contract is documented inline: "labels = window[1:] (shifted by 1 at load time)" (`trainer.py:117`).

Train/validation splitting uses a fraction-based approach (`dataset.py:36-44`) with `val_fraction=0.005` (0.5%) by default, splitting from the end of the token stream. This is simpler than nanochat's per-document splitting but risks information leakage for documents that straddle the split boundary.

#### 4.6.2 Document-Aware Packing

The packing module (`packing.py:1-260`) implements the standard "packing with document masking" approach from LLaMA 2 and Mistral. The algorithm (`packing.py:74-173`):

1. Wrap each document with BOS/EOS markers: `[BOS, tok1, ..., tokN, EOS]`
2. Concatenate into a flat stream
3. Slide fixed-length windows over the stream
4. For each window, build a 2D causal + document-separation mask

The mask construction (`packing.py:176-198`) is elegant:

```python
# packing.py:193-198
causal = np.tril(np.ones((seq_len, seq_len), dtype=bool))
same_doc = doc_ids[:, None] == doc_ids[None, :]  # [S, S]
return causal & same_doc
```

The `same_doc` matrix uses broadcasting to create an `[S, S]` boolean matrix where position (q, k) is True only if both positions belong to the same document. Combined with the causal constraint, this prevents cross-document attention contamination.

**Memory cost:** Each packed window stores an `[S, S]` boolean mask. For seq_len=2048, this is 4MB per window (bool = 1 byte). For a corpus producing 100K windows, the total mask storage is ~400GB, which is impractical. The docstring (`packing.py:32`) acknowledges this: "Mask storage: O(n_windows * seq_len^2) in bool -- use sparse rep for large seq_len." For the nano and small presets (seq_len 64-2048), this is manageable. For production-scale training, a sparse representation or on-the-fly mask computation would be needed.

The packed loader generates 3D attention masks for batches (`loader.py:185-187`):

```python
attention_mask = jnp.array(
    np.stack([w.attention_mask for w in batch_windows])
)  # [B, S, S]
```

This 2D per-example mask prevents cross-document attention contamination when multiple documents are packed into a single training sequence.

**Comparison with nanochat:** The original uses a similar packing approach but integrates it with Flash Attention's mask support, which can consume the 2D mask without materializing the full attention matrix. In nanochat-jax, the 2D mask is applied via `jnp.where` to the materialized score matrix, doubling the memory footprint compared to the standard 1D mask path.

### 4.7 Tokenizer

The `BPETokenizer` class (`bpe.py:8-86`) wraps tiktoken's `cl100k_base` encoding with custom special tokens:

```python
# bpe.py:16-24
SPECIAL_TOKENS = {
    "<|bos|>": 100277, "<|eos|>": 100278, "<|pad|>": 100279,
    "<|user|>": 100280, "<|assistant|>": 100281,
    "<|system|>": 100282, "<|end|>": 100283,
}
```

**Critical divergence:** Nanochat trains its own BPE tokenizer with vocab_size=32,768 (2^15) using GPT-4 style patterns. Nanochat-jax uses tiktoken's pre-trained cl100k_base with vocab_size=100,284. This 3x vocabulary difference affects:
- Embedding parameter count (3x more in nanochat-jax for the same d_model)
- Compression ratio (cl100k_base is optimized for English web text and may have different tokens/character ratios)
- Scaling law comparisons (the "N" in L(N) includes embedding parameters, which are disproportionately large with a 100k vocab)

### 4.8 Throughput Benchmarking

The `benchmark_training_throughput` function (`throughput.py:29-99`) provides a standardized protocol for measuring training speed:

```python
# throughput.py:58-66
@nnx.jit
def step(model, optimizer, batch):
    def loss_fn(model):
        logits, _ = model(batch["input_ids"], deterministic=False)
        loss, _ = cross_entropy_loss(logits[:, :-1], batch["labels"][:, 1:])
        return loss
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss
```

The benchmark uses `jax.effects_barrier()` (`throughput.py:71`, `throughput.py:77`) after warmup and measurement phases to ensure all asynchronous dispatches complete before timing. This is critical for accurate throughput measurement: JAX's asynchronous dispatch means that `step()` returns immediately after enqueueing the computation, and without an explicit barrier, the measured time would reflect dispatch latency (~170us) rather than actual computation time.

The `ThroughputReport` dataclass (`throughput.py:19-26`) includes tokens/second, samples/second, MFU, and average step time. Notably, `peak_memory_gb` is always 0.0 (`throughput.py:93`) with the comment "JAX doesn't easily expose this." This is a known limitation: JAX's memory allocator does not provide a simple peak-memory query like PyTorch's `torch.cuda.max_memory_allocated()`. The `jax.local_devices()[0].memory_stats()` API exists but is backend-specific and not always available.

### 4.9 Observability Stack

Nanochat-jax integrates multiple observability tools (declared in `pyproject.toml:27-31`):
- **structlog** (`>= 24.4.0`): Structured logging throughout (every module imports `structlog.get_logger()`)
- **Weights & Biases** (`wandb >= 0.17.0`): Experiment tracking
- **TensorBoard** (`>= 2.17.0`): Training visualization
- **Prometheus** (`prometheus-client >= 0.20.0`): Metrics export
- **Rich** (`>= 13.7.0`): Terminal formatting
- **FastAPI + Uvicorn** (`>= 0.111.0`): Inference server stub

This is a significantly richer observability stack than nanochat, which relies on standard Python logging and optional W&B. The structlog integration is particularly thorough: every constructor logs its initialization parameters (e.g., `transformer.py:112-129`, `attention.py:166-175`, `feedforward.py:88-93`), creating a structured audit trail of model construction. In production, this makes it possible to reconstruct the exact model architecture from log output alone.

### 4.10 Dependency Analysis

The project's dependencies (`pyproject.toml:15-38`) reveal the engineering philosophy:

**Core ML stack:**
- `jax >= 0.4.35`, `jaxlib >= 0.4.35`, `flax >= 0.10.0`, `optax >= 0.2.4`: The JAX ecosystem core
- `orbax-checkpoint >= 0.9.0`: Listed but not used in the checkpoint implementation (which uses pickle)

**Data processing:**
- `tiktoken >= 0.7.0`: BPE tokenization (GPT-4 compatible)
- `datasets >= 2.20.0`: HuggingFace datasets (available but not the primary data path)
- `h5py >= 3.11.0`: Memory-mapped token datasets
- `numpy >= 1.26.0`: Array operations

**Analysis:**
- `scipy >= 1.14.0`: Power law curve fitting
- `pandas >= 2.2.0`: Scaling experiment result analysis
- `matplotlib >= 3.9.0`, `seaborn >= 0.13.0`: Visualization

**Configuration:**
- `pydantic >= 2.8.0`: Validated configuration models
- `hydra-core >= 1.3.2`, `omegaconf >= 2.3.0`: Hierarchical configuration (available but Pydantic is primary)

**Total: 21 runtime dependencies.** This is a heavier dependency footprint than nanochat (which has ~5 core dependencies: PyTorch, tiktoken, numpy, requests, datasets). The additional dependencies enable a more complete research workflow but increase the surface area for version conflicts and installation issues, particularly on systems where JAX and CUDA versions must align precisely.

---

## 5. Scaling Law Methodology

### 5.1 Experiment Framework

The `ScalingRunner` class (`runner.py:80-353`) orchestrates three experiment types:

- **scale_n** (`runner.py:274-292`): Fix token budget, vary model size. Reveals L(N) at constant D.
- **scale_d** (`runner.py:294-312`): Fix model size, vary token budget. Reveals L(D) at constant N.
- **scale_c** (`runner.py:315-343`): Vary both N and D along the Chinchilla compute frontier (C = 6ND = const). Reveals the envelope of optimal loss vs. compute.

Each run (`runner.py:102-237`) follows the same protocol:
1. Build model and optimizer
2. Train for the specified number of steps
3. Record training/validation losses at regular intervals
4. Compute throughput and MFU
5. Serialize results to JSON

The JIT-compiled train step in the runner (`runner.py:145-158`) mirrors the trainer's pattern:

```python
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(m):
        logits, _ = m(batch["input_ids"], deterministic=False)
        loss, metrics = cross_entropy_loss(
            logits[:, :-1, :], batch["labels"][:, :-1])
        return loss, metrics
    (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return loss
```

### 5.2 Model Size Search

The `_build_model_for_target_params` function (`runner.py:43-77`) searches over a grid of (d_model, n_layers) combinations:

```python
# runner.py:56-66
for n_layers in [2, 3, 4, 5, 6, 8, 10, 12]:
    for d_model in [32, 48, 64, 96, 128, 192, 256, 384, 512, 768]:
        approx = 12 * n_layers * d_model * d_model
        delta = abs(approx - target_params)
```

The approximation `N ~ 12 * L * d^2` captures the dominant parameter contribution from attention (4d^2 per layer for Q/K/V/O projections) and FFN (8d^2 per layer for up+down with 4x expansion), ignoring vocabulary parameters. This is adequate for scaling law research where the relevant parameter count excludes embeddings.

### 5.3 Power Law Fitting

The `fit_power_law` function (`analysis.py:11-70`) fits L = a * x^(-alpha) via log-space linear regression:

```python
# analysis.py:36-38
log_x = np.log(xs)
log_y = np.log(ys)
coeffs = np.polyfit(log_x, log_y, 1)
alpha = -coeffs[0]
a = np.exp(coeffs[1])
```

Bootstrap confidence intervals (`analysis.py:47-59`) use 1000 resamples with percentile-based CI (5th and 95th percentiles):

```python
# analysis.py:49-55
for _ in range(bootstrap_n):
    idx = rng.choice(len(xs), size=len(xs), replace=True)
    c = np.polyfit(log_x[idx], log_y[idx], 1)
    alphas.append(-c[0])
```

### 5.4 Chinchilla-Optimal Computation

The `chinchilla_optimal` function (`analysis.py:73-110`) implements the Hoffmann et al. (2022) compute-optimal allocation:

```python
# analysis.py:91-98
gamma = alpha / (alpha + beta)   # exponent for D*
delta = beta / (alpha + beta)    # exponent for N*
for C in compute_budgets:
    N_star = int((C / 6.0) ** delta * 1e6 ** (1 - delta))
    D_star = int(C / (6.0 * max(N_star, 1)))
```

Default parameters (`analysis.py:76-80`): `a_n=406.4`, `a_d=410.7`, `alpha=0.34`, `beta=0.28`, `irreducible_loss=1.69`. These match the Chinchilla paper's published values for the three-term loss model L(N, D) = E + A/N^alpha + B/D^beta.

### 5.5 Synthetic Data Limitation

The scaling runner's synthetic data path (`runner.py:163-173`) generates random tokens:

```python
# runner.py:167-173
ids = jax.random.randint(k, (train_cfg.batch_size, model_cfg.max_seq_len + 1),
                          0, model_cfg.vocab_size)
return {"input_ids": ids[:, :-1], "labels": ids[:, 1:]}
```

Random token sequences have entropy equal to `log(vocab_size)`, which is `log(256) = 5.55` nats for the nano preset. Since there are no learnable patterns, the theoretical minimum loss a model can achieve is the entropy of the uniform distribution, and any improvement below that would require overfitting to the random data generator's seed.

This fundamentally limits the validity of scaling law measurements: the power law L(N) = a * N^(-alpha) assumes the model is learning real statistical regularities that become easier to capture with more parameters. With random data, increasing N cannot improve loss beyond the data entropy bound, so the fitted exponent alpha converges to 0 as training progresses.

**Recommendation:** Scaling experiments should use real text data (TinyShakespeare, OpenWebText, or similar) or at minimum a structured synthetic dataset (e.g., algorithmic sequences, n-gram models) where the data has learnable structure at multiple scales.

### 5.6 FLOP Estimation and MFU

The `estimate_flops_per_token` function (`param_count.py:187-293`) computes per-token FLOPs accounting for the model's specific attention layout and FFN type:

```python
# param_count.py:242-269
flops_q_proj = 2 * d_model * (n_heads * d_head)
flops_k_proj = 2 * d_model * (n_kv_heads * d_head)
# ...
is_gated = cfg.ffn_type in ("swiglu", "geglu")
n_ffn_projections = 3 if is_gated else 2
flops_ffn_per_layer = n_ffn_projections * 2 * d_model * d_ff
```

MFU (`param_count.py:328-398`) uses the 3x rule (1x forward + 2x backward):

```python
# param_count.py:382-387
flops_per_token = 3.0 * estimate_flops_per_token(cfg)
actual_flops = throughput_tps * flops_per_token
mfu = actual_flops / peak_flops
```

The default peak FLOPS (`runner.py:22`) is set to 9.0e12, corresponding to an RTX 3050's FP32 peak. This is appropriate for consumer-GPU scaling experiments but would need adjustment for A100 (312e12 bf16) or H100 (990e12 bf16).

### 5.7 Reported Results and Gap Analysis

The reported scaling result -- `L = 3.29 * N^(-0.027)` at 600 steps on TinyShakespeare -- yields an exponent alpha=0.027, which is significantly below the Kaplan et al. (2020) value of alpha~0.076 and the Chinchilla value of alpha~0.34.

**Potential explanations for the weak exponent:**

1. **Insufficient training duration.** 600 steps with batch_size=8 and seq_len=64-128 yields only ~300K-600K tokens. Kaplan et al. used billions of tokens. At such low token counts, all models are severely undertrained and the power law regime has not been reached.

2. **Synthetic data.** The runner uses random tokens when no data loader is provided (`runner.py:163-173`). Random token sequences have maximum entropy and no learnable structure, so models cannot improve loss by learning patterns -- only by memorizing token frequencies.

3. **Optimizer mismatch.** The scaling runner defaults to AdamW (`runner.py:285-286`), not Muon. The 1D fallback in Muon (plain SGD vs AdamW) further weakens optimization of embedding-heavy small models.

4. **LR schedule mismatch.** Cosine decay starting from warmup penalizes short runs more than nanochat's constant-then-linear schedule.

5. **Softcap and grad clipping interaction.** The wider softcap (30 vs 15) combined with gradient clipping (absent in nanochat) may alter the effective learning dynamics.

[HYPOTHESIS: Running the same experiment grid with (a) 10x more tokens, (b) real text data, (c) Muon optimizer, and (d) nanochat-matched LR schedule would bring the exponent closer to alpha~0.07-0.08.]

### 5.8 Comparison with Nanochat's Scaling Infrastructure

Nanochat's `runs/scaling_laws.sh` provides a single script that sweeps FLOP budgets crossed with depths, using the `--depth` parameter to auto-compute everything. The key difference is automation:

| Feature | nanochat | nanochat-jax |
|---|---|---|
| Hyperparameter auto-tuning | --depth computes width, heads, LR, batch, decay | Manual or preset selection |
| LR scaling | sqrt(B/B_ref) batch-size scaling | Fixed LR per config |
| Compute budgets | FLOP-based sweep | Token-budget or FLOP-based |
| Data | Real web text (DCLM) | Synthetic or TinyShakespeare |
| Power Lines integration | Native | Not implemented |
| T_epoch scaling | Native (adjusts schedule to data size) | Not implemented |

---

## 6. Numerical Fidelity and Precision

### 6.1 Precision Strategy

Both codebases use mixed precision, but with different strategies:

**Nanochat (PyTorch):**
- Custom `Linear` class that casts weights to bf16 at forward-pass time
- Parameters stored in fp32 (master weights)
- Optional fp16 + `GradScaler` for pre-Ampere GPUs
- Optional FP8 via `torch._scaled_mm` (~150 LOC)
- Flash Attention 3 operates in bf16 natively on Hopper

**Nanochat-jax:**
- `param_dtype` always float32 (`training_config.py:166`)
- `compute_dtype` configurable as float32/bfloat16/float16 (`training_config.py:159-163`)
- Explicit dtype management in attention: scores computed in float32 (`attention.py:273-274`), context accumulated in float32 (`attention.py:315`), then cast back to input dtype (`attention.py:321`)
- QK normalization promotes to float32 then casts back (`attention.py:242-243`)
- RMSNorm promotes to float32 for variance computation (`norms.py:87-93`)

The explicit float32 promotions in nanochat-jax are necessary because JAX does not automatically manage precision boundaries the way PyTorch's autocast does. Each promotion-demotion pair adds a type cast operation, which XLA typically fuses away but which can affect numerical results at the boundary.

### 6.2 Attention Score Computation

The attention score computation path reveals a key numerical difference:

**Nanochat-jax** (`attention.py:272-275`):
```python
scores = jnp.matmul(
    q.astype(jnp.float32),
    jnp.transpose(k_exp.astype(jnp.float32), (0, 1, 3, 2))
) * self.attn_scale
```

This promotes Q and K to float32 before the matmul, computes the full `[B, H, S, S]` score matrix, then scales. The multiplication by `attn_scale` (= 1.2/sqrt(d_head)) happens after the matmul.

**Nanochat (PyTorch):** Flash Attention 3 computes the scaled dot-product in a single fused kernel with tiled accumulation in bf16, using online softmax to avoid materializing the full score matrix. The scale factor is passed as a parameter to the kernel.

The numerical implications:
1. **Float32 matmul in nanochat-jax** produces ~7 decimal digits of precision in scores, while bf16 in Flash Attention produces ~3.5 digits. For long sequences where attention scores span a wide range, the float32 computation is more numerically stable.
2. **Post-matmul scaling** in nanochat-jax means the scale factor does not affect the precision of the matmul itself. Flash Attention applies scaling during the tiled computation, which can affect the online softmax's numerical properties.
3. **Full score materialization** in nanochat-jax means every (query, key) pair contributes to the backward pass equally. Flash Attention's tiled recomputation during the backward pass can produce slightly different gradients due to non-associativity of floating-point addition.

### 6.3 Softcap Numerical Behavior

The logit softcap `cap * tanh(x/cap)` (`attention.py:284`) has different numerical behavior at cap=15 vs cap=30:

At cap=15:
- tanh(1) = 0.762, so scores at magnitude 15 are compressed to 15*0.762 = 11.4
- tanh(2) = 0.964, so scores at magnitude 30 are compressed to 15*0.964 = 14.5
- Gradient: `1 - tanh(x/cap)^2`, which is 0.42 at |x|=15 and 0.07 at |x|=30

At cap=30:
- Scores at magnitude 15 are barely compressed: 30*tanh(0.5) = 30*0.462 = 13.9
- Scores at magnitude 30 are compressed to 30*tanh(1) = 30*0.762 = 22.9
- Gradient at |x|=15 is 0.786, at |x|=30 is 0.42

The cap=30 setting in nanochat-jax preserves more gradient flow for moderate-magnitude scores, which is friendlier for training stability at the cost of less aggressive outlier suppression. With QK normalization active, attention scores are already bounded, so the softcap primarily acts as a safety net rather than an active regularizer. [HYPOTHESIS: With QK normalization enabled (default in both codebases), the softcap value has minimal practical impact on training dynamics, and the divergence is cosmetic rather than functional.]

### 6.4 Newton-Schulz Orthogonalization Precision

The NS iteration in `muon.py:87-108` operates entirely in float32:

```python
# muon.py:87
G = G.astype(jnp.float32)
```

The iteration's convergence depends on the initial conditioning. After F-norm normalization (`muon.py:92`), the singular values of G are bounded by `[0, sqrt(min(m,n))]`. The cubic iteration `X <- 1.5*X - 0.5*(X@X^T@X)` converges quadratically when all singular values are in `(0, sqrt(3))`.

After 5 iterations, the error `max|U^T@U - I|` is typically below 0.05 for well-conditioned gradients. The nanochat Polar Express uses a quintic polynomial that achieves similar quality in 3-4 iterations. The practical impact: nanochat-jax's Muon does 5 matrix multiplications per iteration (3 for `X@X^T@X` and 2 for the linear combination) times 5 iterations = 25 matmuls, while nanochat's quintic does ~5 matmuls per iteration times 3-4 iterations = 15-20 matmuls. For a `d_model x d_model` weight matrix, each matmul costs `O(d^3)`, so nanochat-jax's Muon has ~25-67% more orthogonalization overhead.

### 6.5 Loss Function Numerical Stability

The cross-entropy loss (`loss.py:44-48`) uses optax's fused implementation:

```python
ce = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits.reshape(-1, vocab_size),
    labels=safe_labels.reshape(-1),
)
```

This internally computes `log_softmax` then indexes, which is numerically stable (subtracts max before exp). The reshape to `(-1, vocab_size)` before the loss and reshape back is a standard pattern that avoids issues with optax's expectation of 2D inputs.

The masked loss reduction (`loss.py:56-58`):

```python
ce_masked = ce * mask
n_tokens = jnp.maximum(mask.sum(), 1.0)
ce_loss = ce_masked.sum() / n_tokens
```

The `jnp.maximum(mask.sum(), 1.0)` prevents division by zero when all positions are masked. This is particularly important for packed sequences where some batch elements may have fewer valid tokens than others.

### 6.6 Gradient Checkpointing

The `TransformerBlock` supports gradient checkpointing via `jax.checkpoint` (`block.py:285-295`):

```python
@jax.checkpoint
def _checkpointed(x_in, cos_in, sin_in, token_ids_in):
    x_out, _ = self._forward(x_in, cos_in, sin_in, mask, None, deterministic, token_ids_in)
    return x_out
```

The KV cache is computed **outside** the checkpoint boundary (`block.py:300-311`), ensuring it is available during inference without re-triggering recomputation. This is a deliberate design choice: during training, KV caches are not needed (full sequence is processed), but the same block class is used for inference where caches are essential.

**Numerical implication:** When gradient checkpointing is active, intermediate activations are recomputed during the backward pass. If XLA applies different fusion patterns during forward vs. recompute, the recomputed values may differ by floating-point epsilon. This is generally not a concern for training convergence but can affect exact numerical reproducibility between checkpointed and non-checkpointed runs.

### 6.7 Summary of Numerical Divergences

| Component | nanochat | nanochat-jax | Impact |
|---|---|---|---|
| Attention precision | bf16 (Flash Attention) | float32 (explicit) | nanochat-jax more precise per-score, but materializes full matrix |
| Softcap value | 15.0 | 30.0 | Wider linear regime in nanochat-jax |
| RMSNorm | F.rms_norm (bf16) | float32 promotion + rsqrt | nanochat-jax more numerically stable |
| NS orthogonalization | float32, quintic | float32, cubic | Same precision, different convergence rate |
| Gradient accumulation | Implicit in DDP | Not implemented (single device) | N/A for single-device comparison |
| Weight initialization | meta device + scatter | Immediate materialization + in-place scale | Same numerical result, different memory pattern |

The overall numerical fidelity is high: both codebases use float32 for critical computations (attention scores, normalization, optimizer state). The primary numerical divergences stem from the softcap value and the absence of Flash Attention, which affect training dynamics rather than per-operation precision.

### 6.8 XLA Determinism and Reproducibility

JAX provides stronger reproducibility guarantees than PyTorch by default. Given the same PRNG key, JAX operations produce bitwise-identical results across runs on the same hardware. The nanochat-jax codebase uses `nnx.Rngs` for RNG management (`transformer.py:60`, `runner.py:136`):

```python
# runner.py:136
rngs = nnx.Rngs(params=42, dropout=43)
model = TransformerLM(model_cfg, rngs=rngs)
```

The separation of `params` and `dropout` RNG streams ensures that model initialization is deterministic independently of dropout decisions during training. However, several factors can break reproducibility:

1. **Float32 non-associativity:** XLA may reorder floating-point reductions (e.g., `jnp.sum` over large arrays) differently across compilations or hardware, producing results that differ by floating-point epsilon.

2. **Gradient checkpointing:** When `use_remat=True` (`block.py:89`), the recomputed forward activations may use different XLA fusion patterns than the original forward pass, producing epsilon-different intermediate values.

3. **Dynamic shapes:** The attention mask construction (`transformer.py:165-170`) uses `jnp.tril(jnp.ones((S, S)))` where S comes from the input shape. If S changes between batches (e.g., variable-length sequences), XLA would need to recompile, and the resulting program may have different numerical behavior.

4. **PRNG splitting:** The scaling runner splits the PRNG key at each step (`runner.py:165`): `rng, k = jax.random.split(rng)`. This is deterministic given the initial seed but creates a dependency chain that must be exactly reproduced to get identical data across runs.

**Comparison with nanochat:** PyTorch's `torch.manual_seed()` provides reproducibility within a single run but `torch.compile()` and CUDA kernel selection can produce non-deterministic results across runs unless `torch.backends.cudnn.deterministic = True` is set. JAX's functional design makes determinism the default, which is advantageous for scaling law research where comparing results across model sizes requires confidence that differences are due to model capacity, not random variation.

### 6.9 Memory Layout and Efficiency

The tensor layouts in nanochat-jax follow standard conventions but with some choices worth noting:

**Attention layout:** `[B, n_heads, S, d_head]` (`attention.py:229`). This is the standard layout for non-fused attention. Flash Attention uses `[B, S, n_heads, d_head]` (sequence-major) to enable coalesced memory access patterns in the tiled kernel. If nanochat-jax were to adopt Pallas-based Flash Attention, the attention module would need a layout transpose.

**KV cache layout:** `[B, n_kv_heads, max_len, d_head]` (`kv_cache.py:16-17`). Pre-allocation with zeros means the cache always consumes `B * n_kv_heads * max_len * d_head * bytes_per_element` regardless of fill level. For the large preset (n_kv_heads=8, max_len=4096, d_head=64) with batch_size=1 in bf16, this is `1 * 8 * 4096 * 64 * 2 = 4MB` per layer, or `4MB * 24 = 96MB` total. This is modest, but for the xlarge preset with batch_size > 1, cache memory becomes significant.

**Embedding table layout:** `[vocab_size, d_model]` (`embeddings.py:78-82`). The value embedding table has the same layout (`value_embeddings.py:78-79`). For the BPE tokenizer with vocab_size=100,284 and d_model=512, each embedding table is `100,284 * 512 * 4 = 205MB` in float32. With both token embedding and value embedding, this is 410MB of embedding parameters -- a substantial fraction of total model memory for smaller models.

---

*Sections 7-12 continue in Part 2: Capability Gap Analysis, Performance Benchmarking, Ecosystem Integration, Research Reproducibility, Recommendations, and Appendices.*
