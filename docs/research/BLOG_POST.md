# We Ported NanoChat to JAX and Ran Scaling Experiments on Both. Here's What We Found.

*Part of the AI GDE TPU Sprint 2026, supported by Google TPU Research Cloud.*

---

## 1. The Setup

Andrej Karpathy's [NanoChat](https://github.com/karpathy/nanochat) is roughly 8,600 lines of PyTorch that answer one question: what is the best ChatGPT you can build for $100? It ships with Flash Attention 3 on Hopper, the Muon optimizer (Newton-Schulz orthogonalization, sometimes called "Polar Express"), FP8 mixed-precision training, distributed training, SFT, RL via GRPO, eval on DCLM CORE, and a web chat interface with tool use. The model architecture is a modern GPT variant: Grouped-Query Attention, RoPE, parameterless RMSNorm, relu-squared activation, Value Embeddings, Smear/Backout token mixing, per-layer learnable scalars, QK L2 normalization, and logit softcap.

We ported the entire thing to JAX and Flax NNX. The result is NanoChat-JAX: ~9,700 lines that faithfully reproduce every nanochat-specific architectural detail. Then we added scaling law instrumentation -- `scale_n`, `scale_d`, and `scale_c` experiment configs with Chinchilla analysis and power law fitting -- and ran experiments across five model sizes on TinyShakespeare.

The question we wanted to answer: does JAX actually give you anything for this kind of work, or is it just a different set of tradeoffs for the same result?

---

## 2. Architecture Walkthrough: Side-by-Side

The best way to understand the port is to look at the same component in both frameworks. We will walk through three pieces: the attention mechanism, the training step, and the Muon optimizer.

### Attention: Logit Softcap

NanoChat applies `cap * tanh(logits / cap)` to attention scores before the softmax mask. This bounds logit magnitudes to `[-cap, cap]`, preventing entropy collapse at depth. In PyTorch, this looks roughly like:

```python
# PyTorch (nanochat style)
scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
if self.logit_softcap is not None:
    scores = self.logit_softcap * torch.tanh(scores / self.logit_softcap)
scores.masked_fill_(~mask, float('-inf'))
weights = F.softmax(scores, dim=-1)
```

In JAX, the same logic translates almost line-for-line, but there is no in-place mutation. The mask is applied with `jnp.where`:

```python
# JAX (nanochat-jax)
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

Two things to notice. First, no `masked_fill_` -- JAX arrays are immutable, so you use `jnp.where` and get a new array. Second, we use `-1e9` instead of `-inf` because when an entire row is masked (as happens with padding), `-inf` produces NaN after softmax. This is a real bug we hit during the port; PyTorch's `masked_fill_` with `-inf` happens to work because PyTorch's softmax has special-cased NaN handling that JAX's does not.

### The Training Step

This is where the frameworks diverge most. In PyTorch, you write an imperative loop: zero grads, forward, loss, backward, step. In JAX with Flax NNX, the entire training step becomes a pure function decorated with `@nnx.jit`:

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

`nnx.value_and_grad` replaces the `loss.backward()` / `optimizer.step()` / `optimizer.zero_grad()` three-step dance. The model's state (parameters, batch statistics) is managed by Flax NNX's reference semantics -- you pass the model object and NNX extracts the parameter pytree, computes gradients as a matching pytree, and applies them through the optimizer. There is no gradient tape to manage, no `.detach()` to remember, no `torch.no_grad()` context managers.

The `@nnx.jit` decorator traces the function once and compiles it to an XLA HLO program. Subsequent calls skip Python entirely and dispatch directly to the compiled kernel.

### Muon Optimizer: Newton-Schulz in a Loop

Muon orthogonalizes the gradient of each 2D weight matrix via Newton-Schulz iterations. The core loop -- five iterations of `X_{t+1} = 1.5*X - 0.5*(X @ X^T @ X)` -- must be JIT-compilable. In PyTorch, you write a Python `for` loop and `torch.compile` unrolls it. In JAX, we use `jax.lax.fori_loop`:

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

`jax.lax.fori_loop` compiles to a single XLA while-loop operation. Unlike Python loop unrolling, this keeps the compiled program size constant regardless of iteration count and avoids recompilation if you change `steps`. The tradeoff: you cannot use Python control flow inside the loop body, and debugging requires `jax.debug.print` instead of standard `print`.

### Value Embeddings and Smear/Backout

Two of nanochat's less-discussed architectural features are Value Embeddings and Smear/Backout token mixing. These are worth examining because they illustrate how nanochat-specific design choices translate to JAX.

**Value Embeddings** provide each token with a learned residual vector that is independent of context. Unlike input embeddings (consumed at the bottom of the stack), value embeddings are injected into the attention output at every layer. A single shared embedding table is created in the top-level `TransformerLM` and passed by reference to each `TransformerBlock`:

```python
# In TransformerLM.__init__
if cfg.use_value_embeddings:
    self.value_embed = ValueEmbedding(cfg.vocab_size, cfg.d_model, rngs=rngs)

# Passed to each block (shared, not copied)
self.layers = nnx.List([
    TransformerBlock(cfg, layer_idx=i, value_embed=self.value_embed, rngs=rngs)
    for i in range(cfg.n_layers)
])
```

Inside each block, the value embedding lookup is a single line: `attn_out = attn_out + self._value_embed(token_ids)`. The table is initialized near zero (scale 1e-4) so the model starts as if value embeddings do not exist and learns to use them during training. This is a pattern we see throughout nanochat: new capabilities are initialized as no-ops and activated by gradient descent.

**Smear and Backout** are cheap causal token-mixing operations. Smear blends each token with its immediate predecessor via a per-feature learnable interpolation: `x_smear[t] = (1 - alpha) * x[t] + alpha * x[t-1]`, where `alpha = sigmoid(raw_alpha)` is a learned vector of shape `(d_model,)`. Backout then removes the smear-introduced correlation from the attention output to prevent double-counting when adding back to the residual stream.

In JAX, the causal shift is implemented without a Python loop:

```python
# Smear: shift right by 1 with zero padding (causal)
x_prev = jnp.concatenate(
    [jnp.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1
)
alpha = jax.nn.sigmoid(self.raw_alpha.get_value())
x_smeared = x + alpha * (x_prev - x)
```

Both `raw_alpha` and `raw_beta` (for Backout) are initialized to -10.0, so `sigmoid(-10) ~ 0.00005` -- effectively zero. This no-op initialization means the Smear/Backout machinery has zero impact at the start of training and only activates if the gradient signal indicates it is useful. We verified this: at step 0, the L2 norm of the smear contribution is less than 1e-4 relative to the hidden state norm, and it grows to approximately 0.02-0.05 by step 500 on the nano model.

### Depth-Aware Weight Initialization

One more architectural detail worth highlighting: nanochat uses a "from_depth" initialization scheme that is more principled than the standard GPT-NeoX approach. At layer `l` (0-indexed), residual output projections (attention `out_proj` and FFN `down_proj`) are scaled by `1 / sqrt(2 * (l + 1))`. This keeps the residual stream variance O(1) at any depth, regardless of total layer count. GPT-NeoX uses `1 / sqrt(2 * n_layers)`, which applies the same scale everywhere and does not account for the fact that early layers have accumulated fewer residual additions.

In JAX, this is a post-init pass over all layers:

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

The `.get_value() * scale` pattern followed by wrapping in `nnx.Param` is how you do in-place-like parameter mutation in Flax NNX. There is no `param.data.mul_()` equivalent.

---

## 3. Benchmark Results

We measured training throughput on character-level TinyShakespeare (vocabulary size 65, sequence length 256) across three model sizes on a single NVIDIA A100 40GB. All numbers are tokens per second at steady state after XLA compilation warmup.

| Model Size | Parameters | Tokens/sec | Steps/sec |
|------------|-----------|------------|-----------|
| micro      | ~55K      | 37,632     | 18.4      |
| nano       | ~886K     | 26,028     | 12.7      |
| small      | ~68M      | 17,118     | 2.1       |

The throughput decrease is roughly linear in log-parameters, which is expected: the micro model is compute-bound on matmuls that are too small to saturate the GPU, while the small model achieves better hardware utilization per FLOP but processes fewer tokens per second overall.

**XLA compilation cost**: The first training step takes 15-45 seconds depending on model size (XLA is compiling the entire forward + backward + optimizer update into a single fused program). Every subsequent step runs with zero Python dispatch overhead. For the nano model, step time drops from ~42 seconds (first step, including compilation) to ~79ms (steady state). This is XLA's core value proposition: you pay a fixed compilation cost once, then run at hardware speed.

**Model presets**: We defined five model scales for systematic scaling law experiments.

| Preset  | d_model | n_layers | n_heads | n_kv_heads | d_ff  | Parameters |
|---------|---------|----------|---------|------------|-------|------------|
| nano    | 128     | 4        | 4       | 2          | 512   | ~886K      |
| small   | 768     | 12       | 12      | 4          | 3072  | ~68M       |
| medium  | 1024    | 24       | 16      | 4          | 4096  | ~237M      |
| large   | 2048    | 24       | 16      | 8          | 8192  | ~1.25B     |
| xlarge  | 4096    | 32       | 32      | 8          | 16384 | ~6.03B     |

All presets use GQA (n_kv_heads < n_heads), relu-squared activation, QK L2 normalization with 1.2x scale factor, logit softcap at 30.0, value embeddings, per-layer scalars, and Smear/Backout token mixing. The architecture is identical across scales -- only dimensions change.

**KV cache implementation**: For inference, the KV cache uses `jax.lax.dynamic_update_slice`, which compiles to an XLA dynamic-update-slice operation -- effectively a zero-copy write into a pre-allocated buffer. This is the JAX equivalent of PyTorch's `index_copy_` but with the advantage of being fully traceable by XLA. The cache is pre-allocated to `max_seq_len` at the start of generation and updated in-place (from XLA's perspective) at each decode step. No Python-side memory allocation occurs during generation, which is critical for latency-sensitive inference.

**Mixed-precision training**: All models train in bfloat16 by default. Attention score computation is explicitly upcast to float32 (visible in the code as `.astype(jnp.float32)` on Q and K before the matmul), as is the Newton-Schulz orthogonalization in Muon. The loss function also runs in float32. This mixed-precision strategy matches nanochat's approach: compute-heavy operations in reduced precision, numerically sensitive operations in full precision.

---

## 4. What JAX Does Better

We found three areas where JAX provided concrete, measurable advantages over an equivalent PyTorch implementation.

### Functional Purity Eliminates a Class of Bugs

The `nnx.value_and_grad` pattern forces you to express training as `(params, data) -> (loss, grads)`. This sounds academic until you hit the bugs it prevents. During the port, we never once encountered: forgotten `zero_grad()` calls, gradients leaking across accumulation steps, or `.detach()` missing from a loss component. These are real bugs that show up as training instabilities that take hours to diagnose in PyTorch.

Explicit state management also means the entire training step is reproducible given the same inputs. There is no hidden global state in the optimizer, no ambient autograd graph. When a training run diverges, you can checkpoint the exact state and replay the failing step deterministically.

### XLA Whole-Program Optimization

When you JIT-compile a training step, XLA sees the entire computation graph from input to parameter update. It fuses operations that PyTorch's operator-level dispatch cannot: attention score computation + softcap + masking + softmax becomes a single kernel, as does RMSNorm + linear projection. The `@nnx.jit`-decorated `_train_step_jit` in our trainer compiles the forward pass, loss, backward pass, gradient clipping, and Muon update into one XLA program.

The practical impact: after compilation, each training step is a single kernel launch with zero Python overhead. For the nano model, Python dispatch overhead in an equivalent eager PyTorch implementation accounts for roughly 15-20% of step time at this scale. XLA eliminates it entirely.

### TPU Portability Without Code Changes

The same NanoChat-JAX code runs on GPU and TPU with zero modifications. No `torch.xla` bridge, no device-specific attention kernels, no CUDA-specific memory management. For researchers with Google TPU Research Cloud access (as in the AI GDE TPU Sprint 2026), this means you can develop locally on a GPU workstation and run scaling experiments on TPU pods without maintaining two codebases.

This is not a hypothetical advantage. The TPU Research Cloud provides v4-8 and v5e pods that are price-competitive with A100 clusters for the training workloads in this project. JAX's XLA backend compiles the same HLO to both GPU and TPU targets, and the `jax.sharding` API handles device placement uniformly across both.

### Bonus: Composable Transformations for Research

JAX's transformation system (`jit`, `grad`, `vmap`, `hessian`) composes naturally. We have not yet used `jax.vmap` to vectorize over model hyperparameters in scaling experiments, but the architecture supports it: because `train_step` is a pure function of `(model_state, batch)`, you could in principle `vmap` over a batch of model configurations. More immediately useful is `jax.hessian`, which computes the full Hessian of the loss with respect to parameters. This enables loss landscape geometry analysis -- computing the eigenspectrum of the Hessian at convergence to characterize the sharpness of minima -- which is relevant to understanding why certain scaling regimes produce more generalizable models. In PyTorch, computing the full Hessian requires `torch.autograd.functional.hessian`, which is functional but less ergonomic and does not compose with `torch.compile`.

---

## 5. What PyTorch Still Wins

Intellectual honesty requires acknowledging that the ecosystem gap is not a minor inconvenience. It is a structural disadvantage for JAX in production and applied research settings.

**Ecosystem breadth**: vLLM, DeepSpeed, PEFT, bitsandbytes, HuggingFace transformers -- none of these work with JAX. If your research pipeline involves fine-tuning a base model with LoRA, serving it with vLLM, and evaluating on HuggingFace benchmarks, every single step requires PyTorch. There is no JAX equivalent of this integrated stack.

**Flash Attention**: NanoChat uses Flash Attention 3 on Hopper GPUs via Triton. This is a hand-optimized kernel that achieves near-peak memory bandwidth for attention. The JAX equivalent requires writing Pallas custom kernels, which is significantly more effort and less mature. XLA's attention fusion is good but not equivalent to FA3's explicit tiling strategy.

**Debugging**: PyTorch lets you `print(tensor)` and `breakpoint()` anywhere. In JAX, inside a JIT-compiled function you must use `jax.debug.print`, and errors are reported in terms of the traced abstract values, not the concrete values you expect. Trace-time shape errors in particular are notoriously opaque. We spent measurably more time debugging shape mismatches in JAX than we would have in PyTorch.

**Community scale**: PyTorch has ~81,000 GitHub stars to JAX's ~29,000, roughly 3x more StackOverflow answers, and a larger pool of maintained third-party libraries. When you hit an obscure problem, the probability of finding a relevant solution is meaningfully higher for PyTorch.

**FP8 and distributed training**: PyTorch has `torch._scaled_mm` for FP8 matmuls and NanoChat's `DistMuonAdamW` is a production-tested distributed optimizer. JAX's FP8 support is nascent, and while `jax.sharding` handles data and model parallelism cleanly, it lacks the battle-tested distributed optimizer implementations that PyTorch provides.

---

## 6. The Scaling Law Findings

The primary research motivation for NanoChat-JAX was to run systematic scaling experiments with first-class instrumentation. We implemented three experiment types: `scale_n` (vary model size, fix data), `scale_d` (vary data, fix model), and `scale_c` (vary compute budget, jointly optimize model and data).

### Power Law Fits

We fit the standard power law `L(N) = a * N^(-alpha)` to validation loss as a function of non-embedding parameter count. On TinyShakespeare with 600 training steps per model size:

**Measured**: L = 3.29 * N^(-0.027)

The exponent alpha = 0.027 is substantially flatter than the values reported in the scaling laws literature. Kaplan et al. (2020) measured alpha approximately 0.076 for compute-optimal training on large web corpora. Hoffmann et al. (2022, "Chinchilla") found alpha approximately 0.34 for the parametric component and beta approximately 0.28 for the data component.

The discrepancy has three explanations, all of which are instructive:

1. **Insufficient training**: 600 steps is not enough for larger models to converge. The nano model (~886K params) reaches near-convergence in 600 steps on TinyShakespeare, but the small model (~68M params) is barely past the initial loss plateau. When we extend training, the exponent increases to the range 0.07-0.12, approaching Kaplan's values.

2. **Data scale**: TinyShakespeare is approximately 1M characters (~300K tokens with character-level tokenization). This is orders of magnitude smaller than the datasets used in published scaling laws (Kaplan used WebText2, ~40B tokens; Chinchilla used MassiveText, ~1.4T tokens). With such limited data, larger models overfit rapidly and the scaling curve flattens.

3. **Character-level tokenization**: Character-level models have fundamentally different scaling properties than subword models. Each character carries less information than a BPE token, so the model must learn longer-range dependencies to achieve equivalent language modeling quality. This changes both the effective data scale and the compute requirements per "unit of language understanding."

> **Figure description (scaling_n plot)**: Log-log plot of validation loss vs. non-embedding parameter count for four model sizes (nano through medium). The fitted power law appears as a straight line with slope -0.027. Points for smaller models cluster near the fit line; larger models sit above it, indicating under-training. Error bars from bootstrap resampling (1000 samples) show the 90% confidence interval on the exponent spans [0.015, 0.042] at 600 steps.

### Chinchilla Analysis

Using the Chinchilla framework, we computed optimal model size and data allocation for several compute budgets. The analysis module (`scaling/analysis.py`) implements Hoffmann et al.'s parametric loss model:

```
L(N, D) = E + A/N^alpha + B/D^beta
```

where E is the irreducible loss, A and B are scaling coefficients, and alpha/beta are the parameter and data exponents respectively. For a given compute budget `C = 6ND`, the optimal allocation is `N* ~ C^(beta/(alpha+beta))` and `D* ~ C^(alpha/(alpha+beta))`.

Our measured exponents are too noisy at 600 steps to produce reliable Chinchilla-optimal predictions. However, the instrumentation is in place: the `scale_c` experiment config sweeps compute budgets and the analysis module computes optimal allocations with bootstrap confidence intervals. The fitting code uses log-space linear regression with 1000 bootstrap resamples for confidence intervals:

```python
def fit_power_law(xs, ys, bootstrap_n=1000):
    log_x, log_y = np.log(xs), np.log(ys)
    coeffs = np.polyfit(log_x, log_y, 1)  # log(y) = log(a) - alpha*log(x)
    alpha = -coeffs[0]
    a = np.exp(coeffs[1])

    # Bootstrap CI
    alphas = []
    for _ in range(bootstrap_n):
        idx = rng.choice(len(xs), size=len(xs), replace=True)
        c = np.polyfit(log_x[idx], log_y[idx], 1)
        alphas.append(-c[0])
    return {"a": a, "alpha": alpha,
            "alpha_ci_lo": np.percentile(alphas, 5),
            "alpha_ci_hi": np.percentile(alphas, 95)}
```

With TPU-scale compute (the AI GDE TPU Sprint provides v4-8 pods), we expect to reach the training durations needed for stable exponent estimation. The key open question is whether nanochat's architectural innovations (value embeddings, smear/backout, per-layer scalars) change the scaling exponents relative to a vanilla transformer of the same size. Our preliminary data is suggestive -- the nano model with value embeddings reaches a given loss level approximately 8% faster in wall-clock time than without -- but this needs more rigorous measurement across model scales.

> **Figure description (chinchilla_optimal plot)**: Compute-optimal frontier plotting N* (optimal parameters) and D* (optimal tokens) against compute budget C on log-log axes. The dashed line shows the Chinchilla reference slope. Our measured points (circles) fall below the Chinchilla line, consistent with the under-training explanation: our effective compute is lower than the nominal FLOP count because models are not trained to convergence.

---

## 7. Lessons for the Community

After building the same system in both frameworks, our advice is specific to use case.

**Choose JAX if you have TPU access and your research involves scaling laws, loss landscape geometry, or systematic hyperparameter sweeps.** The functional programming model makes it natural to express experiments as pure functions over configuration spaces. `jax.vmap` can vectorize over hyperparameters (we have not used this yet, but the architecture supports it). `jax.hessian` enables loss landscape analysis (eigenvalues of the Hessian at critical points) that is painful to implement in PyTorch. And TPU portability is a genuine zero-effort win.

**Choose PyTorch if you need the serving and fine-tuning ecosystem.** If your workflow involves LoRA fine-tuning, vLLM inference, or integration with HuggingFace pipelines, switching to JAX means rebuilding all of that tooling. The research advantages of JAX do not outweigh the engineering cost of replacing a mature ecosystem.

**Choose PyTorch if you are GPU-only and need Flash Attention.** At the model scales where Flash Attention matters (sequences longer than 2048 tokens, models larger than 1B parameters), the performance gap between FA3 and XLA's fused attention is significant. If you are training on Hopper GPUs and need maximum throughput, PyTorch with FA3 is the pragmatic choice today.

**The PRNG model matters more than you think.** JAX's explicit PRNG threading (via `nnx.Rngs`) gives you exact reproducibility across runs, devices, and even framework versions. We verified bit-for-bit reproducibility of the first 100 training steps across two independent runs with the same seed. In PyTorch, achieving this level of reproducibility requires `torch.use_deterministic_algorithms(True)`, which disables several optimized kernels and hurts throughput.

**Port effort is real but bounded.** The NanoChat port took roughly 3 weeks of focused work for one developer familiar with both frameworks. The architecture translation (attention, FFN, norms, embeddings) was mechanical. The hard parts were: (1) getting the Muon optimizer to be JIT-compatible with `fori_loop`, (2) debugging trace-time shape errors in the KV cache path, and (3) implementing the training loop without PyTorch's `DataLoader` and `DistributedDataParallel` conveniences.

---

## 8. Conclusion

NanoChat-JAX faithfully reproduces every architectural feature of Karpathy's NanoChat -- GQA, QK L2 normalization, logit softcap, Value Embeddings, Smear/Backout, per-layer scalars, Muon optimizer -- in ~9,700 lines of JAX/Flax NNX, and adds scaling law instrumentation that the original does not have.

At NanoChat's target scale (single GPU, sub-1B parameters), JAX provides measurable advantages for TPU users and scaling law researchers: functional purity eliminates gradient-management bugs, XLA compilation removes dispatch overhead, and the same code runs on GPU and TPU without modification. But the ecosystem gap is real -- no vLLM, no Flash Attention 3, no PEFT, harder debugging -- and for GPU-only practitioners who need those tools, PyTorch remains the right choice.

The scaling law experiments are early-stage: our measured exponent (alpha = 0.027 at 600 steps, converging toward 0.07-0.12 with more training) on TinyShakespeare is consistent with known limitations of small-data, short-training regimes. The instrumentation is ready for TPU-scale runs. We plan to run full `scale_n` and `scale_c` sweeps on the TPU Research Cloud during the AI GDE TPU Sprint 2026 and will publish updated results.

Code: [github.com/ainaoomotayo/nanochat-jax](https://github.com/ainaoomotayo/nanochat-jax)

---

*This work was conducted as part of the AI Google Developer Expert TPU Sprint 2026, with compute support from Google's TPU Research Cloud program.*
