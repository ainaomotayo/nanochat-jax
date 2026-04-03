"""Critical integration test: model must be able to overfit a single batch."""
import math
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.model.transformer import TransformerLM
from nanochat.training.optimizer import build_optimizer
from nanochat.training.loss import cross_entropy_loss
from nanochat.config import ModelConfig, TrainingConfig


def test_model_can_overfit_single_batch():
    """Train on a single batch for N steps. Loss must reach < 0.5.

    This validates: forward pass, gradients, optimizer, and model capacity.
    If this fails, there is a fundamental bug.
    """
    N_STEPS = 300
    TARGET_LOSS = 0.5

    cfg = ModelConfig(
        vocab_size=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4,
        d_ff=128, max_seq_len=16, dropout_rate=0.0, tie_embeddings=True,
        use_bias=False, init_std=0.02, ffn_type="swiglu",
    )

    model = TransformerLM(cfg, rngs=nnx.Rngs(params=0, dropout=1))
    train_cfg = TrainingConfig(learning_rate=3e-3, warmup_steps=0, total_steps=N_STEPS, weight_decay=0.0)
    tx = build_optimizer(train_cfg)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Fixed batch
    batch_ids = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % cfg.vocab_size

    @nnx.jit
    def step(model, optimizer):
        def loss_fn(model):
            logits, _ = model(batch_ids, deterministic=False)
            loss, _ = cross_entropy_loss(logits[:, :-1], batch_ids[:, 1:])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    losses = []
    for i in range(N_STEPS):
        loss = step(model, optimizer)
        losses.append(float(loss))

    assert losses[-1] < TARGET_LOSS, (
        f"Model failed to overfit. Final loss: {losses[-1]:.4f} > {TARGET_LOSS}. "
        f"Last 10 losses: {[round(l, 4) for l in losses[-10:]]}"
    )
    assert not any(math.isnan(l) for l in losses), "NaN detected in training loss"
