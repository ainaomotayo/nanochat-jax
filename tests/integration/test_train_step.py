"""Integration test: full training step."""
import jax
import jax.numpy as jnp
from flax import nnx
from nanochat.model.transformer import TransformerLM
from nanochat.training.optimizer import build_optimizer
from nanochat.training.loss import cross_entropy_loss
from nanochat.config import TrainingConfig


def test_loss_decreases(nano_config):
    """Verify that loss decreases over a few training steps."""
    model = TransformerLM(nano_config, rngs=nnx.Rngs(params=0, dropout=1))
    cfg = TrainingConfig(learning_rate=1e-3, warmup_steps=0, total_steps=20, weight_decay=0.0)
    tx = build_optimizer(cfg)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    rng = jax.random.PRNGKey(42)
    batch = {
        "input_ids": jax.random.randint(rng, (2, nano_config.max_seq_len), 0, nano_config.vocab_size),
    }

    @nnx.jit
    def step(model, optimizer, batch):
        def loss_fn(model):
            logits, _ = model(batch["input_ids"], deterministic=False)
            loss, _ = cross_entropy_loss(logits[:, :-1], batch["input_ids"][:, 1:])
            return loss
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    losses = []
    for _ in range(20):
        loss = step(model, optimizer, batch)
        losses.append(float(loss))

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(not __import__('math').isnan(l) for l in losses), "NaN in losses"
