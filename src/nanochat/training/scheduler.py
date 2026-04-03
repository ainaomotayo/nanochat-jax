"""Learning rate scheduling."""
from __future__ import annotations
import optax

def build_lr_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
    lr_decay_steps: int | None = None,
) -> optax.Schedule:
    """Build linear warmup -> cosine decay -> constant floor schedule.

    Matches GPT-3 / LLaMA training setup.

    Args:
        learning_rate: Peak learning rate
        warmup_steps: Linear warmup duration
        total_steps: Total training steps (for logging reference)
        min_lr_ratio: min_lr = learning_rate * min_lr_ratio
        lr_decay_steps: Cosine decay duration. Defaults to total_steps.

    Returns:
        optax.Schedule that maps step -> lr
    """
    if lr_decay_steps is None:
        lr_decay_steps = total_steps

    decay_steps = lr_decay_steps - warmup_steps

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps,
    )

    cosine = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=max(decay_steps, 1),
        alpha=min_lr_ratio,
    )

    return optax.join_schedules(
        schedules=[warmup, cosine],
        boundaries=[warmup_steps],
    )
