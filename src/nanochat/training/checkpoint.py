"""Checkpoint management — save, load, and resume training."""
from __future__ import annotations
import json
import pickle
import shutil
import structlog
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

logger = structlog.get_logger()


class CheckpointManager:
    """Manages model checkpoints with save/restore/best-k tracking.

    Directory structure::

        checkpoint_dir/
            step_005000/
                model.pkl        # numpy-serialised NNX State tree
                optimizer.pkl    # optimizer state (optional)
                metadata.json    # step, metrics
            step_010000/
                ...
            latest -> step_010000  (symlink)

    Example::

        ckpt = CheckpointManager("checkpoints/")
        ckpt.save(1000, model, {"val_loss": 2.34})
        step = ckpt.load_latest(model)   # restores weights, returns step
    """

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._checkpoints: list[dict[str, Any]] = []
        self._scan_existing()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_existing(self) -> None:
        """Populate self._checkpoints from disk."""
        self._checkpoints = []
        for p in sorted(self.checkpoint_dir.glob("step_*")):
            if p.is_dir():
                meta_path = p / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self._checkpoints.append({"path": p, **meta})
        logger.info("checkpoints_scanned", count=len(self._checkpoints))

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping only keep_last_n."""
        if len(self._checkpoints) > self.keep_last_n:
            to_remove = self._checkpoints[: -self.keep_last_n]
            for ckpt in to_remove:
                path = ckpt["path"]
                if path.exists():
                    shutil.rmtree(path)
                    logger.info("checkpoint_removed", path=str(path))
            self._checkpoints = self._checkpoints[-self.keep_last_n :]

    @staticmethod
    def _state_to_numpy(state: Any) -> Any:
        """Convert NNX State (JAX array leaves) → numpy, preserving tree structure.

        PRNGKey-typed arrays (from nnx.Rngs) are serialised via
        ``jax.random.key_data()`` to avoid the dtype-conversion error.
        """
        def _convert(x):
            if not hasattr(x, "dtype"):
                return x
            dtype_str = str(getattr(x.dtype, "name", x.dtype))
            if "key" in dtype_str:
                # Typed-key array — extract raw uint32 backing data
                return np.asarray(jax.random.key_data(x))
            return np.asarray(x)

        return jax.tree.map(_convert, state)

    @staticmethod
    def _numpy_to_jax(state_np: Any) -> Any:
        """Convert numpy leaves back to JAX arrays."""
        return jax.tree.map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            state_np,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        model: nnx.Module,
        metrics: dict[str, Any] | None = None,
        optimizer: nnx.Optimizer | None = None,
    ) -> Path:
        """Save model (and optionally optimizer) checkpoint.

        Args:
            step: Current training step.
            model: NNX model whose parameters to save.
            metrics: Optional metrics dict (e.g. val_loss).
            optimizer: Optional NNX Optimizer; saves optimizer state if given.

        Returns:
            Path to the saved checkpoint directory.
        """
        step_dir = self.checkpoint_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # ── Model state ──────────────────────────────────────────────
        model_state = nnx.state(model)
        model_np = self._state_to_numpy(model_state)
        with open(step_dir / "model.pkl", "wb") as f:
            pickle.dump(model_np, f, protocol=4)

        # ── Optimizer state (optional) ───────────────────────────────
        if optimizer is not None:
            opt_state = nnx.state(optimizer)
            opt_np = self._state_to_numpy(opt_state)
            with open(step_dir / "optimizer.pkl", "wb") as f:
                pickle.dump(opt_np, f, protocol=4)

        # ── Metadata ─────────────────────────────────────────────────
        metadata: dict[str, Any] = {"step": step, "metrics": metrics or {}}
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # ── Latest symlink ───────────────────────────────────────────
        latest = self.checkpoint_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(step_dir.name)

        self._checkpoints.append({"path": step_dir, "step": step, "metrics": metrics or {}})
        self._cleanup()

        logger.info("checkpoint_saved", step=step, path=str(step_dir))
        return step_dir

    def load(
        self,
        path: Path | str,
        model: nnx.Module,
        optimizer: nnx.Optimizer | None = None,
    ) -> int:
        """Restore model (and optionally optimizer) state from a checkpoint.

        Args:
            path: Path to the checkpoint directory (e.g. ``step_005000/``).
            model: NNX model to update in-place.
            optimizer: Optional NNX Optimizer to update in-place.

        Returns:
            The training step stored in the checkpoint.

        Raises:
            FileNotFoundError: If ``model.pkl`` does not exist at *path*.
        """
        path = Path(path)
        model_pkl = path / "model.pkl"
        if not model_pkl.exists():
            raise FileNotFoundError(f"No model.pkl found at {path}")

        # ── Model ─────────────────────────────────────────────────────
        with open(model_pkl, "rb") as f:
            model_np = pickle.load(f)
        model_state = self._numpy_to_jax(model_np)
        nnx.update(model, model_state)

        # ── Optimizer (optional) ──────────────────────────────────────
        opt_pkl = path / "optimizer.pkl"
        if optimizer is not None and opt_pkl.exists():
            with open(opt_pkl, "rb") as f:
                opt_np = pickle.load(f)
            opt_state = self._numpy_to_jax(opt_np)
            nnx.update(optimizer, opt_state)

        step = self.get_step(path)
        logger.info("checkpoint_loaded", step=step, path=str(path))
        return step

    def load_latest(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer | None = None,
    ) -> int | None:
        """Load the most recent checkpoint.

        Returns:
            The restored step, or ``None`` if no checkpoints exist.
        """
        path = self.find_latest()
        if path is None:
            logger.info("no_checkpoint_found")
            return None
        return self.load(path, model, optimizer=optimizer)

    def find_latest(self) -> Path | None:
        """Return path of the latest checkpoint directory, or None."""
        latest = self.checkpoint_dir / "latest"
        if latest.is_symlink():
            target = latest.resolve()
            if target.exists():
                return target
        if self._checkpoints:
            return self._checkpoints[-1]["path"]
        return None

    def get_step(self, path: Path) -> int:
        """Read the step number from a checkpoint's metadata.json."""
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return int(json.load(f).get("step", 0))
        # Fall back: parse from directory name (step_XXXXXX)
        try:
            return int(path.name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """Return all tracked checkpoints (oldest first)."""
        return list(self._checkpoints)

    def best_checkpoint(self, metric: str = "val_loss", lower_is_better: bool = True) -> Path | None:
        """Return the checkpoint with the best value for *metric*."""
        candidates = [
            c for c in self._checkpoints
            if metric in c.get("metrics", {})
        ]
        if not candidates:
            return None
        key = lambda c: c["metrics"][metric]
        best = min(candidates, key=key) if lower_is_better else max(candidates, key=key)
        return best["path"]
