"""Checkpoint management using Orbax."""
from __future__ import annotations
import json
import pickle
import shutil
import structlog
from pathlib import Path
from typing import Any

import jax
from flax import nnx

logger = structlog.get_logger()


class CheckpointManager:
    """Manages model checkpoints with save/restore/best-k tracking.

    Directory structure:
        checkpoint_dir/
            step_005000/
                model.msgpack
                metadata.json
            step_010000/
                model.msgpack
                metadata.json
            latest -> step_010000 (symlink)
    """

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._checkpoints: list[dict[str, Any]] = []
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Scan directory for existing checkpoints."""
        self._checkpoints = []
        for p in sorted(self.checkpoint_dir.glob("step_*")):
            if p.is_dir():
                meta_path = p / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    self._checkpoints.append({"path": p, **meta})
        logger.info("checkpoints_scanned", count=len(self._checkpoints))

    def save(
        self,
        step: int,
        model: nnx.Module,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """Save model checkpoint with metadata.

        Args:
            step: Current training step
            model: Model to save
            metrics: Optional metrics dict (e.g., val_loss)

        Returns:
            Path to saved checkpoint directory
        """
        step_dir = self.checkpoint_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        state = nnx.state(model)
        state_path = step_dir / "model.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(
                jax.tree.map(
                    lambda x: x.tolist() if hasattr(x, 'tolist') else x,
                    jax.tree.leaves(state),
                ),
                f,
            )

        # Save metadata
        metadata = {
            "step": step,
            "metrics": metrics or {},
        }
        with open(step_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update latest symlink
        latest = self.checkpoint_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(step_dir.name)

        # Track and cleanup old checkpoints
        self._checkpoints.append({"path": step_dir, "step": step, **(metrics or {})})
        self._cleanup()

        logger.info("checkpoint_saved", step=step, path=str(step_dir))
        return step_dir

    def _cleanup(self) -> None:
        """Remove old checkpoints, keeping only keep_last_n."""
        if len(self._checkpoints) > self.keep_last_n:
            to_remove = self._checkpoints[:-self.keep_last_n]
            for ckpt in to_remove:
                path = ckpt["path"]
                if path.exists():
                    shutil.rmtree(path)
                    logger.info("checkpoint_removed", path=str(path))
            self._checkpoints = self._checkpoints[-self.keep_last_n:]

    def find_latest(self) -> Path | None:
        """Find the latest checkpoint."""
        latest = self.checkpoint_dir / "latest"
        if latest.is_symlink():
            target = self.checkpoint_dir / latest.resolve().name
            if target.exists():
                return target
        if self._checkpoints:
            return self._checkpoints[-1]["path"]
        return None

    def get_step(self, path: Path) -> int:
        """Extract step number from checkpoint path."""
        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)["step"]
        return 0

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints."""
        return list(self._checkpoints)
