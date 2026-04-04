"""Evaluation suite for running lm-eval-harness benchmarks on nanochat-jax models.

Provides predefined task sets and a convenience wrapper around
``lm_eval.simple_evaluate()``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog

from nanochat.evaluation.lm_eval_adapter import NanoChatJAXModel, _LM_EVAL_AVAILABLE

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Predefined task sets
# ---------------------------------------------------------------------------

QUICK_TASKS: list[str] = ["hellaswag", "arc_easy", "piqa", "lambada_openai"]

STANDARD_TASKS: list[str] = QUICK_TASKS + ["arc_challenge", "winogrande"]


def _ensure_lm_eval() -> None:
    if not _LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm-eval is required to run the evaluation suite. "
            "Install it with: pip install lm-eval>=0.4"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval_suite(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    tasks: Optional[list[str]] = None,
    num_fewshot: int = 0,
    batch_size: int = 8,
    limit: Optional[int] = None,
    output_path: Optional[str | Path] = None,
    model_obj: Optional[NanoChatJAXModel] = None,
) -> dict[str, dict[str, Any]]:
    """Run an lm-eval-harness evaluation suite.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        device: Target device (``"cpu"``, ``"gpu"``, ``"tpu"``).
        tasks: List of lm-eval task names. Defaults to :data:`QUICK_TASKS`.
        num_fewshot: Number of few-shot examples per task.
        batch_size: Batch size for evaluation.
        limit: Max examples per task (for quick testing). None = all.
        output_path: If given, write ``results.json`` and ``summary.md`` here.
        model_obj: Pre-built :class:`NanoChatJAXModel`. If None, loads from
            *checkpoint_path*.

    Returns:
        Nested dict of ``{task_name: {metric_name: value}}``.
    """
    _ensure_lm_eval()
    import lm_eval

    if tasks is None:
        tasks = list(QUICK_TASKS)

    # Build or reuse model
    if model_obj is None:
        adapter = NanoChatJAXModel.from_checkpoint(
            checkpoint_path, device=device, batch_size=batch_size
        )
    else:
        adapter = model_obj

    logger.info("eval_suite_start", tasks=tasks, num_fewshot=num_fewshot, limit=limit)

    eval_kwargs: dict[str, Any] = {
        "model": adapter,
        "tasks": tasks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
    }
    if limit is not None:
        eval_kwargs["limit"] = limit

    raw_results = lm_eval.simple_evaluate(**eval_kwargs)

    # Extract per-task metrics into a clean nested dict
    task_results: dict[str, dict[str, Any]] = {}
    results_dict = raw_results.get("results", {})
    for task_name, metrics in results_dict.items():
        clean_metrics: dict[str, Any] = {}
        for metric_key, value in metrics.items():
            # lm-eval uses keys like "acc,none" or "acc_norm,none"
            if isinstance(value, (int, float)):
                clean_key = metric_key.split(",")[0] if "," in metric_key else metric_key
                clean_metrics[clean_key] = value
        task_results[task_name] = clean_metrics

    logger.info("eval_suite_complete", n_tasks=len(task_results))

    # Write outputs
    if output_path is not None:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_results_json(out_dir / "results.json", raw_results, task_results)
        _write_summary_md(out_dir / "summary.md", task_results, num_fewshot)

    return task_results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_results_json(
    path: Path,
    raw_results: dict[str, Any],
    task_results: dict[str, dict[str, Any]],
) -> None:
    """Write the full raw results and clean summary to a JSON file."""
    output = {
        "summary": task_results,
        "raw": raw_results,
    }
    # Convert any non-serializable values
    def _default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=_default)
    logger.info("results_written", path=str(path))


def _write_summary_md(
    path: Path,
    task_results: dict[str, dict[str, Any]],
    num_fewshot: int,
) -> None:
    """Write a human-readable Markdown summary table."""
    lines = [
        "# Evaluation Summary",
        "",
        f"**Few-shot:** {num_fewshot}",
        "",
        "| Task | Metric | Value |",
        "|------|--------|-------|",
    ]
    for task_name, metrics in sorted(task_results.items()):
        for metric_name, value in sorted(metrics.items()):
            if isinstance(value, float):
                formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            lines.append(f"| {task_name} | {metric_name} | {formatted} |")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("summary_written", path=str(path))
