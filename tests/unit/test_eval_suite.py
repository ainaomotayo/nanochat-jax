"""Tests for the evaluation suite configuration."""
from __future__ import annotations


def test_quick_tasks_defined():
    """QUICK_TASKS should be a non-empty list of known benchmark names."""
    from nanochat.evaluation.suite import QUICK_TASKS

    assert isinstance(QUICK_TASKS, list)
    assert len(QUICK_TASKS) > 0
    for task in QUICK_TASKS:
        assert isinstance(task, str)
        assert len(task) > 0

    # Verify expected tasks are present
    expected = {"hellaswag", "arc_easy", "piqa", "lambada_openai"}
    assert expected == set(QUICK_TASKS)


def test_standard_tasks_superset_of_quick():
    """STANDARD_TASKS must contain every task in QUICK_TASKS, plus more."""
    from nanochat.evaluation.suite import QUICK_TASKS, STANDARD_TASKS

    assert isinstance(STANDARD_TASKS, list)
    assert len(STANDARD_TASKS) > len(QUICK_TASKS)

    quick_set = set(QUICK_TASKS)
    standard_set = set(STANDARD_TASKS)
    assert quick_set.issubset(standard_set), (
        f"QUICK_TASKS has items not in STANDARD_TASKS: {quick_set - standard_set}"
    )

    # Verify the additional tasks
    extra = standard_set - quick_set
    assert "arc_challenge" in extra
    assert "winogrande" in extra
