"""Tool use support for nanochat-jax inference.

Provides a registry for callable tools and detection/execution of tool
calls embedded in generated text.  The built-in calculator tool safely
evaluates arithmetic expressions via AST inspection (never ``eval``).
"""
from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# Patterns for tool call detection in generated text.
TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>(\w+)\(([^)]*)\)<\|tool_end\|>"
)
TOOL_RESULT_TEMPLATE = "<|tool_result|>{result}<|tool_end|>"

# ---------------------------------------------------------------------------
# Safe calculator
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_ALLOWED_UNARYOPS: dict[type, Callable] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> int | float:
    """Recursively evaluate an AST node containing only arithmetic."""
    # Python 3.8+: ast.Constant replaces ast.Num
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # Legacy ast.Num for older Python (kept for safety)
    if isinstance(node, ast.Num):  # type: ignore[attr-defined]
        return node.n  # type: ignore[attr-defined]
    if isinstance(node, ast.BinOp):
        op_func = _ALLOWED_BINOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        op_func = _ALLOWED_UNARYOPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))
    # Reject everything else (Name, Call, Attribute, etc.)
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def safe_calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression and return the result as a string.

    Supports ``+``, ``-``, ``*``, ``/``, ``**``, ``%``, parentheses,
    integers, and floats.  Rejects any expression containing variable
    names, function calls, attribute access, or other non-arithmetic
    constructs.

    Raises:
        ValueError: If the expression is invalid or contains disallowed constructs.
    """
    expression = expression.strip()
    if not expression:
        raise ValueError("Empty expression")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression syntax: {exc}") from exc

    # Walk the entire AST and reject anything that is not a number,
    # binary op, or unary op.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Constant, ast.Num,  # type: ignore[attr-defined]
                             ast.BinOp, ast.UnaryOp)):
            continue
        # Allow the operator type nodes themselves (Add, Sub, etc.)
        if isinstance(node, tuple(_ALLOWED_BINOPS.keys()) + tuple(_ALLOWED_UNARYOPS.keys())):
            continue
        raise ValueError(f"Disallowed construct in expression: {type(node).__name__}")

    result = _safe_eval_node(tree.body)
    return str(result)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Registry of callable tools that the model can invoke.

    Usage::

        registry = ToolRegistry()
        registry.register("calculator", safe_calculator, "Evaluate arithmetic expressions")
        result = registry.execute("calculator", "2 + 3")
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[Callable[..., str], str]] = {}

    def register(self, name: str, func: Callable[..., str], description: str = "") -> None:
        """Register a tool by name."""
        self._tools[name] = (func, description)
        logger.info("tool_registered", name=name, description=description)

    def execute(self, name: str, args_str: str) -> str:
        """Execute a registered tool and return its string result.

        Args:
            name: Tool name.
            args_str: Raw argument string (passed to the tool function).

        Returns:
            Tool output as a string.

        Raises:
            KeyError: If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        func, _ = self._tools[name]
        logger.info("tool_execute", name=name, args=args_str)
        return func(args_str)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools


def default_registry() -> ToolRegistry:
    """Return a :class:`ToolRegistry` with the built-in calculator tool."""
    registry = ToolRegistry()
    registry.register("calculator", safe_calculator, "Evaluate arithmetic expressions")
    return registry


# ---------------------------------------------------------------------------
# Detection and execution helpers
# ---------------------------------------------------------------------------

def detect_and_execute_tools(
    text: str,
    registry: ToolRegistry | None = None,
) -> tuple[str, bool]:
    """Scan *text* for tool-call patterns, execute them, and inject results.

    Tool calls have the form::

        <|tool_call|>tool_name(args)<|tool_end|>

    Each matched call is replaced in-place with::

        <|tool_call|>tool_name(args)<|tool_end|><|tool_result|>result<|tool_end|>

    Args:
        text: Model-generated text potentially containing tool calls.
        registry: Tool registry to use.  Falls back to :func:`default_registry`.

    Returns:
        A ``(processed_text, tool_was_called)`` tuple.
    """
    if registry is None:
        registry = default_registry()

    tool_was_called = False

    def _replace(match: re.Match) -> str:
        nonlocal tool_was_called
        tool_name = match.group(1)
        args_str = match.group(2)
        try:
            result = registry.execute(tool_name, args_str)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            result = f"Error: {exc}"
        tool_was_called = True
        original = match.group(0)
        return original + TOOL_RESULT_TEMPLATE.format(result=result)

    processed = TOOL_CALL_RE.sub(_replace, text)
    return processed, tool_was_called
