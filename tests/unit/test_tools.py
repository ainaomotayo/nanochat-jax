"""Tests for nanochat.inference.tools — calculator and tool detection."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from nanochat.inference.tools import (
    ToolRegistry,
    default_registry,
    detect_and_execute_tools,
    safe_calculator,
)


# ------------------------------------------------------------------
# Calculator tests
# ------------------------------------------------------------------


class TestCalculatorBasicArithmetic:
    def test_addition(self):
        assert safe_calculator("2 + 3") == "5"

    def test_subtraction(self):
        assert safe_calculator("10 - 4") == "6"

    def test_multiplication(self):
        assert safe_calculator("6 * 7") == "42"

    def test_division(self):
        assert safe_calculator("15 / 4") == "3.75"

    def test_power(self):
        assert safe_calculator("2 ** 10") == "1024"

    def test_modulo(self):
        assert safe_calculator("17 % 5") == "2"

    def test_parentheses(self):
        assert safe_calculator("(2 + 3) * 4") == "20"

    def test_float(self):
        assert safe_calculator("3.14 * 2") == "6.28"

    def test_negative_number(self):
        assert safe_calculator("-5 + 3") == "-2"

    def test_complex_expression(self):
        assert safe_calculator("(10 + 5) * 2 - 3 ** 2") == "21"


class TestCalculatorRejectsDangerousInput:
    def test_rejects_function_call(self):
        with pytest.raises(ValueError, match="Disallowed"):
            safe_calculator("__import__('os').system('rm -rf /')")

    def test_rejects_variable_name(self):
        with pytest.raises(ValueError, match="Disallowed"):
            safe_calculator("x + 1")

    def test_rejects_builtin(self):
        with pytest.raises(ValueError, match="Disallowed"):
            safe_calculator("eval('1+1')")

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError, match="Disallowed"):
            safe_calculator("os.system('echo hi')")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Empty"):
            safe_calculator("")

    def test_rejects_list_comprehension(self):
        with pytest.raises(ValueError):
            safe_calculator("[x for x in range(10)]")

    def test_rejects_lambda(self):
        with pytest.raises(ValueError):
            safe_calculator("(lambda: 1)()")

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_calculator("1 / 0")


# ------------------------------------------------------------------
# Tool detection / injection tests
# ------------------------------------------------------------------


class TestToolDetectionPattern:
    def test_detects_tool_call(self):
        text = "Let me compute that: <|tool_call|>calculator(2 + 3)<|tool_end|>"
        processed, called = detect_and_execute_tools(text)
        assert called is True
        assert "<|tool_result|>5<|tool_end|>" in processed

    def test_no_tool_call(self):
        text = "This is plain text with no tool calls."
        processed, called = detect_and_execute_tools(text)
        assert called is False
        assert processed == text

    def test_preserves_surrounding_text(self):
        text = "Before <|tool_call|>calculator(10 * 5)<|tool_end|> after"
        processed, called = detect_and_execute_tools(text)
        assert called is True
        assert processed.startswith("Before ")
        assert processed.endswith(" after")
        assert "<|tool_result|>50<|tool_end|>" in processed

    def test_unknown_tool(self):
        text = "<|tool_call|>unknown_tool(abc)<|tool_end|>"
        processed, called = detect_and_execute_tools(text)
        assert called is True
        assert "Error" in processed


class TestToolResultInjection:
    def test_result_follows_call(self):
        text = "<|tool_call|>calculator(7 + 8)<|tool_end|>"
        processed, _ = detect_and_execute_tools(text)
        # Result should immediately follow the tool_end of the call
        expected_call = "<|tool_call|>calculator(7 + 8)<|tool_end|>"
        expected_result = "<|tool_result|>15<|tool_end|>"
        assert expected_call + expected_result == processed

    def test_multiple_calls(self):
        text = (
            "<|tool_call|>calculator(1 + 1)<|tool_end|> and "
            "<|tool_call|>calculator(2 * 3)<|tool_end|>"
        )
        processed, called = detect_and_execute_tools(text)
        assert called is True
        assert "<|tool_result|>2<|tool_end|>" in processed
        assert "<|tool_result|>6<|tool_end|>" in processed


# ------------------------------------------------------------------
# Registry tests
# ------------------------------------------------------------------


class TestToolRegistry:
    def test_register_and_execute(self):
        reg = ToolRegistry()
        reg.register("echo", lambda s: s, "Echo input")
        assert reg.has_tool("echo")
        assert reg.execute("echo", "hello") == "hello"

    def test_unknown_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="Unknown tool"):
            reg.execute("nope", "")

    def test_default_registry_has_calculator(self):
        reg = default_registry()
        assert reg.has_tool("calculator")
        assert reg.execute("calculator", "3 + 4") == "7"

    def test_tool_names(self):
        reg = ToolRegistry()
        reg.register("a", lambda s: s)
        reg.register("b", lambda s: s)
        assert sorted(reg.tool_names) == ["a", "b"]
