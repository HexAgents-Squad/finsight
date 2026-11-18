"""Unit tests for the production LLM Executor (W2-04)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from agent.executor import executor_node, route_calculation_step
from agent.state import Plan, PlanStep


@pytest.mark.asyncio
async def test_route_calculation_step():
    """Test that ADR-025 math routing correctly extracts figures in chronological order."""
    plan = Plan(
        reasoning="Calculate growth",
        refuse=False,
        steps=[PlanStep(tool="get_financial_fact", arguments={"company": "AAPL", "metric": "revenue", "year": 2023})],
        calculation={"type": "growth"}
    )
    tool_results = [{"figure": 100}, {"figure": 120}]
    calc_step = route_calculation_step(plan, tool_results)
    
    assert calc_step is not None
    assert calc_step["tool"] == "calculate"
    assert calc_step["arguments"]["operation"] == "growth"
    assert calc_step["arguments"]["operands"] == [100, 120]


@pytest.mark.asyncio
async def test_executor_step_limit_guard():
    """Test that execution halts when max_steps limit is tripped."""
    plan = Plan(
        reasoning="Exceed step limit",
        refuse=False,
        steps=[
            PlanStep(tool="dummy_tool", arguments={"arg": 1}),
            PlanStep(tool="dummy_tool", arguments={"arg": 2}),
        ],
        calculation=None
    )
    # State with max_steps set strictly to 1, but 2 steps planned
    state = {
        "plan": plan,
        "tool_results": [],
        "limits": {"max_steps": 1, "max_tokens": 12000, "timeout_s": 60.0}
    }

    with patch("agent.executor.get_tool") as mock_get_tool:
        mock_tool = AsyncMock(return_value={"figure": 50})
        mock_get_tool.return_value = mock_tool

        result = await executor_node(state)
        
        # Should stop before completing all steps due to max_steps = 1
        assert "stop_reason" in result
        assert "step limit reached" in result["stop_reason"]


@pytest.mark.asyncio
async def test_executor_duplicate_tool_call_ban():
    """Test that duplicate tool calls are blocked immediately."""
    plan = Plan(
        reasoning="Duplicate calls",
        refuse=False,
        steps=[
            PlanStep(tool="get_financial_fact", arguments={"company": "AAPL", "metric": "revenue", "year": 2024}),
            PlanStep(tool="get_financial_fact", arguments={"company": "AAPL", "metric": "revenue", "year": 2024}),
        ],
        calculation=None
    )
    state = {
        "plan": plan,
        "tool_results": [],
        "limits": {"max_steps": 10, "max_tokens": 12000, "timeout_s": 60.0}
    }

    with patch("agent.executor.get_tool") as mock_get_tool:
        mock_tool = AsyncMock(return_value={"figure": 100})
        mock_get_tool.return_value = mock_tool

        result = await executor_node(state)
        
        # Second identical call should trigger a tool_error blocking duplication
        assert result.get("tool_error") is not None
        assert "Duplicate tool call blocked" in result["tool_error"]


@pytest.mark.asyncio
async def test_executor_tool_error_recovery():
    """Test that tool errors are caught gracefully without raising unhandled exceptions."""
    plan = Plan(
        reasoning="Failing tool",
        refuse=False,
        steps=[PlanStep(tool="failing_tool", arguments={})],
        calculation=None
    )
    state = {
        "plan": plan,
        "tool_results": [],
        "limits": {"max_steps": 10, "max_tokens": 12000, "timeout_s": 60.0}
    }

    with patch("agent.executor.get_tool") as mock_get_tool:
        mock_tool = AsyncMock(side_effect=Exception("Database connection error"))
        mock_get_tool.return_value = mock_tool

        result = await executor_node(state)
        
        # Should catch exception and record tool_error gracefully
        assert result.get("tool_error") == "Database connection error"
        assert isinstance(result.get("tool_results"), list)