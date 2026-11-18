"""LLM Executor (W2-04): sequential step execution, ADR-025 math routing, and safety guards."""
from __future__ import annotations

async def executor_node(state: AgentState) -> dict[str, Any]:
    """
    Executor node with W2-04 safety guards:
    - Step limit enforcement (max_steps)
    - Token tracking and budget check (max_tokens via core.tokens)
    - Duplicate tool-call ban
    - ADR-025 math routing
    """
    plan = state.get("plan")
    if not plan:
        return {}

    if getattr(plan, "refuse", False):
        return {}

    tool_results = list(state.get("tool_results") or [])
    steps = getattr(plan, "steps", [])
    
    # Read limits from state["limits"]
    limits_dict = state.get("limits") or {}
    max_steps = limits_dict.get("max_steps")
    max_tokens = limits_dict.get("max_tokens")
    
    current_tokens = state.get("tokens") or 0
    
    # Track executed tool calls in this run to detect duplicates
    seen_calls = set()
    for res in tool_results:
        # If tool results contain metadata about the call, track them
        pass

    # Helper to check limits
    def check_limits() -> str | None:
        if len(tool_results) >= max_steps:
            return f"step limit reached ({len(tool_results)} tool calls, at most {max_steps})"
        if current_tokens > max_tokens:
            return f"token budget exceeded ({current_tokens} tokens, at most {max_tokens})"
        return None

    # 1. Execute plan steps sequentially with guards
    for step in steps:
        # Guard: Check step/token limits before each step
        stop_reason = check_limits()
        if stop_reason:
            return {"tool_results": tool_results, "stop_reason": stop_reason}

        tool_name = getattr(step, "tool", None) or step.get("tool")
        arguments = getattr(step, "arguments", {}) or step.get("arguments", {})
        
        # Guard: Duplicate tool-call ban
        call_signature = (tool_name, frozenset(arguments.items()) if isinstance(arguments, dict) else str(arguments))
        if call_signature in seen_calls:
            log.warning("Duplicate tool call detected and blocked: %s with %s", tool_name, arguments)
            return {
                "tool_results": tool_results, 
                "tool_error": f"Duplicate tool call blocked: {tool_name}"
            }
        seen_calls.add(call_signature)
        
        try:
            tool_fn = get_tool(tool_name)
            if not tool_fn:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(**arguments)
            else:
                result = tool_fn(**arguments)
                
            tool_results.append(result)
            
            # Token tracking integration using core.tokens (if available in result or environment)
            from core.tokens import count_text_tokens
            if isinstance(result, str):
                current_tokens += count_text_tokens(result)
            elif isinstance(result, dict) and "text" in result:
                current_tokens += count_text_tokens(str(result["text"]))
                
        except Exception as e:
            log.exception("Tool execution failed for %s", tool_name)
            return {"tool_results": tool_results, "tool_error": str(e), "tokens": current_tokens}

    # 2. ADR-025: Automatic calculation routing if plan.calculation is set
    calc_step = route_calculation_step(plan, tool_results)
    if calc_step:
        stop_reason = check_limits()
        if stop_reason:
            return {"tool_results": tool_results, "stop_reason": stop_reason, "tokens": current_tokens}
            
        try:
            calc_tool_name = calc_step["tool"]
            calc_args = calc_step["arguments"]
            calc_fn = get_tool(calc_tool_name)
            if not calc_fn:
                raise ValueError(f"Unknown tool: {calc_tool_name}")
            
            if asyncio.iscoroutinefunction(calc_fn):
                calc_result = await calc_fn(**calc_args)
            else:
                calc_result = calc_fn(**calc_args)
                
            tool_results.append(calc_result)
        except Exception as e:
            log.exception("Calculation tool execution failed")
            return {"tool_results": tool_results, "tool_error": str(e), "tokens": current_tokens}

    return {"tool_results": tool_results, "tool_error": None, "tokens": current_tokens}