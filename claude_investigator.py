"""
Claude Agent SDK: the optional "investigator" for a pod labeled Watch.

The agent can only use two read-only tools we define (served as an in-process MCP server):
get_pod_evidence and compare_to_peers. Built-in tools that write files, run shell commands or
use the web are blocked, permission_mode="dontAsk" denies anything not pre-approved, and the
run stops at max_turns or max_budget_usd. A PreToolUse hook logs every tool call before it runs.

Needs: ANTHROPIC_API_KEY (the SDK bundles the Claude Code binary it runs).
Run:   python claude_investigator.py
"""
import asyncio
import json

from claude_agent_sdk import (ClaudeAgentOptions, HookMatcher, ResultMessage, create_sdk_mcp_server,
                              query, tool)

from common import SYSTEM, all_pods, get_evidence


@tool("get_pod_evidence", "Evidence packet for one pod this week.", {"pod": str})
async def get_pod_evidence(args):
    return {"content": [{"type": "text", "text": json.dumps(get_evidence(args["pod"]))}]}


@tool("compare_to_peers", "Evidence packets for every other active pod, to check whether a loss is market-wide.", {"pod": str})
async def compare_to_peers(args):
    peers = {p: get_evidence(p) for p in all_pods() if p != args["pod"]}
    return {"content": [{"type": "text", "text": json.dumps(peers)}]}


risk_tools = create_sdk_mcp_server(name="risk", tools=[get_pod_evidence, compare_to_peers])


async def log_tool_call(input_data, tool_use_id, context):
    print(f"  [audit] {input_data['tool_name']} {json.dumps(input_data['tool_input'])}")
    return {}  # empty = allow; returning a deny decision here would block the call


options = ClaudeAgentOptions(
    system_prompt=SYSTEM + " You are investigating one pod in depth. Finish with a short memo: "
                  "label, confidence, recommended action, and the evidence behind it.",
    mcp_servers={"risk": risk_tools},
    allowed_tools=["mcp__risk__get_pod_evidence", "mcp__risk__compare_to_peers"],
    disallowed_tools=["Bash", "Write", "Edit", "WebSearch", "WebFetch"],
    permission_mode="dontAsk",
    max_turns=8,
    max_budget_usd=0.25,
    hooks={"PreToolUse": [HookMatcher(hooks=[log_tool_call])]},
)


async def main():
    async for message in query(prompt="Investigate pod C_event. Is it broken or unlucky?", options=options):
        if isinstance(message, ResultMessage):
            print(f"\nstatus: {message.subtype}, turns: {message.num_turns}, cost: ${message.total_cost_usd or 0:.4f}")
            if message.subtype == "success":
                print(message.result)


asyncio.run(main())
