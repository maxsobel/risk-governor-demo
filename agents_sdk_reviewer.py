"""
OpenAI Agents SDK: the same review as an Agent, with the loop run by Runner.

Shows:
- @function_tool turning a Python function into a tool (schema built from the type hints)
- output_type as a Pydantic model, so the run only ends with a valid Verdict
- an output guardrail: a "close" recommendation with confidence under 0.7 trips the guardrail
  and stops the run, instead of passing a weak closure on to the enforcer

Needs: OPENAI_API_KEY. Optional: OPENAI_MODEL (default gpt-5-mini).
Run:   python agents_sdk_reviewer.py
"""
import os
from typing import Literal

from agents import (Agent, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered, Runner,
                    function_tool, output_guardrail)
from pydantic import BaseModel

from common import SYSTEM, get_evidence


class Verdict(BaseModel):
    label: Literal["Healthy", "Unlucky", "Watch", "Broken"]
    confidence: float
    action: Literal["hold", "cut", "close"]
    rationale: str


@function_tool
def get_pod_evidence(pod: str) -> dict:
    """Return this week's evidence packet for a pod (Sharpe, drawdown, mean-shift t-stat, hit rates)."""
    return get_evidence(pod)


@output_guardrail
async def no_weak_closures(ctx, agent, output: Verdict) -> GuardrailFunctionOutput:
    weak_close = output.action == "close" and output.confidence < 0.7
    return GuardrailFunctionOutput(output_info={"weak_close": weak_close}, tripwire_triggered=weak_close)


reviewer = Agent(
    name="Risk reviewer",
    instructions=SYSTEM + " Always call get_pod_evidence before deciding.",
    tools=[get_pod_evidence],
    output_type=Verdict,
    output_guardrails=[no_weak_closures],
    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
)

for pod in ["A_momentum", "C_event"]:
    try:
        result = Runner.run_sync(reviewer, f"Review pod {pod}.", max_turns=5)
        v: Verdict = result.final_output
        print(f"{pod}: {v.label} ({v.confidence:.2f}) -> {v.action}. {v.rationale}")
    except OutputGuardrailTripwireTriggered:
        print(f"{pod}: guardrail stopped a low-confidence close; sending to a human instead")
