"""Shared pieces: the verdict schema, the reviewer prompt, and sample evidence packets.

sample_evidence.json holds the evidence packets that risk_loop_langgraph.py computed for each
pod at week 36 (seed 5). At that point A_momentum is losing but within normal variance, and
C_event is broken (its edge was switched off at week 30).
"""
import json
from pathlib import Path

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["Healthy", "Unlucky", "Watch", "Broken"]},
        "confidence": {"type": "number"},
        "action": {"type": "string", "enum": ["hold", "cut", "close"]},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "action", "rationale"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are the risk manager at a multi-manager fund. Given one pod's evidence packet, decide whether "
    "a losing pod is BROKEN (its edge is gone) or just UNLUCKY (normal variance for its volatility). "
    "Losses alone prove nothing; look for drawdowns large relative to the pod's own volatility "
    "(drawdown_vs_normal), a shift in the mean return (mean_shift_t_stat), and a drop in hit rate. "
    "Use Watch when the evidence conflicts. Cite the numbers in your rationale."
)

_DATA = json.loads((Path(__file__).parent / "sample_evidence.json").read_text())


def get_evidence(pod: str) -> dict:
    return {"pod": pod, "week": _DATA["week"], **_DATA["evidence"][pod]}


def all_pods() -> list[str]:
    return list(_DATA["evidence"])
