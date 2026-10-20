"""
Risk governor demo: the weekly loop as a LangGraph graph.

Graph:
    simulate_week -> (review week?) -> compute_evidence -> review_pod (one per pod, via Send)
                  -> enforce -> record -> back to simulate_week, until the horizon ends

- Only review_pod uses an LLM. Every other node is plain deterministic Python.
- The simulator's ground truth (which pod breaks and when) lives outside the graph
  state, so the reviewer can never see it. It is only used for the scorecard at the end.
- An InMemorySaver checkpointer saves state after every step (Postgres in the real project),
  which is what lets us replay any week afterwards.
- If the reviewer recommends closing a pod, enforce() pauses with interrupt() for a human
  to approve. Run with --interactive to approve by hand; otherwise the script auto-approves.

Reviewer:
    By default a rule-based stand-in, so the demo runs offline with no API key.
    Set LLM_MODEL (e.g. LLM_MODEL=anthropic/claude-sonnet-4-5 plus ANTHROPIC_API_KEY)
    to send the review to a real model through LiteLLM instead.

Run:  python risk_loop_langgraph.py [--interactive]
"""
import json
import math
import operator
import os
import sys
from typing import Annotated, TypedDict

import numpy as np
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from common import SYSTEM, VERDICT_SCHEMA

# ---------------------------------------------------------------- simulator (hidden truth)
WEEKS = 104                  # two years of weekly returns
REVIEW_EVERY = 4             # agent reviews pods every 4 weeks
CUT_AT, CLOSE_AT = -0.05, -0.075   # mechanical stopout on drawdown from the pod's peak

PODS = {  # name: (true annual Sharpe, weekly vol, week the edge breaks or None)
    "A_momentum": (1.2, 0.008, None),
    "B_stat_arb": (1.0, 0.007, None),
    "C_event":    (1.1, 0.008, 30),     # edge disappears at week 30
    "D_macro":    (0.8, 0.009, None),
}
rng = np.random.default_rng(int(os.getenv("SEED", 5)))  # seed 5 picked because it shows each case
HIDDEN_RETURNS = {}
for name, (sharpe, vol, break_week) in PODS.items():
    mu = sharpe * vol / math.sqrt(52)
    r = rng.normal(mu, vol, WEEKS)
    if break_week is not None:
        r[break_week:] = rng.normal(-0.6 * vol, vol, WEEKS - break_week)  # small negative drift after the break
    HIDDEN_RETURNS[name] = r.tolist()


def stopout_only_close_week(name: str):
    """Baseline for the scorecard: when would the mechanical stopout alone have closed this pod?"""
    capital, peak, alloc = 1.0, 1.0, 1.0
    for t, r in enumerate(HIDDEN_RETURNS[name]):
        capital *= 1 + r * alloc
        peak = max(peak, capital)
        dd = capital / peak - 1
        if dd <= CLOSE_AT:
            return t + 1
        if dd <= CUT_AT:
            alloc = 0.5
    return None


# ---------------------------------------------------------------- graph state
def merge_verdicts(left: list, right: list | str) -> list:
    """Reducer: parallel review_pod branches append; record() resets with 'RESET'."""
    if right == "RESET":
        return []
    return (left or []) + right


class State(TypedDict):
    week: int
    pods: dict          # name -> {"returns": [...], "capital": float, "peak": float, "status": str}
    evidence: dict      # name -> evidence packet for this review
    verdicts: Annotated[list, merge_verdicts]
    log: Annotated[list, operator.add]


class PodTask(TypedDict):   # what one review_pod branch receives
    pod: str
    evidence: dict


# ---------------------------------------------------------------- deterministic nodes
def simulate_week(state: State) -> dict:
    t = state["week"]
    pods = json.loads(json.dumps(state["pods"]))  # copy
    for name, p in pods.items():
        if p["status"] == "closed":
            continue
        r = HIDDEN_RETURNS[name][t]
        p["returns"].append(r)
        p["capital"] *= 1 + r * p["alloc"]
        p["peak"] = max(p["peak"], p["capital"])
    return {"pods": pods, "week": t + 1}


def compute_evidence(state: State) -> dict:
    """All the statistics live here, in code. The LLM only reads the results."""
    ev = {}
    for name, p in state["pods"].items():
        if p["status"] == "closed":
            continue
        r = np.array(p["returns"])
        recent, before = r[-13:], r[:-13]
        vol = r.std(ddof=1) if len(r) > 2 else 1e-9
        # did the mean change? two-sample t-stat, last 13 weeks vs everything before
        if len(before) >= 13:
            se = math.sqrt(recent.var(ddof=1) / len(recent) + before.var(ddof=1) / len(before))
            shift_t = (recent.mean() - before.mean()) / se
        else:
            shift_t = None  # not enough history yet
        drawdown = p["capital"] / p["peak"] - 1
        ev[name] = {
            "weeks_of_history": len(r),
            "sharpe_full": round(r.mean() / vol * math.sqrt(52), 2),
            "sharpe_last_13w": round(recent.mean() / recent.std(ddof=1) * math.sqrt(52), 2),
            "drawdown_pct": round(100 * drawdown, 2),
            # drawdown measured in units of what this pod's own volatility would explain over 13 weeks
            "drawdown_vs_normal": round(-drawdown / (vol * math.sqrt(13)), 2),
            "mean_shift_t_stat": round(shift_t, 2) if shift_t is not None else None,
            "hit_rate_last_13w": round(float((recent > 0).mean()), 2),
            "hit_rate_before": round(float((before > 0).mean()), 2) if len(before) else None,
        }
        # numpy floats can't be checkpointed, so convert to plain Python floats
        ev[name] = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in ev[name].items()}
    return {"evidence": ev}


def fan_out_reviews(state: State):
    return [Send("review_pod", {"pod": name, "evidence": e}) for name, e in state["evidence"].items()]


def enforce(state: State) -> dict:
    """Hard limits. The agent can recommend; this code decides."""
    pods = json.loads(json.dumps(state["pods"]))
    proposals = {v["pod"]: v for v in state["verdicts"]}
    log = []

    # human approval for any recommended closure (interrupt pauses the graph here)
    to_close = [n for n, v in proposals.items() if v["action"] == "close" and pods[n]["status"] != "closed"]
    approved = set()
    if to_close:
        answer = interrupt({"week": state["week"], "close_requests": {n: proposals[n]["rationale"] for n in to_close}})
        approved = {n for n in to_close if answer.get(n) is True}

    for name, p in pods.items():
        if p["status"] == "closed":
            continue
        dd = p["capital"] / p["peak"] - 1
        v = proposals.get(name)
        # 1. mechanical stopout always applies, whatever the agent said
        if dd <= CLOSE_AT:
            p["status"], p["alloc"] = "closed", 0.0
            log.append(f"wk {state['week']:>3} {name}: STOPOUT close (dd {dd:.1%})")
            continue
        if dd <= CUT_AT and p["alloc"] > 0.5:
            p["alloc"] = 0.5
            log.append(f"wk {state['week']:>3} {name}: STOPOUT cut to 50% (dd {dd:.1%})")
        # 2. agent proposals, only within limits (it can reduce risk, never add it)
        if v and v["action"] == "close" and name in approved:
            p["status"], p["alloc"] = "closed", 0.0
            log.append(f"wk {state['week']:>3} {name}: CLOSED on agent recommendation")
        elif v and v["action"] == "cut" and p["alloc"] > 0.5:
            p["alloc"] = 0.5
            log.append(f"wk {state['week']:>3} {name}: cut to 50% on agent recommendation")
    return {"pods": pods, "log": log}


def record(state: State) -> dict:
    lines = [f"wk {state['week']:>3} {v['pod']}: {v['label']:<8} conf {v['confidence']:.2f}  {v['rationale']}"
             for v in sorted(state["verdicts"], key=lambda v: v["pod"])]
    return {"verdicts": "RESET", "log": lines}


# ---------------------------------------------------------------- the one LLM node
# The prompt and schema are shared with the API snippets (common.py).
LLM_SYSTEM = SYSTEM + " Reply with JSON only, matching this schema: " + json.dumps(VERDICT_SCHEMA)


def rule_based_reviewer(e: dict) -> dict:
    """Offline stand-in for the LLM so the demo runs without a key."""
    shift, ddn = e["mean_shift_t_stat"], e["drawdown_vs_normal"]
    if shift is None:
        shift = 0.0  # too little history to test for a shift
    if shift < -2.0 and ddn > 1.5:
        return {"label": "Broken", "confidence": 0.8, "action": "close",
                "rationale": f"mean shifted (t={shift}) and drawdown is {ddn}x normal"}
    if shift < -2.0 or ddn > 1.5:
        return {"label": "Watch", "confidence": 0.6, "action": "cut",
                "rationale": f"one warning sign (t={shift}, dd {ddn}x normal)"}
    if e["drawdown_pct"] < -1.0:
        return {"label": "Unlucky", "confidence": 0.6, "action": "hold",
                "rationale": f"losing but within normal range (dd {ddn}x normal, t={shift})"}
    return {"label": "Healthy", "confidence": 0.7, "action": "hold", "rationale": "no warning signs"}


def review_pod(task: PodTask) -> dict:
    model = os.getenv("LLM_MODEL")
    if model:
        import litellm  # model gateway: the model is chosen in config, not in code
        resp = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": LLM_SYSTEM},
                      {"role": "user", "content": json.dumps({"pod": task["pod"], **task["evidence"]})}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip().removeprefix("```json").removesuffix("```")
        verdict = json.loads(text)
    else:
        verdict = rule_based_reviewer(task["evidence"])
    return {"verdicts": [{"pod": task["pod"], **verdict}]}


# ---------------------------------------------------------------- wiring
def after_simulate(state: State) -> str:
    return "compute_evidence" if state["week"] % REVIEW_EVERY == 0 and state["week"] >= 13 else "next_week"


def next_week(state: State) -> str:
    return END if state["week"] >= WEEKS else "simulate_week"


builder = StateGraph(State)
builder.add_node("simulate_week", simulate_week)
builder.add_node("compute_evidence", compute_evidence)
builder.add_node("review_pod", review_pod)
builder.add_node("enforce", enforce)
builder.add_node("record", record)
builder.add_edge(START, "simulate_week")
builder.add_conditional_edges("simulate_week", after_simulate, {"compute_evidence": "compute_evidence", "next_week": "enforce"})
builder.add_conditional_edges("compute_evidence", fan_out_reviews, ["review_pod"])
builder.add_edge("review_pod", "enforce")          # unconditional: nothing skips the enforcer
builder.add_edge("enforce", "record")
builder.add_conditional_edges("record", next_week, {END: END, "simulate_week": "simulate_week"})
graph = builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------- run
if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    config = {"configurable": {"thread_id": "run-1"}, "recursion_limit": 5000}
    start = {"week": 0, "evidence": {}, "verdicts": [], "log": [],
             "pods": {n: {"returns": [], "capital": 1.0, "peak": 1.0, "alloc": 1.0, "status": "active"} for n in PODS}}

    result = graph.invoke(start, config)
    while "__interrupt__" in result:  # paused in enforce() waiting for a human
        req = result["__interrupt__"][0].value
        answers = {}
        for pod, why in req["close_requests"].items():
            print(f"\n[week {req['week']}] agent wants to CLOSE {pod}: {why}")
            ok = input("approve? [y/n] ").strip().lower() == "y" if interactive else True
            print("  -> approved" if ok else "  -> rejected", "(auto)" if not interactive else "")
            answers[pod] = ok
        result = graph.invoke(Command(resume=answers), config)

    print("\n=== decision log ===")
    for line in result["log"]:
        print(line)

    print("\n=== scorecard (uses hidden truth) ===")
    for name, (_, _, brk) in PODS.items():
        p = result["pods"][name]
        truth = f"broken at wk {brk}" if brk is not None else "never broken"
        closed = next((l for l in result["log"] if name in l and ("CLOSED" in l or "STOPOUT close" in l)), None)
        base = stopout_only_close_week(name)
        print(f"{name:<11} truth: {truth:<17} final: {p['status']:<7} capital {p['capital']:.3f}"
              f" | stopout-only baseline closes it: {'wk ' + str(base) if base else 'never'}"
              + (f"\n{'':<12}[{closed.strip()}]" if closed else ""))

    # replay: every step was checkpointed, so we can look at the state at any point in the run
    history = list(graph.get_state_history(config))
    wk52 = next(s for s in history if s.values.get("week") == 52)
    print(f"\n{len(history)} checkpoints saved. State at week 52 (C_event):",
          {k: round(v, 3) if isinstance(v, float) else v
           for k, v in wk52.values["pods"]["C_event"].items() if k != "returns"})
