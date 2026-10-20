# Risk governor demo: Decision & Agent APIs / Agent Framework

Max Sobel. This is the demo for section 5. There is one runnable piece (the LangGraph loop) and three small snippets showing the same pod review written with each of the other tools.

## Setup

```
pip install -r requirements.txt
```

## Files

**risk_loop_langgraph.py** (runs offline, no API key needed)

This is the weekly risk loop from section 4, written as a LangGraph graph:

simulate_week → compute_evidence → review_pod (one per pod, using Send) → enforce → record → next week

- There are 4 simulated pods over 104 weeks. C_event's edge is switched off at week 30. The others never break.
- The ground truth is kept outside the graph state, so the reviewer can't see it. It is only used for the scorecard at the end.
- compute_evidence computes everything in code: full and recent Sharpe, drawdown compared with the pod's own volatility, a t-stat for whether the mean return shifted (last 13 weeks vs. before), and hit rates.
- review_pod is the only step that could use an LLM. By default it uses a simple rule-based stand-in so the script runs without a key. Setting `LLM_MODEL` (for example `LLM_MODEL=anthropic/claude-sonnet-4-5` with `ANTHROPIC_API_KEY`) sends the review to a real model through LiteLLM instead.
- enforce applies the -5% / -7.5% stopout no matter what the reviewer says. The reviewer can only reduce risk. A recommended close pauses the graph with interrupt() until a person approves it. Run with `--interactive` to approve by hand; otherwise it auto-approves.
- The InMemorySaver checkpointer saves state after every step. At the end the script reads the state back at week 52 to show that a run can be replayed. The real project would use the Postgres checkpointer instead.

Output with the default seed (5):
- The reviewer labels C_event Unlucky, then Watch at week 32, then Broken at week 36. The pod is closed that week.
- The stopout rule alone would have closed it at week 37, so the reviewer was one week earlier. That is a small gain, partly because it only reviews every 4 weeks.
- None of the three healthy pods are closed. A_momentum has a real drawdown around week 36 but stays open, because its mean-shift t-stat is only -0.9.

I picked seed 5 because it shows each case. I tried seeds 1-40, and in 11 of them the reviewer closes C_event without closing a healthy pod. With other seeds the stopout fires first, or a healthy pod gets closed. This is a toy simulation, not evidence that the method works.

**responses_api_verdict.py** (needs OPENAI_API_KEY)

Direct Responses API calls with no framework:
1. A single call with a strict JSON schema, so the verdict always parses.
2. The function-calling loop written by hand. The model returns a function_call, we run it, and we send back a function_call_output with the same call_id. The other tools automate this loop.

**agents_sdk_reviewer.py** (needs OPENAI_API_KEY)

The same review as an OpenAI Agents SDK Agent. A @function_tool supplies the evidence, and output_type is a Pydantic Verdict. An output guardrail stops the run if the agent recommends "close" with confidence under 0.7.

**claude_investigator.py** (needs ANTHROPIC_API_KEY)

The optional investigator from section 4, built with the Claude Agent SDK. It can only use two read-only tools that we define (served as an in-process MCP server). Built-in tools for shell, file editing and web are blocked, `permission_mode="dontAsk"` denies anything not pre-approved, and the run is capped by max_turns and max_budget_usd. A PreToolUse hook prints every tool call before it runs.

**common.py, sample_evidence.json**

The shared prompt and verdict schema, plus the evidence packets the LangGraph run computed at week 36. The three snippets use this file so they all review the same numbers. At week 36, A_momentum is losing but within normal variance and C_event is broken.

## What I tested

- risk_loop_langgraph.py runs end to end offline.
- Its LiteLLM path was tested with a stubbed model call.
- The three API snippets build their clients, agents and options without errors, but I haven't run them against the real APIs because I don't have keys in this environment.
- Model names are set by environment variable, so they can be changed without editing code.
