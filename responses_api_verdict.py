"""
OpenAI Responses API: two direct calls, no framework.

Part 1: one request with a strict JSON schema, so the verdict always parses.
Part 2: the function-calling loop done by hand. The model asks for evidence with a
        function_call, we run the function, and send back a function_call_output with
        the same call_id. This loop is what the agent SDKs automate.

Needs: OPENAI_API_KEY. Optional: OPENAI_MODEL (default gpt-5-mini).
Run:   python responses_api_verdict.py
"""
import json
import os

from openai import OpenAI

from common import SYSTEM, VERDICT_SCHEMA, get_evidence

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
VERDICT_FORMAT = {"format": {"type": "json_schema", "name": "verdict", "schema": VERDICT_SCHEMA, "strict": True}}

# ---- Part 1: evidence passed in directly, structured output back
for pod in ["A_momentum", "C_event"]:
    resp = client.responses.create(
        model=MODEL,
        instructions=SYSTEM,
        input=json.dumps(get_evidence(pod)),
        text=VERDICT_FORMAT,
        store=False,  # stateless: each review is rebuilt from our data, not OpenAI's stored history
    )
    print(pod, "->", json.loads(resp.output_text))

# ---- Part 2: the model fetches evidence itself through a function tool
tools = [{
    "type": "function",
    "name": "get_pod_evidence",
    "description": "Return this week's evidence packet (Sharpe, drawdown, mean-shift t-stat, hit rates) for a pod.",
    "parameters": {"type": "object", "properties": {"pod": {"type": "string"}},
                   "required": ["pod"], "additionalProperties": False},
    "strict": True,
}]
history = [{"role": "user", "content": "Review pod C_event. Fetch its evidence first."}]

while True:
    resp = client.responses.create(model=MODEL, instructions=SYSTEM, input=history,
                                   tools=tools, text=VERDICT_FORMAT, store=False)
    calls = [item for item in resp.output if item.type == "function_call"]
    if not calls:
        break
    history += resp.output                      # keep the model's function_call items (and reasoning items)
    for call in calls:
        result = get_evidence(**json.loads(call.arguments))
        print(f"model called {call.name}({call.arguments})")
        history.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)})

print("C_event (tool loop) ->", json.loads(resp.output_text))
