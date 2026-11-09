"""Tier 5: policy reasoning with lookup tools.

The tools are called through a JSON protocol in plain text (`{"tool": ...,
"args": {...}}` per turn) rather than each server's native tool-calling API.
That keeps the instruction identical for every model and every serving stack
(vLLM, llama.cpp and Ollama parse native tool calls differently per chat
template), so the measurement is the model's reasoning, not its parser.

Tool results come from the complaint's `tool_data`, so a run is deterministic.
On cloud routes every tool result is pseudonymised with the same mapping as
the complaint before it is sent, and placeholders in tool arguments are
restored before the lookup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .pseudonymise import Pseudonymiser
from .schemas import Complaint
from .tiers import extract_json

MAX_STEPS = 6  # model turns; the prompt allows 5 tool calls plus the final answer
TOOLS: dict[str, tuple[str, ...]] = {
    "get_order": ("order_number",),
    "get_contact_history": ("customer_id",),
}
NUDGE = 'Answer with one JSON object: a tool call {"tool": ..., "args": {...}} or {"final": {...}}.'


@dataclass
class LoopResult:
    final_text: str | None  # the model output that held {"final": ...}; None if it never decided
    transcript: list[dict[str, str]] = field(default_factory=list)  # messages as sent (pseudonymised)
    raw_outputs: list[str] = field(default_factory=list)
    tool_calls: int = 0
    valid_tool_calls: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_id: str | None = None
    mapping: dict[str, str] = field(default_factory=dict)


def valid_call(name: Any, args: Any) -> bool:
    """A known tool with exactly its required string arguments."""
    return (
        name in TOOLS
        and isinstance(args, dict)
        and set(args) == set(TOOLS[name])
        and all(isinstance(args[k], str) and args[k].strip() for k in TOOLS[name])
    )


def execute(name: str, args: dict[str, str], c: Complaint) -> dict[str, Any]:
    if name == "get_order":
        order = (c.tool_data.get("orders") or {}).get(args["order_number"].strip())
        return order if order is not None else {"error": f"order {args['order_number']} not found"}
    contacts = (c.tool_data.get("contacts") or {}).get(args["customer_id"].strip())
    return {"customer_id": args["customer_id"], "contacts": contacts or []}


def run_tool_loop(
    client,
    system: str,
    first_user: str,
    c: Complaint,
    tier: int = 5,
    pseudonymiser: Pseudonymiser | None = None,
    mapping: dict[str, str] | None = None,
) -> LoopResult:
    """Drive the model until it returns {"final": ...} or MAX_STEPS turns are used.

    `pseudonymiser` is set for cloud routes: tool results are then redacted
    with (and extend) `mapping` before they are sent.
    """
    res = LoopResult(None, [{"role": "user", "content": first_user}], mapping=dict(mapping or {}))
    for _ in range(MAX_STEPS):
        comp = client.chat(system, list(res.transcript), tier=tier)
        res.latency_s += comp.latency_s
        res.prompt_tokens += comp.prompt_tokens or 0
        res.completion_tokens += comp.completion_tokens or 0
        res.model_id = comp.model_id
        res.raw_outputs.append(comp.text)
        res.transcript.append({"role": "assistant", "content": comp.text})
        obj = extract_json(comp.text)
        if obj is not None and isinstance(obj.get("final"), dict):
            res.final_text = comp.text
            return res
        if obj is not None and "tool" in obj:
            res.tool_calls += 1
            name, args = obj.get("tool"), obj.get("args")
            if valid_call(name, args):
                res.valid_tool_calls += 1
                real_args = {k: Pseudonymiser.restore(v, res.mapping) for k, v in args.items()}
                result = execute(name, real_args, c)
            else:
                result = {"error": f"invalid tool call; available tools and arguments: {dict(TOOLS)}"}
            reply = json.dumps({"tool_result": result}, ensure_ascii=False)
            if pseudonymiser is not None:
                reply, res.mapping = pseudonymiser.redact(reply, c.language, res.mapping)
        else:
            reply = NUDGE
        res.transcript.append({"role": "user", "content": reply})
    return res
