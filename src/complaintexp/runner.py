"""Run every complaint through every variant and log the result.

One complaint through one variant (`process`):
1. Safeguard: real complaints only reach variants cleared for them.
2. Route: hybrid picks the variant that answers (router model for draft tiers).
3. Preprocess: pseudonymise (pseudonymise, abstract, own_tenant routes), then
   for the abstract route a local model rewrites the complaint as a summary.
4. Call: one request, or the tool loop for tier 5.
5. Parse against the tier's output contract, restore placeholders, score tiers 1/2/5.
6. Count annotated PII still visible in what was sent (cloud routes).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

from .config import Variant
from .evaluate import leaks
from .llm import LLMClient
from .prompting import (build_user_message, load_abstract_prompt, load_router_prompt, load_system_prompt,
                        parse_output)
from .pseudonymise import Pseudonymiser
from .schemas import Complaint
from .tiers import autoscore, extract_json
from .tools import run_tool_loop

WARMUP_TEXT = "# Complaint (language: en)\nWarm-up request, please ignore.\n"


class RealDataBlocked(Exception):
    pass


def check_allowed(v: Variant, c: Complaint) -> None:
    """Real complaints must never reach a variant that is not cleared for them."""
    if c.source == "real" and not v.allow_real_data:
        raise RealDataBlocked(
            f"variant {v.key} is not cleared for real data (complaint {c.id}); "
            "set allow_real_data only after DPO sign-off and only for local variants"
        )
    if c.source == "real" and v.cloud:
        raise RealDataBlocked(f"refusing to send real complaint {c.id} to cloud variant {v.key}")


class Context:
    """Clients and helpers shared by every complaint of one run."""

    def __init__(self, variants: dict[str, Variant], pseudonymiser: Pseudonymiser | None,
                 clients: dict[str, Any] | None = None, con: sqlite3.Connection | None = None):
        self.variants = variants
        self.ps = pseudonymiser
        self.clients = dict(clients or {})
        self.con = con

    def summarise(self, abstractor: str, c: Complaint, redacted: str) -> tuple[str, float]:
        """The abstractor's summary of a pseudonymised complaint, and its model time.

        Cached in the `summaries` table: every API model on the abstract route gets
        the same summary, so differences come from the API model, not the summary.
        The cached model time still counts towards each run's preprocessing latency.
        """
        prompt = load_abstract_prompt()
        key = hashlib.sha256(f"{abstractor}\n{prompt}\n{redacted}".encode()).hexdigest()
        if self.con is not None:
            hit = self.con.execute("SELECT summary, latency_s FROM summaries WHERE key=?", (key,)).fetchone()
            if hit:
                return hit[0], hit[1] or 0.0
        comp = self.client(abstractor).complete(prompt, redacted, tier=None)
        summary = (extract_json(comp.text) or {}).get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("abstractor returned no summary")
        if self.con is not None:
            self.con.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?)",
                             (key, abstractor, c.id, summary, comp.latency_s))
            self.con.commit()
        return summary, comp.latency_s

    def client(self, key: str):
        if key not in self.clients:
            self.clients[key] = LLMClient(self.variants[key])
        return self.clients[key]

    def pseudonymiser(self) -> Pseudonymiser:
        if self.ps is None:
            self.ps = Pseudonymiser()
        return self.ps


def _sub_variants(v: Variant) -> list[str]:
    """Other variants a variant calls: the abstractor, or the hybrid router and targets."""
    if v.route == "hybrid":
        return [k for k in (v.hybrid.get("router"), v.hybrid.get("easy"), v.hybrid.get("hard")) if k]
    return [v.abstractor] if v.pii == "abstract" and v.abstractor else []


def _route_hybrid(v: Variant, c: Complaint, ctx: Context, user: str) -> tuple[str, str | None, float]:
    """Returns (target variant key, router answer, router latency)."""
    h = v.hybrid
    if c.tier in (h.get("hard_tiers") or []):
        return h["hard"], None, 0.0
    if c.tier not in (h.get("router_tiers") or []):
        return h["easy"], None, 0.0
    router = ctx.variants[h["router"]]
    check_allowed(router, c)
    comp = ctx.client(router.key).complete(load_router_prompt(), user, tier=None)
    obj = extract_json(comp.text) or {}
    target = h["hard"] if obj.get("complexity") == "hard" else h["easy"]
    return target, comp.text, comp.latency_s


def _restore(obj: Any, mapping: dict[str, str]) -> Any:
    """Put the real values back into every string of a parsed answer."""
    if isinstance(obj, str):
        return Pseudonymiser.restore(obj, mapping)
    if isinstance(obj, dict):
        return {k: _restore(x, mapping) for k, x in obj.items()}
    if isinstance(obj, list):
        return [_restore(x, mapping) for x in obj]
    return obj


def process(v: Variant, c: Complaint, ctx: Context) -> dict[str, Any]:
    """Answer one complaint with variant `v`. Returns the row fields (raises RealDataBlocked)."""
    row: dict[str, Any] = dict(
        tier=c.tier, route=v.route, model_key=v.model_key, size_class=v.size_class,
        reasoning=v.reasoning, hardware=v.hardware, routed_to=None,
        preprocess_output=None, preprocess_latency_s=None,
    )
    target = v
    pre_s = 0.0
    if v.route == "hybrid":
        key, answer, router_s = _route_hybrid(v, c, ctx, build_user_message(c))
        target = ctx.variants[key]
        row.update(routed_to=key, preprocess_output=answer)
        pre_s += router_s
    check_allowed(target, c)

    system = load_system_prompt(c.tier)
    mapping: dict[str, str] = {}
    redacts = target.pii in ("pseudonymise", "abstract")
    if target.pii == "abstract":
        ps = ctx.pseudonymiser()
        t0 = time.perf_counter()
        redacted, mapping = ps.redact(c.text, c.language)
        pre_s += time.perf_counter() - t0
        summary, abstract_s = ctx.summarise(ctx.variants[target.abstractor].key, c, redacted)
        pre_s += abstract_s
        t0 = time.perf_counter()
        user, mapping = ps.redact(build_user_message(c, text=summary, heading="Case summary"), c.language, mapping)
        pre_s += time.perf_counter() - t0
        row["preprocess_output"] = summary
    elif target.pii == "pseudonymise":
        t0 = time.perf_counter()
        user, mapping = ctx.pseudonymiser().redact(build_user_message(c), c.language)
        pre_s += time.perf_counter() - t0
    else:
        user = build_user_message(c)
    row["preprocess_latency_s"] = pre_s if (redacts or v.route == "hybrid") else None
    row["prompt_sent"] = user

    client = ctx.client(target.key)
    sent = [user]
    if c.tier == 5:
        loop = run_tool_loop(client, system, user, c, pseudonymiser=ctx.pseudonymiser() if redacts else None,
                             mapping=mapping)
        mapping = loop.mapping
        sent += [m["content"] for m in loop.transcript[1:] if m["role"] == "user"]
        text = loop.final_text or (loop.raw_outputs[-1] if loop.raw_outputs else "")
        row.update(model_id=loop.model_id, output_raw="\n---\n".join(loop.raw_outputs),
                   prompt_tokens=loop.prompt_tokens, completion_tokens=loop.completion_tokens,
                   tool_calls=loop.tool_calls, valid_tool_calls=loop.valid_tool_calls,
                   trace_json=json.dumps(loop.transcript, ensure_ascii=False))
        model_s = loop.latency_s
        parsed = parse_output(text if loop.final_text else "", 5)
    else:
        comp = client.complete(system, user, tier=c.tier)
        row.update(model_id=comp.model_id, output_raw=comp.text, prompt_tokens=comp.prompt_tokens,
                   completion_tokens=comp.completion_tokens)
        model_s = comp.latency_s
        parsed = parse_output(comp.text, c.tier)

    restored = _restore(parsed.obj, mapping)
    parsed.obj = restored
    metrics = autoscore(c.tier, c, parsed)
    row.update(
        latency_s=pre_s + model_s,
        parse_ok=int(parsed.ok),
        output_json=json.dumps(restored, ensure_ascii=False),
        draft=Pseudonymiser.restore(parsed.display, mapping),
        agent_note=Pseudonymiser.restore(parsed.note, mapping),
        metrics_json=json.dumps(metrics) if metrics is not None else None,
    )
    if target.cloud:
        out = "\n".join(sent)
        row.update(pii_annotated=len(c.pii), pii_leaked=sum(leaks(p["text"], out) for p in c.pii))
    return row


def run_variant(
    v: Variant,
    complaints: list[Complaint],
    con: sqlite3.Connection,
    pseudonymiser: Pseudonymiser | None = None,
    limit: int | None = None,
    log: Callable[[str], Any] = print,
    client=None,
    warmup: bool = False,
    variants: dict[str, Variant] | None = None,
    clients: dict[str, Any] | None = None,
    tiers: list[int] | None = None,
) -> dict[str, int]:
    """Answer every complaint with variant `v` and store one row each.

    Resumable: complaints with a successful row are skipped. Rows that ended in
    an error (timeout, rate limit, server down) are retried on the next run.
    `client` (for `v` itself) and `clients` (by variant key) can be injected for
    tests; they need `complete(system, user, tier)` and `chat(system, messages, tier)`.
    `variants` resolves the abstractor and hybrid targets.
    `warmup` sends one untimed, unlogged request to every model involved first,
    so a local server's model load does not land in the first latency. It uses
    a fixed dummy text, never complaint data.
    """
    variants = {**(variants or {}), v.key: v}
    clients = {**(clients or {}), **({v.key: client} if client is not None else {})}
    ctx = Context(variants, pseudonymiser, clients, con)
    involved = [k for k in [v.key, *_sub_variants(v)] if variants[k].route != "hybrid"]
    if any(variants[k].pii in ("pseudonymise", "abstract") for k in involved):
        if not ctx.pseudonymiser().has_ner:
            log(
                f"WARNING [{v.key}]: running WITHOUT NER (Presidio/spaCy missing): "
                "names and addresses will NOT be redacted. Install `.[redaction]`."
            )
    prior = {
        r["complaint_id"]: r["error"]
        for r in con.execute("SELECT complaint_id, error FROM runs WHERE variant=?", (v.key,))
    }
    stats = {"run": 0, "retried": 0, "skipped_done": 0, "blocked": 0, "errors": 0}
    todo = [c for c in complaints if not tiers or c.tier in tiers]
    todo = todo[:limit] if limit else todo
    pending = [c for c in todo if not (c.id in prior and prior[c.id] is None)]
    if warmup and pending:
        for k in dict.fromkeys(involved):
            try:
                ctx.client(k).complete(load_system_prompt(3), WARMUP_TEXT, tier=1)
            except Exception as e:
                log(f"WARNING [{k}]: warm-up failed: {type(e).__name__}: {e}")
    for c in todo:
        if c.id in prior and prior[c.id] is None:
            stats["skipped_done"] += 1
            continue
        base = dict(variant=v.key, complaint_id=c.id, source=c.source, language=c.language,
                    type=c.type, difficulty=c.difficulty,
                    ts=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        try:
            row = {**base, **process(v, c, ctx), "error": None}
        except RealDataBlocked as e:
            log(f"BLOCKED: {e}")
            stats["blocked"] += 1
            continue
        except Exception as e:  # network, timeout, rate limit, empty summary: log and carry on
            row = {**base, "tier": c.tier, "route": v.route, "model_key": v.model_key,
                   "size_class": v.size_class, "reasoning": v.reasoning, "hardware": v.hardware,
                   "parse_ok": 0, "error": f"{type(e).__name__}: {e}"[:500]}
            stats["errors"] += 1
            print(f"ERROR [{v.key}] {c.id}: {row['error']}", file=sys.stderr)
        else:
            stats["run"] += 1
        if c.id in prior:
            stats["retried"] += 1
        cols = ",".join(row)
        con.execute(
            f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({','.join('?' * len(row))})",
            list(row.values()),
        )
        con.commit()
    return stats
