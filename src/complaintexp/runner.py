"""Run every complaint through every variant and log the result."""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timezone

from .config import Variant
from .llm import LLMClient
from .prompting import build_user_message, load_system_prompt, parse_output
from .pseudonymise import Pseudonymiser
from .schemas import Complaint


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


def run_variant(
    v: Variant,
    complaints: list[Complaint],
    con: sqlite3.Connection,
    pseudonymiser: Pseudonymiser | None = None,
    limit: int | None = None,
    log=print,
    client=None,
    warmup: bool = False,
) -> dict[str, int]:
    """Draft a reply for every complaint with variant `v` and store one row each.

    Resumable: complaints with a successful row are skipped. Rows that ended in
    an error (timeout, rate limit, server down) are retried on the next run.
    `client` can be injected for tests; it needs a `complete(system, user)` method.
    `warmup` sends one untimed, unlogged request first so a local server's model
    load does not land in the first complaint's latency. It uses a fixed dummy
    text, never complaint data.
    """
    client = client or LLMClient(v)
    system = load_system_prompt()
    if v.pii == "pseudonymise":
        pseudonymiser = pseudonymiser or Pseudonymiser()
        if not pseudonymiser.has_ner:
            log(
                f"WARNING [{v.key}]: running WITHOUT NER (Presidio/spaCy missing): "
                "names and addresses will NOT be redacted. Install `.[redaction]`."
            )
    prior = {
        r["complaint_id"]: r["error"]
        for r in con.execute("SELECT complaint_id, error FROM runs WHERE variant=?", (v.key,))
    }
    stats = {"run": 0, "retried": 0, "skipped_done": 0, "blocked": 0, "errors": 0}
    todo = complaints[:limit] if limit else list(complaints)
    pending = [c for c in todo if not (c.id in prior and prior[c.id] is None)]
    if warmup and pending:
        try:
            client.complete(system, "# Complaint (language: en)\nWarm-up request, please ignore.\n")
        except Exception as e:
            log(f"WARNING [{v.key}]: warm-up failed: {type(e).__name__}: {e}")
    for c in todo:
        if c.id in prior and prior[c.id] is None:
            stats["skipped_done"] += 1
            continue
        try:
            check_allowed(v, c)
        except RealDataBlocked as e:
            log(f"BLOCKED: {e}")
            stats["blocked"] += 1
            continue
        if c.id in prior:
            stats["retried"] += 1

        user = build_user_message(c)
        mapping: dict[str, str] = {}
        if v.pii == "pseudonymise":
            user, mapping = pseudonymiser.redact(user, c.language)

        row = dict(
            variant=v.key, complaint_id=c.id, source=c.source, language=c.language,
            type=c.type, difficulty=c.difficulty,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            model_id=None, prompt_sent=user, output_raw=None, draft=None, agent_note=None,
            parse_ok=0, latency_s=None, prompt_tokens=None, completion_tokens=None, error=None,
        )
        try:
            comp = client.complete(system, user)
            parsed = parse_output(comp.text)
            row.update(
                model_id=comp.model_id, output_raw=comp.text,
                draft=Pseudonymiser.restore(parsed.draft, mapping),
                agent_note=Pseudonymiser.restore(parsed.note, mapping),
                parse_ok=int(parsed.ok), latency_s=comp.latency_s,
                prompt_tokens=comp.prompt_tokens, completion_tokens=comp.completion_tokens,
            )
            stats["run"] += 1
        except Exception as e:  # network, timeout, rate limit: log and carry on
            row["error"] = f"{type(e).__name__}: {e}"[:500]
            stats["errors"] += 1
            print(f"ERROR [{v.key}] {c.id}: {row['error']}", file=sys.stderr)
        cols = ",".join(row)
        con.execute(
            f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({','.join('?' * len(row))})",
            list(row.values()),
        )
        con.commit()
    return stats
