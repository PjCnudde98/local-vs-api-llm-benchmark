"""Generate synthetic complaints for every tier of the ladder with an LLM.

The harness picks the scenario (language, type, difficulty, PII patterns and
the tier's answer key where it can), the model writes the complaint and the
tier-specific data. Output is a draft: a support agent must review it for
realism, and the gold answers for correctness, before it counts. Generated
rows carry `"status": "needs_review"`.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

from . import ROOT
from .config import Variant
from .llm import LLMClient
from .schemas import Complaint
from .tiers import FIELDS, LABELS, validate_gold

GEN_SYSTEM = """You write realistic but entirely fictional customer complaints for testing a support assistant.
Invent all names, addresses, numbers. Never use real people or companies.
Return only JSON with keys:
  "text": the complaint exactly as a customer would write it, in the requested language,
  "facts": object with structured customer facts (order_status, delivery_date, return_window_days, prior_contacts, amount_eur),
  "pii": list of {"type": one of PERSON|LOCATION|EMAIL|PHONE|IBAN|ORDER|NATIONAL_ID, "text": the exact substring in "text"}{extra}
Every "pii" item must appear verbatim in "text"."""

EXTRA = {
    2: ',\n  "fields": {"order_number": ..., "product": ..., "requested_remedy": ..., "deadline": "YYYY-MM-DD" or null}: '
       "the values as planted in the text (order_number and product copied verbatim)",
    5: ',\n  "tool_data": {"orders": {"<order number from the text>": {"order_number", "customer_id" (format CUST-12345), '
       '"status", "ordered_on", "delivered_on", "promised_by", "amount_eur", "items", "photo_received_on", '
       '"charges": [{"date", "amount_eur"}]}}, "contacts": {"<customer_id>": [{"date", "channel", "summary"}]}}: '
       "what the company's systems say; it must make the requested case true",
    6: ',\n  "key_facts": a list of 4-8 short facts a correct reply must get right, including how the contradiction resolves',
}


def _policy_lines() -> list[str]:
    text = (ROOT / "prompts" / "policy.md").read_text(encoding="utf-8")
    return [ln.strip() for ln in text.splitlines() if re.match(r"\s*\d+\.", ln)]


def _scenario(tier: int, sc: dict, rng: random.Random) -> tuple[str, dict, dict]:
    """(prompt for the generator, row fields chosen here, gold chosen here)."""
    langs, weights = zip(*sc["languages"].items())
    t = (sc.get("tiers") or {}).get(tier, {})
    lang = rng.choices(langs, weights)[0]
    ctype = rng.choice(sc["types"])
    diff = rng.choice(t.get("difficulties") or sc["difficulties"])
    pii = rng.sample(sc["pii_patterns"], k=rng.randint(3, 5))
    sens = rng.choice(sc["sensitive_context"])
    lines = [f"Language: {lang}", f"Complaint type: {ctype}", f"Style/difficulty: {diff}",
             f"Include these PII patterns: {', '.join(pii)}", f"Sensitive context: {sens}"]
    row: dict = dict(language=lang, type=ctype, difficulty=diff)
    gold: dict = {}
    if tier == 1:
        gold = dict(type=ctype, language=lang, urgency=rng.choice(t.get("urgency") or LABELS["urgency"]),
                    sentiment=rng.choice(t.get("sentiment") or LABELS["sentiment"]))
        lines += [f"Urgency: {gold['urgency']} (high = legal or press threat, health or safety risk, "
                  "or a deadline within 2 days)", f"Sentiment: {gold['sentiment']}"]
    elif tier == 2:
        remedy = rng.choice(t.get("remedies") or ["refund"])
        deadline = rng.random() < t.get("deadline_share", 0.5)
        lines += [f"The customer asks for: {remedy}", "Mention one order number (format ORD-1234567) and the product.",
                  "Set a concrete deadline date for a response." if deadline else "Do not set any deadline."]
    elif tier == 5:
        case = rng.choice(t["cases"])
        gold = {"decision": case["decision"]}
        row["type"] = None
        lines += [f"Situation: {case['situation']}",
                  f"Under policy rule {case['policy_rule']} the correct remedy is: {case['decision']}. "
                  "The complaint itself must not state the dates or records that decide this: they are only in tool_data.",
                  "Mention exactly one order number (format ORD-1234567)."]
    elif tier == 6:
        lo, hi = t.get("messages") or [8, 14]
        n = rng.randint(lo, hi)
        lines += [f"Write an email thread of {n} messages between the customer and agents, oldest first, "
                  "each starting with a 'From:' and 'Date:' line.",
                  f"Contradiction to include: {rng.choice(t.get('contradictions') or ['two agents disagree'])}"]
        policy = _policy_lines()
        row["context_docs"] = rng.sample(policy, k=min(3, len(policy)))
    return "\n".join(lines), row, gold


def generate(
    scenarios_path: Path, variant: Variant, n: int, out_path: Path, seed: int = 1, start_index: int = 1,
    client=None, log=print, tier: int = 3,
) -> int:
    """Append `n` generated complaints of one tier to `out_path`. Returns how many were written.

    Rows whose gold data fails validation are dropped and logged, so the file
    always loads.
    """
    sc = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    rng = random.Random(seed * 100 + tier)
    client = client or LLMClient(variant)
    system = GEN_SYSTEM.replace("{extra}", EXTRA.get(tier, ""))
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(start_index, start_index + n):
            user, chosen, gold = _scenario(tier, sc, rng)
            try:
                obj = json.loads(client.complete(system, user).text)
                text, pii_list = obj["text"], obj.get("pii", [])
            except Exception as e:
                log(f"generation t{tier}/{i} failed: {type(e).__name__}: {e}")
                continue
            if tier == 2:
                fields = obj.get("fields") or {}
                gold = {"fields": {k: fields.get(k) for k in FIELDS}}
            elif tier == 6:
                gold = {"key_facts": [str(x) for x in obj.get("key_facts") or []]}
            row = dict(
                id=f"syn-t{tier}-{i:04d}", source="synthetic", tier=tier, text=text,
                facts=obj.get("facts", {}), pii=[p for p in pii_list if p.get("text") and p["text"] in text],
                gold=gold, status="needs_review",
                **{k: v for k, v in chosen.items() if v is not None},
            )
            if tier == 5:
                row["tool_data"] = obj.get("tool_data") or {}
            problems = validate_gold(Complaint.from_dict(row))
            if problems:
                log(f"generation t{tier}/{i} dropped: {'; '.join(problems)}")
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
