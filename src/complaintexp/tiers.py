"""The task complexity ladder (plan: "Task complexity ladder").

Every model runs the same six tiers, from classification to long-context
reasoning. This module holds what is specific to each tier: its gold-label
format, how the model's answer is parsed and how tiers 1, 2 and 5 are scored
automatically. Tiers 3, 4 and 6 are scored by the rubric (LLM judge and blind
human raters). Thresholds live in config/gates.yaml under `tiers`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schemas import Complaint

TIERS = {
    1: "Classify",
    2: "Extract",
    3: "Simple draft",
    4: "Complex draft",
    5: "Policy reasoning with tools",
    6: "Long context",
}
DRAFT_TIERS = (3, 4, 6)  # scored with the rubric

LABELS = {
    "type": ("delivery", "damage", "billing", "refund", "staff", "repeat"),
    "language": ("nl", "fr", "en"),
    "urgency": ("low", "normal", "high"),
    "sentiment": ("positive", "neutral", "negative", "very_negative"),
}
FIELDS = ("order_number", "product", "requested_remedy", "deadline")
REMEDIES = ("refund", "replacement", "redelivery", "voucher", "apology", "information", "other")
DECISIONS = ("refund", "replacement", "redelivery", "voucher", "decline", "escalate")


# -- gold-label checks (used by schemas.load_dataset) ---------------------------
def validate_gold(c: "Complaint") -> list[str]:
    """Problems with the tier-specific gold data of one complaint (empty = fine)."""
    g, p = c.gold or {}, []
    if c.tier == 1:
        for k, allowed in LABELS.items():
            if g.get(k) not in allowed:
                p.append(f"tier 1 needs gold.{k} in {list(allowed)}")
    elif c.tier == 2:
        f = g.get("fields")
        if not isinstance(f, dict) or set(f) != set(FIELDS):
            p.append(f"tier 2 needs gold.fields with exactly {list(FIELDS)} (null when absent)")
        elif f["requested_remedy"] not in REMEDIES:
            p.append(f"gold.fields.requested_remedy must be one of {list(REMEDIES)}")
    elif c.tier == 5:
        if g.get("decision") not in DECISIONS:
            p.append(f"tier 5 needs gold.decision in {list(DECISIONS)}")
        if not isinstance(c.tool_data, dict) or not c.tool_data.get("orders"):
            p.append("tier 5 needs tool_data.orders (what the lookup tools return)")
    elif c.tier == 6:
        kf = g.get("key_facts")
        if not isinstance(kf, list) or not kf or not all(isinstance(x, str) and x for x in kf):
            p.append("tier 6 needs gold.key_facts: a non-empty list of strings")
    return p


# -- parsing -------------------------------------------------------------------
def extract_json(raw: str) -> dict | None:
    """Strict JSON first, then the outermost {...} block. None if neither is an object."""
    candidates = [raw]
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


@dataclass
class Parsed:
    ok: bool
    obj: dict = field(default_factory=dict)
    display: str = ""  # human-readable answer (the draft for draft tiers)
    note: str = ""


def parse(tier: int, raw: str) -> Parsed:
    """Validate the answer against the tier's output contract."""
    obj = extract_json(raw)
    if obj is None:
        return Parsed(False, {}, raw or "", "")
    if tier == 1:
        ok = all(k in obj for k in LABELS)
        return Parsed(ok, obj, ", ".join(f"{k}: {obj.get(k)}" for k in LABELS))
    if tier == 2:
        ok = all(k in obj for k in FIELDS)
        return Parsed(ok, obj, json.dumps({k: obj.get(k) for k in FIELDS}, ensure_ascii=False, indent=1))
    if tier == 5:
        final = obj.get("final") if isinstance(obj.get("final"), dict) else obj
        ok = final.get("decision") in DECISIONS
        return Parsed(ok, final, f"Decision: {final.get('decision')}\n\n{final.get('justification', '')}".strip(),
                      str(final.get("agent_note", "")))
    draft = obj.get("draft_reply")
    ok = isinstance(draft, str) and bool(draft.strip())
    return Parsed(ok, obj, draft if ok else (raw or ""), str(obj.get("agent_note", "")))


# -- automatic scoring -----------------------------------------------------------
def _norm(x: Any) -> str:
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _field_match(name: str, pred: Any, gold: Any) -> bool:
    p, g = _norm(pred), _norm(gold)
    if name == "product":  # free text: either contains the other
        return bool(p and g) and (p in g or g in p)
    return p == g


def autoscore(tier: int, c: "Complaint", parsed: Parsed) -> dict[str, Any] | None:
    """Per-answer metrics for the automatically scored tiers. None for rubric tiers.

    Tier 1: share of the four labels that match gold.
    Tier 2: per-field true/false positives and false negatives (null = not extracted),
            so the report can compute micro-F1 over a set.
    Tier 5: decision correct (tool-call counts are added by the runner).
    A malformed answer scores zero, it is not skipped.
    """
    g = c.gold or {}
    if tier == 1:
        hits = [parsed.ok and _norm(parsed.obj.get(k)) == _norm(g.get(k)) for k in LABELS]
        return {"label_accuracy": sum(hits) / len(LABELS), "labels_correct": sum(hits)}
    if tier == 2:
        gold_fields = g.get("fields") or {}
        tp = fp = fn = 0
        for k in FIELDS:
            gv = gold_fields.get(k)
            pv = parsed.obj.get(k) if parsed.ok else None
            pv = None if pv in ("", "null", "none") else pv
            if pv is not None and gv is not None and _field_match(k, pv, gv):
                tp += 1
            else:
                fp += pv is not None
                fn += gv is not None
        return {"tp": tp, "fp": fp, "fn": fn, "valid_json": int(parsed.ok)}
    if tier == 5:
        return {"decision_correct": int(parsed.ok and parsed.obj.get("decision") == g.get("decision"))}
    return None
