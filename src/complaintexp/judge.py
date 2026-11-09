"""LLM judge using the same rubric as the human reviewers (draft tiers 3, 4 and 6).

For tier 6 the judge also checks the draft against the whole thread: is it
factually consistent with it, and which of the gold key facts does it miss.
"""

from __future__ import annotations

import json
import sqlite3

from .config import Variant, load_rubric, load_variants
from .llm import LLMClient
from .prompting import build_user_message, load_base_prompt, normalise_for_blind
from .schemas import Complaint
from .tiers import DRAFT_TIERS, extract_json

JUDGE_SYSTEM = """You are a strict reviewer of customer-care reply drafts.
Score the draft on each criterion from {lo} to {hi} (integers) using the policy and facts given.
Criteria:
{criteria}
Also set "hallucination" to 1 if the draft invents facts or promises something the policy does not allow, else 0.
Return only JSON: {{"scores": {{<criterion>: <int>, ...}}, "hallucination": 0 or 1}}.
You do not know which system wrote the draft."""

LONG_CONTEXT_ADDENDUM = """
This complaint is a long email thread. Also check the draft against the numbered key facts:
- "consistent": 1 if nothing in the draft contradicts the thread or what the company promised in it, else 0
- "missed_facts": the numbers of the key facts the draft gets wrong or ignores although the reply needs them
Return only JSON: {"scores": {...}, "hallucination": 0 or 1, "consistent": 0 or 1, "missed_facts": [<int>, ...]}."""


def parse_judgement(raw: str, criteria: list[str], lo: float, hi: float) -> tuple[dict[str, float], int]:
    """Validate a judge reply. Raises ValueError on anything off-rubric."""
    out = extract_json(raw)
    if out is None:
        raise ValueError("judge reply is not a JSON object")
    scores = {k: float(out["scores"][k]) for k in criteria}
    for k, s in scores.items():
        if not lo <= s <= hi:
            raise ValueError(f"score {k}={s} outside {lo}-{hi}")
    hall = out.get("hallucination", 0)
    if hall not in (0, 1, True, False, "0", "1"):
        raise ValueError(f"hallucination must be 0 or 1, got {hall!r}")
    return scores, int(hall)


def parse_long_context(raw: str, n_facts: int) -> dict:
    """Tier 6 extras: consistency flag and missed key facts (1-based numbers)."""
    out = extract_json(raw) or {}
    cons = out.get("consistent")
    if cons not in (0, 1, True, False, "0", "1"):
        raise ValueError(f"consistent must be 0 or 1, got {cons!r}")
    missed = out.get("missed_facts", [])
    if not isinstance(missed, list) or any(not str(m).isdigit() or not 1 <= int(m) <= n_facts for m in missed):
        raise ValueError(f"missed_facts must list key-fact numbers 1-{n_facts}, got {missed!r}")
    return {"consistent": int(cons), "missed_facts": sorted({int(m) for m in missed}), "n_key_facts": n_facts}


def judge_variant(
    variant_key: str,
    complaints: list[Complaint],
    con: sqlite3.Connection,
    log=print,
    variants: dict[str, Variant] | None = None,
    client=None,
    judge_key: str = "JUDGE",
) -> int:
    """Score every parsed draft of one variant. Resumable per (variant, complaint, judge model).

    `variants` and `client` can be injected for tests.
    """
    variants = variants or load_variants()
    judge, target = variants[judge_key], variants[variant_key]
    if judge.model == target.model or judge.model in _models_behind(target, variants):
        raise ValueError(
            f"judge model {judge.model!r} equals variant {variant_key}'s model: "
            "a variant must never judge itself"
        )
    rubric = load_rubric()
    lo, hi = rubric["scale"]
    criteria = list(rubric["criteria"])
    system = JUDGE_SYSTEM.format(
        lo=lo, hi=hi,
        criteria="\n".join(f"- {k}: {v}" for k, v in rubric["criteria"].items()),
    )
    client = client or LLMClient(judge)
    policy = load_base_prompt()
    by_id = {c.id: c for c in complaints}
    done = {
        r[0] for r in con.execute(
            "SELECT complaint_id FROM judgements WHERE variant=? AND judge_model=?",
            (variant_key, judge.model),
        )
    }
    n = 0
    tiers = ",".join(str(t) for t in DRAFT_TIERS)
    for r in con.execute(
        f"SELECT complaint_id, draft FROM runs WHERE variant=? AND parse_ok=1 "
        f"AND (tier IS NULL OR tier IN ({tiers}))", (variant_key,)
    ).fetchall():
        c = by_id.get(r["complaint_id"])
        if c is None or c.id in done:
            continue
        if c.source == "real" and (judge.cloud or not judge.allow_real_data):
            continue  # real complaints never go to a cloud judge
        key_facts = (c.gold or {}).get("key_facts") if c.tier == 6 else None
        user = (
            f"# Instructions given to the drafting assistant\n{policy}\n\n"
            f"{build_user_message(c)}\n# Draft reply\n{normalise_for_blind(r['draft'])}\n"
        )
        if key_facts:
            user += "\n# Key facts\n" + "\n".join(f"{i}. {f}" for i, f in enumerate(key_facts, 1)) + "\n"
        try:
            raw = client.complete(system + (LONG_CONTEXT_ADDENDUM if key_facts else ""), user).text
            scores, hall = parse_judgement(raw, criteria, lo, hi)
            extra = parse_long_context(raw, len(key_facts)) if key_facts else None
        except Exception as e:
            log(f"judge failed for {variant_key}/{c.id}: {type(e).__name__}: {e}")
            continue
        con.execute(
            "INSERT OR REPLACE INTO judgements (variant, complaint_id, judge_model, scores_json, "
            "hallucination, extra_json) VALUES (?,?,?,?,?,?)",
            (variant_key, c.id, judge.model, json.dumps(scores), hall, json.dumps(extra) if extra else None),
        )
        con.commit()
        n += 1
    return n


def _models_behind(v: Variant, variants: dict[str, Variant]) -> set[str]:
    """Models that can write a hybrid variant's drafts."""
    if v.route != "hybrid":
        return set()
    return {variants[k].model for k in (v.hybrid.get("easy"), v.hybrid.get("hard")) if k in variants}
