"""LLM judge using the same rubric as the human reviewers."""

from __future__ import annotations

import json
import sqlite3

from .config import Variant, load_rubric, load_variants
from .llm import LLMClient
from .prompting import load_system_prompt, normalise_for_blind
from .schemas import Complaint

JUDGE_SYSTEM = """You are a strict reviewer of customer-care reply drafts.
Score the draft on each criterion from {lo} to {hi} (integers) using the policy and facts given.
Criteria:
{criteria}
Also set "hallucination" to 1 if the draft invents facts or promises something the policy does not allow, else 0.
Return only JSON: {{"scores": {{<criterion>: <int>, ...}}, "hallucination": 0 or 1}}.
You do not know which system wrote the draft."""


def parse_judgement(raw: str, criteria: list[str], lo: float, hi: float) -> tuple[dict[str, float], int]:
    """Validate a judge reply. Raises ValueError on anything off-rubric."""
    out = json.loads(raw)
    scores = {k: float(out["scores"][k]) for k in criteria}
    for k, s in scores.items():
        if not lo <= s <= hi:
            raise ValueError(f"score {k}={s} outside {lo}-{hi}")
    hall = out.get("hallucination", 0)
    if hall not in (0, 1, True, False, "0", "1"):
        raise ValueError(f"hallucination must be 0 or 1, got {hall!r}")
    return scores, int(hall)


def judge_variant(
    variant_key: str,
    complaints: list[Complaint],
    con: sqlite3.Connection,
    log=print,
    variants: dict[str, Variant] | None = None,
    client=None,
) -> int:
    """Score every parsed draft of one variant. Resumable per (variant, complaint, judge model).

    `variants` and `client` can be injected for tests.
    """
    variants = variants or load_variants()
    judge, target = variants["JUDGE"], variants[variant_key]
    if judge.model == target.model:
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
    policy = load_system_prompt()
    by_id = {c.id: c for c in complaints}
    done = {
        r[0] for r in con.execute(
            "SELECT complaint_id FROM judgements WHERE variant=? AND judge_model=?",
            (variant_key, judge.model),
        )
    }
    n = 0
    for r in con.execute(
        "SELECT complaint_id, draft FROM runs WHERE variant=? AND parse_ok=1", (variant_key,)
    ).fetchall():
        c = by_id.get(r["complaint_id"])
        if c is None or c.id in done:
            continue
        if c.source == "real" and (judge.cloud or not judge.allow_real_data):
            continue  # real complaints never go to a cloud judge
        user = (
            f"# Instructions given to the drafting assistant\n{policy}\n\n"
            f"# Complaint ({c.language})\n{c.text}\n\n# Customer facts\n"
            f"{json.dumps(c.facts, ensure_ascii=False)}\n\n"
            f"# Draft reply\n{normalise_for_blind(r['draft'])}\n"
        )
        try:
            scores, hall = parse_judgement(client.complete(system, user).text, criteria, lo, hi)
        except Exception as e:
            log(f"judge failed for {variant_key}/{c.id}: {type(e).__name__}: {e}")
            continue
        con.execute(
            "INSERT OR REPLACE INTO judgements VALUES (?,?,?,?,?)",
            (variant_key, c.id, judge.model, json.dumps(scores), hall),
        )
        con.commit()
        n += 1
    return n
