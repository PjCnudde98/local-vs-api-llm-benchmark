"""Generate synthetic complaints with an LLM from the scenario axes.

Output is a draft: a support agent must review it for realism before it counts
as part of the test set. Generated rows carry `"status": "needs_review"`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from .llm import LLMClient
from .config import Variant

GEN_SYSTEM = """You write realistic but entirely fictional customer complaints for testing a support assistant.
Invent all names, addresses, numbers. Never use real people or companies.
Return only JSON with keys:
  "text": the complaint exactly as a customer would write it, in the requested language,
  "facts": object with structured customer facts (order_status, delivery_date, return_window_days, prior_contacts, amount_eur),
  "pii": list of {"type": one of PERSON|LOCATION|EMAIL|PHONE|IBAN|ORDER|NATIONAL_ID, "text": the exact substring in "text"}.
Every "pii" item must appear verbatim in "text"."""


def generate(
    scenarios_path: Path, variant: Variant, n: int, out_path: Path, seed: int = 1, start_index: int = 1,
    client=None, log=print,
) -> int:
    sc = yaml.safe_load(scenarios_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    langs, weights = zip(*sc["languages"].items())
    client = client or LLMClient(variant)
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(start_index, start_index + n):
            lang = rng.choices(langs, weights)[0]
            ctype = rng.choice(sc["types"])
            diff = rng.choice(sc["difficulties"])
            pii = rng.sample(sc["pii_patterns"], k=rng.randint(3, 5))
            sens = rng.choice(sc["sensitive_context"])
            user = (
                f"Language: {lang}\nComplaint type: {ctype}\nStyle/difficulty: {diff}\n"
                f"Include these PII patterns: {', '.join(pii)}\nSensitive context: {sens}"
            )
            try:
                obj = json.loads(client.complete(GEN_SYSTEM, user).text)
                text, pii_list = obj["text"], obj.get("pii", [])
            except Exception as e:
                log(f"generation {i} failed: {type(e).__name__}: {e}")
                continue
            pii_list = [p for p in pii_list if p.get("text") and p["text"] in text]
            row = dict(
                id=f"syn-{i:04d}", source="synthetic", language=lang, type=ctype, difficulty=diff,
                text=text, facts=obj.get("facts", {}), pii=pii_list, status="needs_review",
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written
