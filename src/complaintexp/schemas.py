"""Complaint dataset records (JSONL)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SOURCES = {"synthetic", "real"}
LANGUAGES = {"nl", "fr", "en"}


@dataclass
class Complaint:
    id: str
    source: str  # "synthetic" | "real"
    language: str  # nl | fr | en
    text: str
    type: str = ""  # delivery, damage, billing, refund, staff, repeat
    difficulty: str = ""  # simple, angry, rambling, multi_issue, legal, out_of_policy
    facts: dict = field(default_factory=dict)  # structured customer facts
    # Hand-annotated PII: [{"type": "PERSON", "text": "Jan Peeters"}, ...].
    # Annotated by text (not offsets) so it survives light edits of the complaint.
    pii: list[dict] = field(default_factory=list)
    gold_reply: str | None = None  # experienced agent's reply, ~30 complaints
    sent_reply: str | None = None  # real set: reply the agent actually sent

    @classmethod
    def from_dict(cls, d: dict) -> "Complaint":
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


def load_dataset(*paths: Path) -> list[Complaint]:
    """Load one or more JSONL files. Fails loudly, with file:line, on bad records."""
    items: list[Complaint] = []
    seen: set[str] = set()
    for path in paths:
        for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path}:{n}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON ({where}): {e.msg}") from e
            missing = [k for k in ("id", "source", "language", "text") if not raw.get(k)]
            if missing:
                raise ValueError(f"missing required field(s) {missing} ({where})")
            c = Complaint.from_dict(raw)
            if c.id in seen:
                raise ValueError(f"duplicate complaint id {c.id!r} ({where})")
            if c.source not in SOURCES:
                raise ValueError(f"bad source {c.source!r} for {c.id} ({where})")
            if c.language not in LANGUAGES:
                raise ValueError(f"bad language {c.language!r} for {c.id} ({where})")
            if not isinstance(c.pii, list) or any(
                not isinstance(p, dict) or not p.get("type") or not p.get("text") for p in c.pii
            ):
                raise ValueError(f"pii must be a list of {{type, text}} objects ({where})")
            seen.add(c.id)
            items.append(c)
    return items
