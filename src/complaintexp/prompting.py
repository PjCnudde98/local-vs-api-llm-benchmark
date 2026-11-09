"""Build the identical prompt for every variant and parse the model output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import ROOT
from .schemas import Complaint


def load_system_prompt() -> str:
    system = (ROOT / "prompts" / "system.md").read_text(encoding="utf-8")
    policy = (ROOT / "prompts" / "policy.md").read_text(encoding="utf-8").strip()
    return system.replace("{{policy}}", policy)


def build_user_message(c: Complaint) -> str:
    facts = json.dumps(c.facts, ensure_ascii=False, indent=2) if c.facts else "{}"
    return f"# Complaint (language: {c.language})\n{c.text.strip()}\n\n# Customer facts\n{facts}\n"


@dataclass
class Parsed:
    draft: str
    note: str
    ok: bool


def parse_output(raw: str) -> Parsed:
    """Strict JSON first, then the first {...} block; otherwise mark malformed."""
    candidates = [raw]
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("draft_reply"), str) and obj["draft_reply"].strip():
            return Parsed(obj["draft_reply"], str(obj.get("agent_note", "")), True)
    return Parsed(raw, "", False)


def normalise_for_blind(text: str) -> str:
    """Strip formatting quirks that would reveal which model wrote a draft."""
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^\s*[#>]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
