"""Build the identical prompt for every variant and parse the model output.

System prompt = shared base (role, rules, policy) + the task prompt of the
complaint's tier. Within a tier every variant gets exactly the same text.
"""

from __future__ import annotations

import json
import re

from . import ROOT
from .schemas import Complaint
from .tiers import Parsed, parse

PROMPTS = ROOT / "prompts"
TASK_FILES = {
    1: "tier1_classify.md",
    2: "tier2_extract.md",
    3: "tier3_draft.md",
    4: "tier4_draft.md",
    5: "tier5_tools.md",
    6: "tier6_long.md",
}


def load_base_prompt() -> str:
    system = (PROMPTS / "system.md").read_text(encoding="utf-8")
    policy = (PROMPTS / "policy.md").read_text(encoding="utf-8").strip()
    return system.replace("{{policy}}", policy).strip()


def load_system_prompt(tier: int = 3) -> str:
    task = (PROMPTS / "tasks" / TASK_FILES[tier]).read_text(encoding="utf-8").strip()
    return f"{load_base_prompt()}\n\n{task}\n"


def load_abstract_prompt() -> str:
    return (PROMPTS / "abstract.md").read_text(encoding="utf-8").strip() + "\n"


def load_router_prompt() -> str:
    return (PROMPTS / "router.md").read_text(encoding="utf-8").strip() + "\n"


def build_user_message(c: Complaint, text: str | None = None, heading: str | None = None) -> str:
    """Complaint (or a replacement text, e.g. the abstract route's summary), language and facts.

    Tier 6 adds the retrieved policy pages after the thread.
    """
    heading = heading or ("Email thread" if c.tier == 6 else "Complaint")
    facts = json.dumps(c.facts, ensure_ascii=False, indent=2) if c.facts else "{}"
    msg = f"# {heading} (language: {c.language})\n{(text if text is not None else c.text).strip()}\n\n# Customer facts\n{facts}\n"
    if c.tier == 6 and c.context_docs:
        pages = "\n\n".join(f"## Page {i}\n{d.strip()}" for i, d in enumerate(c.context_docs, 1))
        msg += f"\n# Retrieved policy pages\n{pages}\n"
    return msg


def parse_output(raw: str, tier: int = 3) -> Parsed:
    """Parse a model answer against the tier's output contract (see tiers.parse)."""
    return parse(tier, raw)


def normalise_for_blind(text: str) -> str:
    """Strip formatting quirks that would reveal which model wrote a draft."""
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"^\s*[#>]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
