"""Shared fixtures.

Design rules for this suite:
- No test may reach the network. `_no_network` blocks every socket connect, so
  a missing fake shows up as a failing test, not as a surprise Azure bill or a
  complaint leaving the machine.
- Model calls go through `ScriptedClient`, a hand-rolled fake with the same
  `complete(system, user, tier)` and `chat(system, messages, tier)` contract as `LLMClient`.
- Every test gets its own SQLite file in tmp_path; the real results DB is never touched.
"""

from __future__ import annotations

import json
import socket
from collections.abc import Iterable

import pytest

from complaintexp import ROOT
from complaintexp.config import Variant
from complaintexp.llm import Completion
from complaintexp.pseudonymise import Pseudonymiser
from complaintexp.schemas import Complaint, load_dataset
from complaintexp.store import connect

SAMPLE = ROOT / "data" / "synthetic" / "sample.jsonl"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(self, address, *a, **k):
        if self.family == socket.AF_UNIX:
            return _orig(self, address, *a, **k)
        raise RuntimeError(f"network access attempted in a test: {address!r}")

    _orig = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture(scope="session")
def ps() -> Pseudonymiser:
    """Regex-only pseudonymiser: fast and identical on every machine."""
    return Pseudonymiser(use_ner=False)


@pytest.fixture(scope="session")
def ner_ps() -> Pseudonymiser:
    p = Pseudonymiser(use_ner=True)
    if not p.has_ner:
        pytest.skip("Presidio/spaCy models not installed (uv sync --extra redaction)")
    return p


@pytest.fixture
def sample() -> list[Complaint]:
    return load_dataset(SAMPLE)


@pytest.fixture
def con(tmp_path):
    c = connect(tmp_path / "test.sqlite")
    yield c
    c.close()


def make_variant(key="T", **kw) -> Variant:
    defaults = dict(label=key.lower(), provider="fake", model=f"model-{key}", cloud=False,
                    pii="none", allow_real_data=False)
    return Variant(key=key, **{**defaults, **kw})


def real_complaint(cid="r1", text="Mijn pakket is niet geleverd. Jan Peeters, jan@example.com", tier=3) -> Complaint:
    return Complaint(id=cid, source="real", language="nl", text=text, tier=tier)


def by_tier(complaints, tier):
    return [c for c in complaints if c.tier == tier]


def draft_json(text="Dear customer, we are sorry.", note="note") -> str:
    return json.dumps({"draft_reply": text, "agent_note": note})


class ScriptedClient:
    """Fake LLM client. Each response is a string (returned) or an Exception (raised).

    The last response repeats once the script runs out. Every call is recorded
    in `.calls` as (system, last user message) and in `.conversations` as
    (system, messages) so tests can inspect what would have been sent.
    """

    def __init__(self, responses: Iterable[str | Exception] = (draft_json(),), latency_s: float = 1.0,
                 model_id: str = "scripted-1"):
        self.responses = list(responses)
        self.latency_s = latency_s
        self.model_id = model_id
        self.calls: list[tuple[str, str]] = []
        self.conversations: list[tuple[str, list[dict]]] = []
        self.tiers: list[int | None] = []

    def complete(self, system: str, user: str, tier: int | None = None) -> Completion:
        return self.chat(system, [{"role": "user", "content": user}], tier)

    def chat(self, system: str, messages: list[dict], tier: int | None = None) -> Completion:
        self.calls.append((system, messages[-1]["content"]))
        self.conversations.append((system, [dict(m) for m in messages]))
        self.tiers.append(tier)
        r = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(r, Exception):
            raise r
        return Completion(r, self.latency_s, 100, 50, self.model_id)


def insert_run(con, variant, cid, *, latency=1.0, error=None, parse_ok=1, language="en",
               type_="delivery", difficulty="simple", source="synthetic", pt=100, ct=50, tier=3,
               route=None, model_key=None, size_class=None, metrics=None, tool_calls=None,
               valid_tool_calls=None, routed_to=None, hardware=None, reasoning=None):
    route = route or (variant.split("/")[0] if "/" in variant else "local")
    con.execute(
        "INSERT INTO runs (variant, complaint_id, source, language, type, difficulty, tier, route, model_key, "
        "size_class, reasoning, hardware, routed_to, ts, model_id, draft, parse_ok, metrics_json, tool_calls, "
        "valid_tool_calls, latency_s, prompt_tokens, completion_tokens, error) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (variant, cid, source, language, type_, difficulty, tier, route,
         model_key or (variant.split("/")[-1].removesuffix("+think")), size_class, reasoning, hardware, routed_to,
         "2026-11-06T00:00:00+00:00", f"m-{variant}", None if error else "draft", parse_ok,
         json.dumps(metrics) if metrics is not None else None, tool_calls, valid_tool_calls,
         None if error else latency, pt, ct, error),
    )
    con.commit()


def insert_score(con, table, variant, cid, score, hallucination=0, who="r", edit=None, extra=None,
                 policy=None):
    crit = ["factual_correctness", "policy_compliance", "tone_empathy", "language_quality", "completeness"]
    vals = {c: score for c in crit}
    if policy is not None:
        vals["policy_compliance"] = policy
    scores = json.dumps(vals)
    if table == "judgements":
        con.execute("INSERT INTO judgements (variant, complaint_id, judge_model, scores_json, hallucination, "
                    "extra_json) VALUES (?,?,?,?,?,?)",
                    (variant, cid, who, scores, hallucination, json.dumps(extra) if extra else None))
    else:
        con.execute("INSERT INTO human_scores VALUES (?,?,?,?,?,?)", (variant, cid, who, scores, hallucination, edit))
    con.commit()
