"""Prompt construction, output parsing and blind normalisation (prompting.py)."""

import json

import pytest

from complaintexp.prompting import build_user_message, load_system_prompt, normalise_for_blind, parse_output
from complaintexp.prompting import TASK_FILES
from complaintexp.schemas import Complaint


def test_system_prompt_has_policy_inlined():
    s = load_system_prompt()
    assert "{{policy}}" not in s
    assert "# Complaint policy" in s and "draft_reply" in s


def test_user_message_contains_language_text_and_facts():
    c = Complaint(id="x", source="synthetic", language="fr", text="  Bonjour  ", facts={"amount_eur": 42.0}, tier=3)
    msg = build_user_message(c)
    assert "language: fr" in msg and "Bonjour" in msg and '"amount_eur": 42.0' in msg


def test_user_message_with_no_facts_is_stable():
    c = Complaint(id="x", source="synthetic", language="en", text="t")
    assert build_user_message(c).endswith("# Customer facts\n{}\n")


def test_user_message_keeps_non_ascii():
    c = Complaint(id="x", source="synthetic", language="fr", text="t", facts={"store": "Liège"})
    assert "Liège" in build_user_message(c)


GOOD = json.dumps({"draft_reply": "Hi", "agent_note": "n"})


@pytest.mark.parametrize(
    "raw",
    [GOOD, f"Sure!\n```json\n{GOOD}\n```", f"<think>hmm</think>{GOOD}", f"  {GOOD}  \n"],
)
def test_parse_output_accepts_valid_or_wrapped_json(raw):
    p = parse_output(raw)
    assert p.ok and p.display == "Hi" and p.note == "n"


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        json.dumps({"draft_reply": "  "}),
        json.dumps({"reply": "Hi"}),
        json.dumps({"draft_reply": 3}),
        json.dumps(["draft_reply", "Hi"]),
        "",
        '{"draft_reply": "Hi"',
    ],
)
def test_parse_output_rejects_malformed(raw):
    p = parse_output(raw)
    assert not p.ok and p.display == raw


def test_parse_output_missing_note_is_empty_string():
    p = parse_output(json.dumps({"draft_reply": "Hi"}))
    assert p.ok and p.note == ""


def test_blind_normalisation_strips_markdown():
    assert normalise_for_blind("**Dear** Jan,\n\n\n\n`x`") == "Dear Jan,\n\nx"


def test_blind_normalisation_strips_headings_quotes_and_trailing_space():
    assert normalise_for_blind("# Reply  \n> quoted\n__bold__") == "Reply\nquoted\nbold"


def test_blind_normalisation_is_idempotent():
    t = "**A**\n\n\n# B\n> C  \n"
    assert normalise_for_blind(normalise_for_blind(t)) == normalise_for_blind(t)


@pytest.mark.parametrize("tier", sorted(TASK_FILES))
def test_every_tier_shares_the_base_prompt_and_names_its_task(tier):
    s = load_system_prompt(tier)
    assert s.startswith(load_system_prompt(1).split("# Task")[0]) and f"# Task (tier {tier})" in s


def test_tier6_message_carries_thread_and_policy_pages():
    c = Complaint(id="x", source="synthetic", language="en", text="From: a\nhi", tier=6, context_docs=["3. Returns"])
    msg = build_user_message(c)
    assert msg.startswith("# Email thread (language: en)") and "## Page 1\n3. Returns" in msg


def test_replacement_text_and_heading_for_the_abstract_route():
    c = Complaint(id="x", source="synthetic", language="nl", text="secret", tier=3)
    msg = build_user_message(c, text="summary", heading="Case summary")
    assert msg.startswith("# Case summary (language: nl)\nsummary") and "secret" not in msg


@pytest.mark.parametrize("tier, raw, ok", [
    (1, '{"type": "delivery", "language": "en", "urgency": "low", "sentiment": "neutral"}', True),
    (1, '{"type": "delivery"}', False),
    (2, '{"order_number": null, "product": "lamp", "requested_remedy": "refund", "deadline": null}', True),
    (2, '{"order_number": "ORD-1"}', False),
    (5, '{"final": {"decision": "refund", "justification": "late"}}', True),
    (5, '{"final": {"decision": "money back"}}', False),
])
def test_parse_checks_each_tiers_output_contract(tier, raw, ok):
    assert parse_output(raw, tier).ok is ok
