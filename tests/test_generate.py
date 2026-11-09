"""Synthetic complaint generation (generate.py)."""

import json

from complaintexp import ROOT
from complaintexp.generate import generate
from complaintexp.schemas import load_dataset

import pytest

from .conftest import ScriptedClient, make_variant

SCENARIOS = ROOT / "data" / "synthetic" / "scenarios.yaml"


def gen_reply(text="Ik ben Jan Peeters, mail jan@example.com", pii=None):
    pii = pii if pii is not None else [{"type": "PERSON", "text": "Jan Peeters"}, {"type": "EMAIL", "text": "jan@example.com"}]
    return json.dumps({"text": text, "facts": {"order_status": "shipped"}, "pii": pii})


def test_rows_are_valid_dataset_records_marked_for_review(tmp_path):
    out = tmp_path / "g.jsonl"
    assert generate(SCENARIOS, make_variant(), 3, out, client=ScriptedClient([gen_reply()]), log=lambda *_: None) == 3
    cs = load_dataset(out)
    assert [c.id for c in cs] == ["syn-t3-0001", "syn-t3-0002", "syn-t3-0003"] and {c.tier for c in cs} == {3}
    assert all(c.source == "synthetic" and json.loads(l)["status"] == "needs_review"
               for c, l in zip(cs, out.read_text(encoding="utf-8").splitlines()))


def test_pii_not_in_text_is_dropped(tmp_path):
    out = tmp_path / "g.jsonl"
    reply = gen_reply(pii=[{"type": "PERSON", "text": "Jan Peeters"}, {"type": "PERSON", "text": "Ghost"}])
    generate(SCENARIOS, make_variant(), 1, out, client=ScriptedClient([reply]), log=lambda *_: None)
    assert [p["text"] for p in load_dataset(out)[0].pii] == ["Jan Peeters"]


def test_failed_generations_are_skipped_and_logged(tmp_path):
    out, logs = tmp_path / "g.jsonl", []
    n = generate(SCENARIOS, make_variant(), 2, out, client=ScriptedClient(["nope", gen_reply()]), log=logs.append)
    assert n == 1 and "generation t3/1 failed" in logs[0]


def test_same_seed_gives_same_scenarios(tmp_path):
    clients = [ScriptedClient([gen_reply()]) for _ in range(2)]
    for i, c in enumerate(clients):
        generate(SCENARIOS, make_variant(), 4, tmp_path / f"{i}.jsonl", seed=7, client=c, log=lambda *_: None)
    assert [u for _, u in clients[0].calls] == [u for _, u in clients[1].calls]


def test_appending_with_start_index(tmp_path):
    out = tmp_path / "g.jsonl"
    generate(SCENARIOS, make_variant(), 1, out, client=ScriptedClient([gen_reply()]), log=lambda *_: None)
    generate(SCENARIOS, make_variant(), 1, out, start_index=101, client=ScriptedClient([gen_reply()]), log=lambda *_: None)
    assert [c.id for c in load_dataset(out)] == ["syn-t3-0001", "syn-t3-0101"]


def _gen(tmp_path, tier, reply, n=1):
    out, logs = tmp_path / "g.jsonl", []
    client = ScriptedClient([reply])
    written = generate(SCENARIOS, make_variant(), n, out, client=client, log=logs.append, tier=tier)
    return written, (load_dataset(out) if written else []), logs, client


def test_tier1_gold_labels_are_the_scenario_the_harness_picked(tmp_path):
    _, (c,), _, client = _gen(tmp_path, 1, gen_reply())
    prompt = client.calls[0][1]
    assert f"Urgency: {c.gold['urgency']}" in prompt and f"Sentiment: {c.gold['sentiment']}" in prompt
    assert c.gold["type"] == c.type and c.gold["language"] == c.language


def test_tier2_gold_fields_come_from_the_generator(tmp_path):
    reply = json.loads(gen_reply("Order ORD-1234567, a lamp. Refund please."))
    reply["fields"] = {"order_number": "ORD-1234567", "product": "lamp", "requested_remedy": "refund",
                       "deadline": None}
    _, (c,), _, client = _gen(tmp_path, 2, json.dumps(reply))
    assert c.gold["fields"]["order_number"] == "ORD-1234567" and '"fields"' in client.calls[0][0]


def test_tier5_keeps_tool_data_and_the_cases_decision(tmp_path):
    reply = json.loads(gen_reply("Order ORD-1234567 is late."))
    reply["tool_data"] = {"orders": {"ORD-1234567": {"status": "in transit"}}}
    _, (c,), _, client = _gen(tmp_path, 5, json.dumps(reply))
    assert c.tool_data["orders"] and f"correct remedy is: {c.gold['decision']}" in client.calls[0][1]


def test_tier6_adds_policy_pages_and_key_facts(tmp_path):
    reply = json.loads(gen_reply("From: a\nDate: 2026-01-01\nhello"))
    reply["key_facts"] = ["fact one", "fact two"]
    _, (c,), _, _ = _gen(tmp_path, 6, json.dumps(reply))
    assert c.gold["key_facts"] == ["fact one", "fact two"] and c.context_docs


@pytest.mark.parametrize("tier", [2, 5, 6])
def test_rows_with_invalid_gold_are_dropped_not_written(tmp_path, tier):
    n, _, logs, _ = _gen(tmp_path, tier, gen_reply())
    assert n == 0 and "dropped" in logs[0]
