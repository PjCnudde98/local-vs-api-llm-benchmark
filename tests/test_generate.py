"""Synthetic complaint generation (generate.py)."""

import json

from complaintexp import ROOT
from complaintexp.generate import generate
from complaintexp.schemas import load_dataset

from .conftest import ScriptedClient, make_variant

SCENARIOS = ROOT / "data" / "synthetic" / "scenarios.yaml"


def gen_reply(text="Ik ben Jan Peeters, mail jan@example.com", pii=None):
    pii = pii if pii is not None else [{"type": "PERSON", "text": "Jan Peeters"}, {"type": "EMAIL", "text": "jan@example.com"}]
    return json.dumps({"text": text, "facts": {"order_status": "shipped"}, "pii": pii})


def test_rows_are_valid_dataset_records_marked_for_review(tmp_path):
    out = tmp_path / "g.jsonl"
    assert generate(SCENARIOS, make_variant(), 3, out, client=ScriptedClient([gen_reply()]), log=lambda *_: None) == 3
    cs = load_dataset(out)
    assert [c.id for c in cs] == ["syn-0001", "syn-0002", "syn-0003"]
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
    assert n == 1 and "generation 1 failed" in logs[0]


def test_same_seed_gives_same_scenarios(tmp_path):
    clients = [ScriptedClient([gen_reply()]) for _ in range(2)]
    for i, c in enumerate(clients):
        generate(SCENARIOS, make_variant(), 4, tmp_path / f"{i}.jsonl", seed=7, client=c, log=lambda *_: None)
    assert [u for _, u in clients[0].calls] == [u for _, u in clients[1].calls]


def test_appending_with_start_index(tmp_path):
    out = tmp_path / "g.jsonl"
    generate(SCENARIOS, make_variant(), 1, out, client=ScriptedClient([gen_reply()]), log=lambda *_: None)
    generate(SCENARIOS, make_variant(), 1, out, start_index=101, client=ScriptedClient([gen_reply()]), log=lambda *_: None)
    assert [c.id for c in load_dataset(out)] == ["syn-0001", "syn-0101"]
