"""Running variants (runner.py): safeguards, resumability, logging."""

import pytest

from complaintexp.runner import RealDataBlocked, check_allowed, run_variant

import json

from .conftest import ScriptedClient, by_tier, draft_json, make_variant, real_complaint

quiet = dict(log=lambda *_: None)


# -- real-data safeguard ------------------------------------------------------
@pytest.mark.parametrize(
    "cloud, allow, blocked",
    [(True, True, True), (True, False, True), (False, False, True), (False, True, False)],
)
def test_real_data_only_reaches_cleared_local_variants(cloud, allow, blocked):
    v = make_variant(cloud=cloud, allow_real_data=allow)
    if blocked:
        with pytest.raises(RealDataBlocked):
            check_allowed(v, real_complaint())
    else:
        check_allowed(v, real_complaint())


def test_synthetic_data_is_allowed_everywhere(sample):
    check_allowed(make_variant(cloud=True, allow_real_data=False), sample[0])


def test_blocked_real_complaint_is_never_sent(con):
    client = ScriptedClient()
    stats = run_variant(make_variant(cloud=True), [real_complaint()], con, client=client, **quiet)
    assert stats["blocked"] == 1 and client.calls == []
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


# -- resumability ------------------------------------------------------------
def test_run_is_resumable(con, sample, ps):
    v = make_variant()
    s1 = run_variant(v, sample, con, ps, client=ScriptedClient(), **quiet)
    s2 = run_variant(v, sample, con, ps, client=ScriptedClient(), **quiet)
    assert s1["run"] == len(sample) and s2["run"] == 0 and s2["skipped_done"] == len(sample)


def test_errored_runs_are_retried_on_next_run(con, sample):
    v = make_variant()
    run_variant(v, sample[:2], con, client=ScriptedClient([TimeoutError("slow")]), **quiet)
    s = run_variant(v, sample[:2], con, client=ScriptedClient(), **quiet)
    assert s["retried"] == 2 and s["run"] == 2
    assert con.execute("SELECT COUNT(*) FROM runs WHERE error IS NULL").fetchone()[0] == 2


def test_parse_failures_are_kept_not_retried(con, sample):
    """A malformed answer is a result (it counts as a failure), not a transient error."""
    v = make_variant()
    run_variant(v, sample[:1], con, client=ScriptedClient(["not json"]), **quiet)
    s = run_variant(v, sample[:1], con, client=ScriptedClient(), **quiet)
    assert s["skipped_done"] == 1
    assert con.execute("SELECT parse_ok FROM runs").fetchone()[0] == 0


def test_limit_caps_the_number_of_complaints(con, sample):
    s = run_variant(make_variant(), sample, con, client=ScriptedClient(), limit=2, **quiet)
    assert s["run"] == 2


def test_variants_are_stored_independently(con, sample):
    run_variant(make_variant("C"), sample[:2], con, client=ScriptedClient(), **quiet)
    run_variant(make_variant("D"), sample[:2], con, client=ScriptedClient(), **quiet)
    assert con.execute("SELECT COUNT(DISTINCT variant) FROM runs").fetchone()[0] == 2


# -- what gets logged ----------------------------------------------------------
def test_error_is_logged_and_run_continues(con, sample):
    client = ScriptedClient([ConnectionError("down"), draft_json()])
    s = run_variant(make_variant(), sample[:3], con, client=client, **quiet)
    assert s["errors"] == 1 and s["run"] == 2
    err = con.execute("SELECT error FROM runs WHERE error IS NOT NULL").fetchone()[0]
    assert err == "ConnectionError: down"


def test_successful_row_has_metrics(con, sample):
    """Latency is preprocessing plus model time; a local variant has no preprocessing."""
    run_variant(make_variant(), sample[:1], con, client=ScriptedClient(latency_s=2.5), **quiet)
    r = con.execute("SELECT * FROM runs").fetchone()
    assert (r["latency_s"], r["prompt_tokens"], r["completion_tokens"], r["parse_ok"]) == (2.5, 100, 50, 1)
    assert r["model_id"] == "scripted-1" and r["language"] == sample[0].language


def test_pseudonymised_prompt_is_what_gets_sent_and_logged(con, sample, ps):
    client = ScriptedClient()
    run_variant(make_variant("B", cloud=True, pii="pseudonymise"), sample[:1], con, ps, client=client, **quiet)
    sent = con.execute("SELECT prompt_sent FROM runs").fetchone()[0]
    assert sent == client.calls[0][1]
    assert "jan.peeters@example.com" not in sent and "ORD-4821937" not in sent and "<EMAIL_1>" in sent


def test_placeholders_in_reply_are_restored_in_stored_draft(con, sample, ps):
    client = ScriptedClient([draft_json("We will mail <EMAIL_1> about <ORDER_1>.", "check <ORDER_1>")])
    run_variant(make_variant("B", cloud=True, pii="pseudonymise"), sample[:1], con, ps, client=client, **quiet)
    r = con.execute("SELECT draft, agent_note, output_raw FROM runs").fetchone()
    assert r["draft"] == "We will mail jan.peeters@example.com about ORD-4821937."
    assert r["agent_note"] == "check ORD-4821937"
    assert "<EMAIL_1>" in r["output_raw"]  # raw output kept exactly as received


def test_raw_variant_sends_text_unchanged(con, sample):
    client = ScriptedClient()
    run_variant(make_variant("A", cloud=True, pii="raw"), sample[:1], con, client=client, **quiet)
    assert "jan.peeters@example.com" in client.calls[0][1]


def test_all_variants_get_the_identical_system_prompt(con, sample, ps):
    clients = {k: ScriptedClient() for k in "ABCD"}
    for k, c in clients.items():
        run_variant(make_variant(k, pii="pseudonymise" if k == "B" else "none"), sample[:1], con, ps, client=c, **quiet)
    assert len({c.calls[0][0] for c in clients.values()}) == 1


def test_each_tier_gets_its_own_task_prompt_and_token_budget(con, sample):
    client = ScriptedClient()
    run_variant(make_variant(), sample, con, client=client, **quiet)
    systems = {t: s for (s, _), t in zip(client.calls, client.tiers) if t is not None}
    assert set(systems) == {1, 2, 3, 4, 5, 6}
    assert all(f"# Task (tier {t})" in s for t, s in systems.items())


def test_missing_ner_warning_for_pseudonymising_variant(con, sample, ps):
    logs = []
    run_variant(make_variant("B", pii="pseudonymise"), sample[:1], con, ps, client=ScriptedClient(), log=logs.append)
    assert any("WITHOUT NER" in m for m in logs)


# -- warm-up ------------------------------------------------------------------
def test_warmup_sends_one_dummy_request_without_complaint_data(con, sample):
    client = ScriptedClient()
    run_variant(make_variant(), sample[:2], con, client=client, warmup=True, **quiet)
    assert len(client.calls) == 3 and "Warm-up" in client.calls[0][1]
    assert sample[0].text not in client.calls[0][1]
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2


def test_warmup_skipped_when_nothing_to_do(con, sample):
    run_variant(make_variant(), sample[:1], con, client=ScriptedClient(), **quiet)
    client = ScriptedClient()
    run_variant(make_variant(), sample[:1], con, client=client, warmup=True, **quiet)
    assert client.calls == []


def test_failed_warmup_does_not_stop_the_run(con, sample):
    client = ScriptedClient([RuntimeError("cold"), draft_json()])
    s = run_variant(make_variant(), sample[:1], con, client=client, warmup=True, **quiet)
    assert s["run"] == 1 and s["errors"] == 0


# -- automatic scoring ---------------------------------------------------------------
def test_tier1_answer_is_scored_against_gold(con, sample):
    c = by_tier(sample, 1)[0]
    answer = json.dumps({**c.gold, "urgency": "low" if c.gold["urgency"] != "low" else "high"})
    run_variant(make_variant(), [c], con, client=ScriptedClient([answer]), **quiet)
    m = json.loads(con.execute("SELECT metrics_json FROM runs").fetchone()[0])
    assert m == {"label_accuracy": 0.75, "labels_correct": 3}


def test_tier2_fields_are_compared_after_normalisation(con, sample):
    c = next(x for x in by_tier(sample, 2) if x.gold["fields"]["deadline"])
    f = c.gold["fields"]
    answer = json.dumps({**f, "order_number": f["order_number"].lower(), "product": "Barista Pro"})
    run_variant(make_variant(), [c], con, client=ScriptedClient([answer]), **quiet)
    m = json.loads(con.execute("SELECT metrics_json FROM runs").fetchone()[0])
    assert m == {"tp": 4, "fp": 0, "fn": 0, "valid_json": 1}


def test_malformed_answer_scores_zero_not_skipped(con, sample):
    run_variant(make_variant(), by_tier(sample, 1)[:1], con, client=ScriptedClient(["no"]), **quiet)
    r = con.execute("SELECT parse_ok, metrics_json FROM runs").fetchone()
    assert r["parse_ok"] == 0 and json.loads(r["metrics_json"])["label_accuracy"] == 0


# -- tier 5: tool loop ------------------------------------------------------------------
def _tool(name, **args):
    return json.dumps({"tool": name, "args": args})


def _final(decision):
    return json.dumps({"final": {"decision": decision, "justification": "j", "agent_note": "n"}})


def test_tool_loop_looks_up_data_and_scores_the_decision(con, sample):
    c = next(x for x in by_tier(sample, 5) if x.gold["decision"] == "refund")
    order = next(iter(c.tool_data["orders"]))
    client = ScriptedClient([_tool("get_order", order_number=order),
                             _tool("get_contact_history", customer_id="CUST-48213"), _final("refund")])
    run_variant(make_variant(), [c], con, client=client, **quiet)
    r = con.execute("SELECT * FROM runs").fetchone()
    assert (r["tool_calls"], r["valid_tool_calls"], r["parse_ok"]) == (2, 2, 1)
    assert json.loads(r["metrics_json"]) == {"decision_correct": 1}
    second_turn = client.conversations[1][1]
    assert '"tool_result"' in second_turn[-1]["content"] and '"in transit"' in second_turn[-1]["content"]
    assert r["prompt_tokens"] == 300 and r["latency_s"] == 3.0  # summed over the three turns


def test_invalid_tool_calls_are_counted_and_answered_with_an_error(con, sample):
    c = by_tier(sample, 5)[0]
    client = ScriptedClient([_tool("delete_order", order_number="x"), json.dumps({"tool": "get_order", "args": {}}),
                             _final("escalate")])
    run_variant(make_variant(), [c], con, client=client, **quiet)
    r = con.execute("SELECT tool_calls, valid_tool_calls FROM runs").fetchone()
    assert (r[0], r[1]) == (2, 0) and "invalid tool call" in client.conversations[1][1][-1]["content"]


def test_model_that_never_decides_is_a_parse_failure(con, sample):
    run_variant(make_variant(), by_tier(sample, 5)[:1], con, client=ScriptedClient(["thinking..."]), **quiet)
    r = con.execute("SELECT parse_ok, metrics_json FROM runs").fetchone()
    assert r["parse_ok"] == 0 and json.loads(r["metrics_json"]) == {"decision_correct": 0}


def test_pseudonymised_tool_loop_restores_args_and_redacts_results(con, sample, ps):
    c = next(x for x in by_tier(sample, 5) if x.gold["decision"] == "refund")
    client = ScriptedClient([_tool("get_order", order_number="<ORDER_1>"),
                             _tool("get_contact_history", customer_id="<CUSTOMER_ID_1>"), _final("refund")])
    run_variant(make_variant("P", cloud=True, pii="pseudonymise", route="pseudonymise"), [c], con, ps,
                client=client, **quiet)
    sent = " ".join(m["content"] for m in client.conversations[-1][1] if m["role"] == "user")
    assert "ORD-2208813" not in sent and "CUST-48213" not in sent and "<CUSTOMER_ID_1>" in sent
    assert '"in transit"' in sent  # the lookup found the order behind <ORDER_1>
    r = con.execute("SELECT valid_tool_calls, metrics_json FROM runs").fetchone()
    assert r[0] == 2 and json.loads(r[1]) == {"decision_correct": 1}


# -- routes ---------------------------------------------------------------------------
def test_abstract_route_sends_only_the_summary(con, sample, ps):
    c = sample[0]
    vs = {"L": make_variant("L", allow_real_data=True)}
    v = make_variant("AB", cloud=True, pii="abstract", route="abstract", abstractor="L")
    abstractor = ScriptedClient([json.dumps({"summary": "Late parcel <ORDER_1>, customer angry, wants delivery."})])
    api = ScriptedClient()
    run_variant(v, [c], con, ps, client=api, variants=vs, clients={"L": abstractor}, **quiet)
    assert "ORD-4821937" not in abstractor.calls[0][1]          # the abstractor reads pseudonymised text
    sent = api.calls[0][1]
    assert sent.startswith("# Case summary") and c.text not in sent and "<ORDER_1>" in sent
    r = con.execute("SELECT preprocess_output, pii_leaked, pii_annotated, latency_s FROM runs").fetchone()
    assert r["preprocess_output"].startswith("Late parcel") and r["pii_leaked"] == 0 and r["pii_annotated"] == 5
    assert r["latency_s"] >= 2.0  # abstractor + API call


def test_abstract_route_without_summary_is_an_error(con, sample, ps):
    vs = {"L": make_variant("L")}
    v = make_variant("AB", cloud=True, pii="abstract", route="abstract", abstractor="L")
    s = run_variant(v, sample[:1], con, ps, client=ScriptedClient(), variants=vs,
                    clients={"L": ScriptedClient(["{}"])}, **quiet)
    assert s["errors"] == 1 and "no summary" in con.execute("SELECT error FROM runs").fetchone()[0]


def test_leakage_is_counted_for_cloud_routes(con, sample, ps):
    run_variant(make_variant("P", cloud=True, pii="pseudonymise", route="pseudonymise"), sample[:1], con, ps,
                client=ScriptedClient(), **quiet)
    r = con.execute("SELECT pii_leaked, pii_annotated FROM runs").fetchone()
    assert r["pii_annotated"] == 5 and r["pii_leaked"] == 1  # regex-only: the name stays in


def _hybrid(**kw):
    vs = {"R": make_variant("R"), "E": make_variant("E", allow_real_data=True),
          "X": make_variant("X", cloud=True, pii="raw", route="raw")}
    h = make_variant("H", route="hybrid", provider="hybrid",
                     hybrid={"router": "R", "easy": "E", "hard": "X", "hard_tiers": [5, 6], "router_tiers": [3, 4]},
                     **kw)
    return h, vs


def test_hybrid_routes_by_tier_and_router(con, sample):
    h, vs = _hybrid()
    clients = {"R": ScriptedClient([json.dumps({"complexity": "hard"})]), "E": ScriptedClient(), "X": ScriptedClient()}
    run_variant(h, sample, con, variants=vs, clients=clients, **quiet)
    routed = {r["complaint_id"]: r["routed_to"] for r in con.execute("SELECT complaint_id, routed_to FROM runs")}
    tier = {c.id: c.tier for c in sample}
    assert all(routed[i] == "X" for i in routed if tier[i] in (3, 4, 5, 6))   # router said hard
    assert all(routed[i] == "E" for i in routed if tier[i] in (1, 2))         # not a router tier
    assert len(clients["R"].calls) == sum(t in (3, 4) for t in tier.values())


def test_hybrid_never_sends_real_data_to_its_cloud_target(con):
    h, vs = _hybrid()
    clients = {"R": ScriptedClient([json.dumps({"complexity": "hard"})]), "E": ScriptedClient(), "X": ScriptedClient()}
    s = run_variant(h, [real_complaint(tier=4)], con, variants=vs, clients=clients, **quiet)
    assert s["blocked"] == 1 and clients["X"].calls == []


def test_warmup_reaches_every_model_a_variant_uses(con, sample):
    h, vs = _hybrid()
    clients = {k: ScriptedClient([json.dumps({"complexity": "easy"}), draft_json()]) for k in "REX"}
    run_variant(h, sample[:1], con, variants=vs, clients=clients, warmup=True, **quiet)
    assert all("Warm-up" in c.calls[0][1] for c in clients.values())


def test_abstract_summary_is_shared_by_every_api_model(con, sample, ps):
    vs = {"L": make_variant("L")}
    abstractor = ScriptedClient([json.dumps({"summary": "Late parcel <ORDER_1>."})], latency_s=4.0)
    for key in ("AB1", "AB2"):
        v = make_variant(key, cloud=True, pii="abstract", route="abstract", abstractor="L")
        run_variant(v, sample[:1], con, ps, client=ScriptedClient(), variants=vs, clients={"L": abstractor}, **quiet)
    assert len(abstractor.calls) == 1
    lat = [r[0] for r in con.execute("SELECT preprocess_latency_s FROM runs ORDER BY variant")]
    assert all(x >= 4.0 for x in lat)  # the cached abstractor time still counts for each variant
