"""Running variants (runner.py): safeguards, resumability, logging."""

import pytest

from complaintexp.runner import RealDataBlocked, check_allowed, run_variant

from .conftest import ScriptedClient, draft_json, make_variant, real_complaint

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
