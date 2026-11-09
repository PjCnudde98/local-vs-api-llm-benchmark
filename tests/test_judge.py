"""LLM judge (judge.py): independence, validation, real-data rule."""

import json

import pytest

from complaintexp.judge import judge_variant, parse_judgement

from .conftest import ScriptedClient, insert_run, make_variant, real_complaint

CRIT = ["factual_correctness", "policy_compliance", "tone_empathy", "language_quality", "completeness"]


def verdict(score=4, hallucination=0, **override):
    return json.dumps({"scores": {c: override.get(c, score) for c in CRIT}, "hallucination": hallucination})


def variants(judge_model="judge-1", target_model="model-C", judge_cloud=True):
    return {
        "C": make_variant("C", model=target_model),
        "JUDGE": make_variant("JUDGE", model=judge_model, cloud=judge_cloud),
    }


quiet = dict(log=lambda *_: None)


def test_variant_never_judges_itself(con, sample):
    with pytest.raises(ValueError, match="never judge itself"):
        judge_variant("C", sample, con, variants=variants(judge_model="same", target_model="same"),
                      client=ScriptedClient(), **quiet)


def test_scores_are_stored_per_judge_model(con, sample):
    insert_run(con, "C", sample[0].id)
    n = judge_variant("C", sample, con, variants=variants(), client=ScriptedClient([verdict(5, 1)]), **quiet)
    r = con.execute("SELECT * FROM judgements").fetchone()
    assert n == 1 and r["judge_model"] == "judge-1" and r["hallucination"] == 1
    assert json.loads(r["scores_json"]) == {c: 5.0 for c in CRIT}


def test_judging_is_resumable(con, sample):
    insert_run(con, "C", sample[0].id)
    judge_variant("C", sample, con, variants=variants(), client=ScriptedClient([verdict()]), **quiet)
    client = ScriptedClient([verdict()])
    assert judge_variant("C", sample, con, variants=variants(), client=client, **quiet) == 0
    assert client.calls == []


def test_only_parsed_drafts_are_judged(con, sample):
    insert_run(con, "C", sample[0].id, parse_ok=0)
    assert judge_variant("C", sample, con, variants=variants(), client=ScriptedClient([verdict()]), **quiet) == 0


def test_real_complaints_never_go_to_a_cloud_judge(con):
    r = real_complaint()
    insert_run(con, "C", r.id, source="real")
    client = ScriptedClient([verdict()])
    assert judge_variant("C", [r], con, variants=variants(judge_cloud=True), client=client, **quiet) == 0
    assert client.calls == []


def test_judge_prompt_hides_the_variant(con, sample):
    insert_run(con, "C", sample[0].id)
    client = ScriptedClient([verdict()])
    judge_variant("C", sample, con, variants=variants(), client=client, **quiet)
    system, user = client.calls[0]
    assert "model-C" not in user and "model-C" not in system
    assert sample[0].text in user


@pytest.mark.parametrize(
    "raw",
    ["no json", verdict(6), verdict(0), verdict(hallucination="yes"),
     json.dumps({"scores": {"factual_correctness": 4}, "hallucination": 0})],
)
def test_bad_verdicts_are_skipped_and_logged(con, sample, raw):
    insert_run(con, "C", sample[0].id)
    logs = []
    n = judge_variant("C", sample, con, variants=variants(), client=ScriptedClient([raw]), log=logs.append)
    assert n == 0 and logs and "judge failed" in logs[0]
    assert con.execute("SELECT COUNT(*) FROM judgements").fetchone()[0] == 0


def test_parse_judgement_accepts_bool_hallucination():
    _, h = parse_judgement(json.dumps({"scores": {c: 3 for c in CRIT}, "hallucination": True}), CRIT, 1, 5)
    assert h == 1


def test_parse_judgement_ignores_extra_criteria():
    scores, _ = parse_judgement(json.dumps({"scores": {**{c: 3 for c in CRIT}, "style": 9}}), CRIT, 1, 5)
    assert set(scores) == set(CRIT)
