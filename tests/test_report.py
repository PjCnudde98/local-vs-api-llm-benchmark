"""Aggregation, cost model, gates and the Markdown view (report.py)."""

import pytest

from complaintexp.config import load_gates
from complaintexp.report import (_break_even, _pct, bootstrap_ci, build, costs_table, effort, evaluate_gates, fixed_per_month,
                                 frontier, judge_agreement, operations, quality, render_markdown,
                                 splits, volume_view)
from complaintexp.schemas import Complaint

from .conftest import insert_run, insert_score

COSTS = {
    "volumes_per_month": [1000, 50000],
    "agent_eur_per_hour": 60,
    "usd_to_eur": 0.5,
    "max_utilisation": 1.0,
    "models": {"haiku": {"input_per_1m": 2.0, "output_per_1m": 8.0, "currency": "EUR"},
               "sonnet": {"input_per_1m": 4.0, "output_per_1m": 16.0, "currency": "USD"},
               "gpt": {"input_per_1m": None, "output_per_1m": None}},
    "hardware": {"box": {"hardware_eur": 3600, "amortisation_months": 36, "hosting_eur_per_month": 50,
                         "ops_hours_per_month": 4, "eur_per_ops_hour": 80},
                 "empty": {"hardware_eur": None, "amortisation_months": 36, "hosting_eur_per_month": 50,
                           "ops_hours_per_month": 4, "eur_per_ops_hour": 80},
                 "vm": {"vm_eur_per_hour": 1.0, "hours_per_month": 730}},
}
BOX_FIXED = 3600 / 36 + 50 + 4 * 80


# -- statistics ---------------------------------------------------------------
@pytest.mark.parametrize(
    "values, q, expected",
    [([], 0.5, None), ([7], 0.95, 7), ([1, 2, 3, 4], 0.5, 2), (list(range(1, 21)), 0.95, 19),
     (list(range(1, 101)), 0.95, 95), ([5, 1, 3], 0.0, 1)],
)
def test_nearest_rank_percentile(values, q, expected):
    assert _pct(values, q) == expected


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    xs = [3, 4, 4, 5, 2, 4, 5, 3]
    lo, hi = bootstrap_ci(xs)
    assert bootstrap_ci(xs) == (lo, hi) and lo <= sum(xs) / len(xs) <= hi


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_ci([4.0]) is None


def test_bootstrap_ci_of_constant_is_a_point():
    assert bootstrap_ci([4.0] * 10) == (4.0, 4.0)


# -- operations ---------------------------------------------------------------
def test_operations_counts_errors_and_parse_failures_separately(con):
    for i, lat in enumerate([1, 2, 3, 4]):
        insert_run(con, "C", f"c{i}", latency=lat)
    insert_run(con, "C", "e1", error="Timeout")
    insert_run(con, "C", "p1", parse_ok=0, latency=9)
    (o,) = operations(con)
    assert (o["n"], o["errors"], o["parse_failures"], o["n_ok"]) == (6, 1, 1, 4)
    assert o["failure_rate"] == pytest.approx(2 / 6)
    assert o["p50_s"] == 3 and o["max_s"] == 9  # parse failures still have a latency
    assert o["model_ids"] == ["m-C"]


def test_operations_empty_db(con):
    assert operations(con) == []


# -- cost ---------------------------------------------------------------------
def _op_row(v, route, model_key=None, hardware=None, pt=100, ct=50, n=10):
    return {"variant": v, "route": route, "model_key": model_key, "hardware": hardware, "n": n, "errors": 0,
            "mean_prompt_tokens": pt, "mean_completion_tokens": ct, "tokens_by_tier": {}}


def _costs(*ops, qual=(), variants=None, tp=None):
    return {r["variant"]: r for r in costs_table(list(ops), list(qual), COSTS, variants or {}, tp or {})}


def test_local_fixed_cost_is_amortised_hardware_hosting_and_ops():
    assert fixed_per_month(COSTS, "box") == pytest.approx(BOX_FIXED)
    c = _costs(_op_row("local/m", "local", hardware="box"))["local/m"]
    assert c["eur_per_1000@1000"] == pytest.approx(BOX_FIXED)
    assert c["eur_per_1000@50000"] == pytest.approx(BOX_FIXED / 50)


def test_token_cost_in_eur_and_converted_from_usd():
    rows = _costs(_op_row("pseudonymise/haiku", "pseudonymise", "haiku"),
                  _op_row("pseudonymise/sonnet", "pseudonymise", "sonnet"))
    assert rows["pseudonymise/haiku"]["eur_per_reply_variable"] == pytest.approx((100 * 2 + 50 * 8) / 1e6)
    assert rows["pseudonymise/sonnet"]["eur_per_reply_variable"] == pytest.approx((100 * 4 + 50 * 16) / 1e6 * 0.5)
    assert rows["pseudonymise/haiku"]["eur_per_1000@1000"] == rows["pseudonymise/haiku"]["eur_per_1000@50000"]


def test_vm_cost_is_fixed_per_month():
    assert fixed_per_month(COSTS, "vm") == 730


def test_missing_inputs_give_none_not_zero():
    rows = _costs(_op_row("raw/gpt", "raw", "gpt"), _op_row("local/x", "local", hardware="empty"))
    assert all(r["eur_per_1000@1000"] is None for r in rows.values())
    assert any("empty inputs" in n for n in rows["local/x"]["notes"])


def test_sendable_reply_adds_agent_edit_time():
    q = [{"source": "human", "variant": "local/m", "edit_minutes_median": 2.0}]
    c = _costs(_op_row("local/m", "local", hardware="box"), qual=q)["local/m"]
    assert c["eur_per_sendable@1000"] == pytest.approx(BOX_FIXED / 1000 + 2.0 * 60 / 60)


def test_capacity_cliff_adds_servers_from_measured_throughput():
    tp = {"local/m": {"replies_per_hour": 10, "concurrency": 1, "p95_s": 5}}  # 7,300 replies a month per box
    c = _costs(_op_row("local/m", "local", hardware="box"), tp=tp)["local/m"]
    assert c["servers@1000"] == 1 and c["servers@50000"] == 7
    assert c["eur_per_1000@50000"] == pytest.approx(BOX_FIXED * 7 / 50)


def test_unmeasured_throughput_assumes_one_server_and_says_so():
    c = _costs(_op_row("local/m", "local", hardware="box"))["local/m"]
    assert c["servers@50000"] == 1 and any("throughput not measured" in n for n in c["notes"])


def test_break_even_against_cheapest_api_route():
    rows = _costs(_op_row("pseudonymise/haiku", "pseudonymise", "haiku"),
                  _op_row("pseudonymise/sonnet", "pseudonymise", "sonnet"),
                  _op_row("local/m", "local", hardware="box"))
    cheapest = min(rows["pseudonymise/haiku"]["eur_per_reply_variable"],
                   rows["pseudonymise/sonnet"]["eur_per_reply_variable"])
    assert rows["local/m"]["break_even_replies_per_month"] == pytest.approx(BOX_FIXED / cheapest)
    assert "break_even_replies_per_month" not in rows["pseudonymise/haiku"]


def test_abstract_route_adds_the_abstractors_box():
    from .conftest import make_variant

    vs = {"abstract/haiku": make_variant("abstract/haiku", route="abstract", abstractor="local/m"),
          "local/m": make_variant("local/m", hardware="box")}
    c = _costs(_op_row("abstract/haiku", "abstract", "haiku"), variants=vs)["abstract/haiku"]
    assert c["fixed_eur_per_month"] == pytest.approx(BOX_FIXED)
    assert c["eur_per_1000@1000"] == pytest.approx(((100 * 2 + 50 * 8) / 1e6 + BOX_FIXED / 1000) * 1000)


# -- quality and splits ---------------------------------------------------------
def test_quality_per_source_and_variant(con):
    for i, s in enumerate([4, 5, 3]):
        insert_run(con, "C", f"c{i}")
        insert_score(con, "judgements", "C", f"c{i}", s, hallucination=int(i == 0))
    insert_score(con, "human_scores", "C", "c0", 2, edit=3.0)
    q = {(r["source"], r["variant"]): r for r in quality(con)}
    j = q[("llm_judge", "C")]
    assert j["n"] == 3 and j["overall"] == pytest.approx(4.0)
    assert j["hallucination_rate"] == pytest.approx(1 / 3) and j["ci95"] is not None
    assert q[("human", "C")]["edit_minutes_median"] == 3.0


def test_splits_by_language(con):
    insert_run(con, "C", "n1", language="nl")
    insert_run(con, "C", "f1", language="fr")
    insert_score(con, "judgements", "C", "n1", 5)
    insert_score(con, "judgements", "C", "f1", 3)
    got = {s["value"]: s["result"] for s in splits(con)
           if s["metric"] == "overall_llm_judge" and s["dimension"] == "language"}
    assert got == {"nl": 5, "fr": 3}


# -- gates ----------------------------------------------------------------------
G = load_gates()


def _op(v, p95, n=40, route="local"):
    return {"variant": v, "p95_s": p95, "n": n, "n_latency": n, "route": route}


def _q(v, overall, src="llm_judge", hall=0.0, n=40, edit=None):
    return {"source": src, "variant": v, "overall": overall, "hallucination_rate": hall, "n": n,
            "edit_minutes_median": edit, "n_edit": n if edit is not None else 0}


def _status(gates, name, variant):
    return next(g for g in gates if g["gate"].startswith(name) and g["variant"] == variant)


def test_gate_thresholds_are_inclusive_or_strict_as_documented():
    gates = evaluate_gates([_op("C", 20.0), _op("D", 19.99)], [_q("C", 4.0, hall=0.02), _q("D", 3.99, hall=0.0199)],
                           None, G)
    assert _status(gates, "p95", "C")["status"] == "fail"          # < 20 s: 20.0 fails
    assert _status(gates, "p95", "D")["status"] == "pass"
    assert _status(gates, "Rubric", "C")["status"] == "pass"       # >= 4.0: 4.0 passes
    assert _status(gates, "Rubric", "D")["status"] == "fail"
    assert _status(gates, "Hallucination", "C")["status"] == "fail"  # < 2%
    assert _status(gates, "Hallucination", "D")["status"] == "pass"


def test_human_scores_are_preferred_over_judge():
    gates = evaluate_gates([_op("C", 1)], [_q("C", 3.0, "human"), _q("C", 4.5, "llm_judge")], None, G)
    g = _status(gates, "Rubric", "C")
    assert g["source"] == "human" and g["status"] == "fail"


def test_missing_data_is_not_measured_not_failed():
    gates = evaluate_gates([_op("C", 1)], [], None, G)
    assert _status(gates, "Rubric", "C")["status"] == "not_measured"
    assert _status(gates, "Redaction recall", "pseudonymiser")["status"] == "not_measured"
    assert _status(gates, "Pairwise", "C")["status"] == "not_measured"


def test_dry_run_variants_are_not_gated():
    gates = evaluate_gates([_op("FAKE", 99)], [], None, G)
    assert not any(g["variant"] == "FAKE" for g in gates)


def test_privacy_quality_loss_is_raw_minus_each_preprocessing_route():
    ops = [_op("raw/haiku", 1, route="raw"), _op("pseudonymise/haiku", 1, route="pseudonymise"),
           _op("abstract/haiku", 1, route="abstract")]
    gates = evaluate_gates(ops, [_q("raw/haiku", 4.5), _q("pseudonymise/haiku", 4.2), _q("abstract/haiku", 4.4)],
                           None, G)
    g = _status(gates, "Quality loss from pseudonymise", "pseudonymise/haiku")
    assert g["value"] == pytest.approx(0.3) and g["status"] == "fail"
    assert _status(gates, "Quality loss from abstract", "abstract/haiku")["status"] == "pass"


def test_latency_gate_uses_draft_tiers_only(con):
    insert_run(con, "local/m", "t3", latency=5, tier=3)
    insert_run(con, "local/m", "t5", latency=90, tier=5)  # tool loop: several calls
    (o,) = operations(con)
    assert o["p95_s"] == 5 and o["p95_all_s"] == 90 and o["n_latency"] == 1


def test_redaction_recall_gate_per_direct_identifier_type():
    red = {"has_ner": False, "types": [{"type": "PERSON", "recall": 0.5, "n": 2},
                                       {"type": "EMAIL", "recall": 1.0, "n": 4}]}
    gates = evaluate_gates([], [], red, G)
    assert _status(gates, "Redaction recall: PERSON", "pseudonymiser")["status"] == "fail"
    assert _status(gates, "Redaction recall: EMAIL", "pseudonymiser")["status"] == "pass"
    assert "NER was off" in _status(gates, "Redaction recall: EMAIL", "pseudonymiser")["note"]
    assert _status(gates, "Redaction recall: IBAN", "pseudonymiser")["status"] == "not_measured"


def test_small_samples_are_flagged():
    gates = evaluate_gates([_op("C", 1, n=6)], [_q("C", 4.5, n=6)], None, G)
    assert "small sample" in _status(gates, "Rubric", "C")["note"]
    assert "small sample" in _status(gates, "p95", "C")["note"]


# -- effort ---------------------------------------------------------------------
def test_effort_sums_hours(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("date,person,phase,task_category,variant,hours,notes\n"
                 "2026-11-01,anna,setup,infra,C,2.5,\n2026-11-02,ben,scoring,review,,1.5,\n"
                 "2026-11-02,ben,scoring,review,,oops,\n", encoding="utf-8")
    e = effort(p)
    assert e["total_hours"] == 4.0 and e["entries"] == 2
    assert e["by_phase"] == {"setup": 2.5, "scoring": 1.5} and e["by_variant"] == {"C": 2.5, "shared": 1.5}


def test_effort_without_log(tmp_path):
    assert effort(tmp_path / "missing.csv")["total_hours"] == 0.0


# -- build + markdown ---------------------------------------------------------------
def _seed(con):
    insert_run(con, "C", "s1")
    insert_run(con, "C", "r1", source="real")
    con.execute("UPDATE runs SET draft='secret draft' WHERE complaint_id='r1'")
    con.commit()
    insert_score(con, "judgements", "C", "s1", 4)
    return [Complaint(id="s1", source="synthetic", language="en", text="synthetic text", tier=3),
            Complaint(id="r1", source="real", language="en", text="REAL CUSTOMER TEXT", tier=3)]


def test_build_hides_real_text_by_default(con, tmp_path):
    cs = _seed(con)
    d = build(con, costs=COSTS, complaints=cs, hours_csv=None)
    assert d["complaints"]["s1"]["text"] == "synthetic text"
    assert d["complaints"]["r1"]["text"] is None and d["complaints"]["r1"]["redacted"]
    assert next(r for r in d["records"] if r["complaint_id"] == "r1")["draft"] is None


def test_build_includes_real_text_only_when_asked(con):
    d = build(con, costs=COSTS, complaints=_seed(con), hours_csv=None, include_real_text=True)
    assert d["complaints"]["r1"]["text"] == "REAL CUSTOMER TEXT" and d["meta"]["include_real_text"]


def test_build_without_text(con):
    d = build(con, costs=COSTS, complaints=_seed(con), hours_csv=None, include_text=False)
    assert all(c["text"] is None for c in d["complaints"].values())
    assert all(r["draft"] is None for r in d["records"])


def test_records_carry_judge_scores(con):
    d = build(con, costs=COSTS, complaints=_seed(con), hours_csv=None)
    r = next(r for r in d["records"] if r["complaint_id"] == "s1")
    assert r["judge"]["scores"]["tone_empathy"] == 4 and r["human"] is None


def test_markdown_has_every_section(con):
    md = render_markdown(build(con, costs=COSTS, complaints=_seed(con), hours_csv=None))
    for h in ("## Capability frontier", "## Gates", "## Operations", "## Cost (EUR)", "## Quality", "## Privacy",
              "## Effort", "### Quality by language", "### Quality by tier"):
        assert h in md, h


def test_markdown_on_empty_db(con):
    md = render_markdown(build(con, costs=COSTS, hours_csv=None))
    assert "No scores yet" in md and "No redaction eval stored" in md


# -- display names --------------------------------------------------------------------
from complaintexp.report import display_names  # noqa: E402

from .conftest import make_variant  # noqa: E402


def _ops_with(**models):
    return [{"variant": k, "model_ids": [m] if m else []} for k, m in models.items()]


def test_unknown_variants_are_named_after_the_reported_model():
    names = display_names(["C", "D"], _ops_with(C="mistral-small3.2", D="qwen3:8b"), {})
    assert names == {"C": "mistral-small3.2", "D": "qwen3:8b"}


def test_names_show_model_label_route_and_thinking():
    vs = {"pseudonymise/h": make_variant("pseudonymise/h", model_label="Claude Haiku 5.5", route="pseudonymise"),
          "raw/h": make_variant("raw/h", model_label="Claude Haiku 5.5", route="raw"),
          "local/q+think": make_variant("local/q+think", model_label="Qwen3 8B", route="local", reasoning="on")}
    assert display_names(list(vs), [], vs) == {"pseudonymise/h": "Claude Haiku 5.5 (pseudonymised)",
                                               "raw/h": "Claude Haiku 5.5 (raw text)",
                                               "local/q+think": "Qwen3 8B (local, thinking)"}


def test_unset_model_env_var_is_never_shown():
    vs = {"raw/gpt": make_variant("raw/gpt", model="${AZURE_DEPLOYMENT_GPT}", label="GPT / Azure", route="")}
    assert display_names(["raw/gpt"], [], vs) == {"raw/gpt": "GPT / Azure"}


def test_names_are_always_unique():
    vs = {k: make_variant(k, model="same", label="same") for k in "XY"}
    names = display_names(["X", "Y"], [], vs)
    assert len(set(names.values())) == 2


def test_markdown_shows_model_names_not_keys(con):
    insert_run(con, "C", "s1")
    insert_score(con, "judgements", "C", "s1", 4)
    md = render_markdown(build(con, costs=COSTS, hours_csv=None, variants={}))
    assert "| m-C |" in md and "| C |" not in md


# -- capability frontier ----------------------------------------------------------------
R = load_gates()
RUBRIC = {"scale": [1, 5], "criteria": {}}


def _cell(fr, v, tier):
    return next(c for c in fr["cells"] if c["variant"] == v and c["tier"] == tier)


def test_tier1_label_accuracy_and_parse_failures_count_as_zero(con):
    insert_run(con, "local/m", "a", tier=1, metrics={"label_accuracy": 1.0})
    insert_run(con, "local/m", "b", tier=1, metrics={"label_accuracy": 0.0}, parse_ok=0)
    c = _cell(frontier(con, R, RUBRIC), "local/m", 1)
    assert c["metrics"][0]["value"] == 0.5 and c["status"] == "fail"


def test_tier2_micro_f1_and_valid_json_rate(con):
    insert_run(con, "local/m", "a", tier=2, metrics={"tp": 3, "fp": 0, "fn": 1, "valid_json": 1})
    insert_run(con, "local/m", "b", tier=2, metrics={"tp": 4, "fp": 0, "fn": 0, "valid_json": 1})
    m = {x["metric"]: x for x in _cell(frontier(con, R, RUBRIC), "local/m", 2)["metrics"]}
    assert m["field_f1"]["value"] == pytest.approx(2 * 7 / (2 * 7 + 1))
    assert m["valid_json_rate"]["value"] == 1.0 and m["valid_json_rate"]["status"] == "pass"


def test_tier5_decision_accuracy_and_valid_tool_calls(con):
    insert_run(con, "local/m", "a", tier=5, metrics={"decision_correct": 1}, tool_calls=3, valid_tool_calls=3)
    insert_run(con, "local/m", "b", tier=5, metrics={"decision_correct": 0}, tool_calls=2, valid_tool_calls=1)
    m = {x["metric"]: x["value"] for x in _cell(frontier(con, R, RUBRIC), "local/m", 5)["metrics"]}
    assert m == {"decision_accuracy": 0.5, "valid_tool_call_rate": pytest.approx(4 / 5)}


def test_draft_tier_rubric_counts_malformed_drafts_as_lowest_score(con):
    insert_run(con, "local/m", "a", tier=3)
    insert_score(con, "judgements", "local/m", "a", 5)
    insert_run(con, "local/m", "b", tier=3, parse_ok=0)
    c = _cell(frontier(con, R, RUBRIC), "local/m", 3)
    assert c["metrics"][0]["value"] == 3.0 and c["score_source"] == "llm_judge"


def test_tier4_policy_compliance_rate(con):
    for i, pc in enumerate([5, 4, 3, 5]):
        insert_run(con, "local/m", f"c{i}", tier=4)
        insert_score(con, "judgements", "local/m", f"c{i}", 5, policy=pc)
    m = {x["metric"]: x["value"] for x in _cell(frontier(con, R, RUBRIC), "local/m", 4)["metrics"]}
    assert m["policy_compliance_rate"] == 0.75


def test_tier6_consistency_and_missed_facts_from_judge(con):
    insert_run(con, "local/m", "a", tier=6)
    insert_score(con, "judgements", "local/m", "a", 4,
                 extra={"consistent": 1, "missed_facts": [2], "n_key_facts": 5})
    insert_run(con, "local/m", "b", tier=6)
    insert_score(con, "judgements", "local/m", "b", 4,
                 extra={"consistent": 0, "missed_facts": [], "n_key_facts": 5})
    m = {x["metric"]: x["value"] for x in _cell(frontier(con, R, RUBRIC), "local/m", 6)["metrics"]}
    assert m == {"factual_consistency_rate": 0.5, "missed_fact_rate": pytest.approx(1 / 10)}


def test_failure_point_is_the_first_failing_tier(con):
    insert_run(con, "local/m", "a", tier=1, metrics={"label_accuracy": 1.0})
    insert_run(con, "local/m", "b", tier=2, metrics={"tp": 4, "fp": 0, "fn": 0, "valid_json": 1})
    insert_run(con, "local/m", "c", tier=5, metrics={"decision_correct": 0}, tool_calls=1, valid_tool_calls=1)
    fr = frontier(con, R, RUBRIC)
    s = next(x for x in fr["summary"] if x["variant"] == "local/m")
    assert s["breaks_at"] == 5 and s["passed"] == [1, 2]
    assert _cell(fr, "local/m", 3)["status"] == "not_measured"
    assert "small sample" in _cell(fr, "local/m", 1)["note"]


def test_economic_failure_when_local_costs_more_than_cheapest_passing_api(con):
    for v, route, mk, hw in (("local/m", "local", "m", "box"), ("pseudonymise/haiku", "pseudonymise", "haiku", None)):
        insert_run(con, v, "a", tier=1, metrics={"label_accuracy": 1.0}, route=route, model_key=mk, hardware=hw)
    ops = operations(con)
    rows = costs_table(ops, [], COSTS, {}, {})
    fr = frontier(con, R, RUBRIC, rows, ops, COSTS, {})
    econ = _cell(fr, "local/m", 1)["economic"]
    assert econ[1000]["status"] == "fail"       # ~470 EUR/month spread over 1,000 replies
    assert econ[1000]["api_variant"] == "pseudonymise/haiku"
    assert "economic" not in _cell(fr, "pseudonymise/haiku", 1)


# -- judge agreement ------------------------------------------------------------
def _two_judges(con):
    """Opus gives Claude drafts 5 and the others 3; GPT gives everything 3."""
    for v, opus in (("raw/claude-x", 5), ("local/open-y", 3)):
        for i in range(3):
            insert_run(con, v, f"c{i}")
            insert_score(con, "judgements", v, f"c{i}", opus, who="opus", hallucination=0)
            insert_score(con, "judgements", v, f"c{i}", 3, who="gpt", hallucination=i % 2)


def test_judge_agreement_compares_judges_on_the_same_drafts(con):
    _two_judges(con)
    (j,) = judge_agreement(con)
    assert (j["a"], j["b"], j["n"]) == ("gpt", "opus", 6)
    assert j["mean_diff"] == pytest.approx(-1.0) and j["mean_abs_diff"] == pytest.approx(1.0)
    assert j["hallucination_agreement"] == pytest.approx(4 / 6)


def test_judge_agreement_splits_the_gap_by_the_drafting_models_family(con):
    # Opus is 2 points kinder to Claude drafts and agrees with GPT on the others: self-preference.
    _two_judges(con)

    class V:
        def __init__(self, family):
            self.family = family

    (j,) = judge_agreement(con, {"raw/claude-x": V("claude"), "local/open-y": V("qwen")})
    assert j["by_family"]["claude"]["mean_diff"] == pytest.approx(-2.0)
    assert j["by_family"]["qwen"]["mean_diff"] == pytest.approx(0.0)


def test_humans_are_one_source_averaged_per_draft(con):
    insert_run(con, "raw/a", "c1")
    insert_score(con, "judgements", "raw/a", "c1", 4, who="opus")
    insert_score(con, "human_scores", "raw/a", "c1", 3, who="anna")
    insert_score(con, "human_scores", "raw/a", "c1", 5, who="bert")
    (j,) = judge_agreement(con)
    assert (j["a"], j["b"], j["mean_b"], j["pearson_r"]) == ("opus", "human", 4.0, None)


def test_markdown_explains_how_to_get_a_second_source(con):
    insert_run(con, "raw/a", "c1")
    insert_score(con, "judgements", "raw/a", "c1", 4, who="opus")
    md = render_markdown(build(con, costs=COSTS, hours_csv=None, variants={}))
    assert "## Judge agreement" in md and "--judge JUDGE_GPT" in md


# -- cost against volume -----------------------------------------------------------
def _local(fixed, cap=None):
    return dict(fixed_eur_per_month=fixed, eur_per_reply_variable=0.0, capacity_per_month=cap)


def _api(per_reply, fixed=0.0):
    return dict(fixed_eur_per_month=fixed, eur_per_reply_variable=per_reply, capacity_per_month=None)


def test_break_even_is_fixed_cost_over_api_price():
    vol, note = _break_even(_local(1000), _api(0.01))
    assert vol == pytest.approx(100_000) and "capacity unknown" in note


def test_break_even_within_one_box():
    vol, note = _break_even(_local(1000, cap=200_000), _api(0.01))
    assert vol == pytest.approx(100_000) and note == ""


def test_never_when_a_full_box_still_costs_more_than_the_api():
    # a box serves 50,000 replies: EUR 0.02 each at best, the API charges 0.01
    vol, note = _break_even(_local(1000, cap=50_000), _api(0.01))
    assert vol is None and note.startswith("never")


def test_always_when_the_api_route_carries_a_bigger_box():
    # the abstract route pays for its local summariser
    assert _break_even(_local(168, cap=50_000), _api(0.001, fixed=168))[0] == 0


def test_volume_view_estimates_capacity_from_run_latency_when_not_measured():
    ops = [dict(_op_row("local/m", "local", hardware="box"), mean_s=36.0),
           _op_row("pseudonymise/haiku", "pseudonymise", "haiku")]
    rows = costs_table(ops, [], COSTS, {}, {})
    front = {"summary": [{"variant": "local/m", "passed": [1, 2, 4]}, {"variant": "pseudonymise/haiku", "passed": []}]}
    vv = volume_view(ops, rows, {**COSTS, "max_utilisation": 0.5}, front, {})
    m = next(x for x in vv["local"] if x["key"] == "local/m")
    assert m["capacity_per_month"] == pytest.approx(100 * 730 * 0.5) and "run latency" in m["capacity_source"]
    assert m["quality_through"] == 2  # passes 1 and 2, misses 3
    assert [a["key"] for a in vv["api"]] == ["pseudonymise/haiku"]


def test_volume_view_adds_hardware_profiles_without_a_measured_model():
    ops = [_op_row("local/m", "local", hardware="box"), _op_row("pseudonymise/haiku", "pseudonymise", "haiku")]
    vv = volume_view(ops, costs_table(ops, [], COSTS, {}, {}), COSTS, {"summary": []}, {})
    keys = {x["key"] for x in vv["local"]}
    assert "hardware:vm" in keys and "hardware:box" not in keys and "hardware:empty" not in keys
    vm = next(p for p in vv["break_even"] if p["local"] == "hardware:vm")
    assert vm["volume"] == pytest.approx(730 / vv["api"][0]["eur_per_reply_variable"])


def test_markdown_has_the_break_even_table(con):
    insert_run(con, "local/m", "c1", route="local", hardware="box")
    insert_run(con, "pseudonymise/haiku", "c1", model_key="haiku")
    md = render_markdown(build(con, costs=COSTS, hours_csv=None, variants={}))
    assert "### When local becomes cheaper" in md and "| vm (hardware only) | 730 |" in md
