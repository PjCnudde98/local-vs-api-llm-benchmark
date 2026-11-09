"""Aggregation, cost model, gates and the Markdown view (report.py)."""

import pytest

from complaintexp.config import load_gates
from complaintexp.report import (_pct, bootstrap_ci, build, cost_per_1000, effort, evaluate_gates, operations,
                                 quality, render_markdown, splits)
from complaintexp.schemas import Complaint

from .conftest import insert_run, insert_score

COSTS = {
    "volumes_per_month": [1000, 50000],
    "cloud": {"A": {"input_per_1m": None, "output_per_1m": None},
              "B": {"input_per_1m": 2.0, "output_per_1m": 8.0},
              "E": {"vm_eur_per_hour": 1.0, "hours_per_month": 730}},
    "local": {"C": {"hardware_eur": 3600, "amortisation_months": 36, "power_watts": 350, "eur_per_kwh": 0.30,
                    "hours_per_month": 730, "ops_hours_per_month": 4, "eur_per_ops_hour": 80},
              "D": {"hardware_eur": None}},
}


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
def _ops(*variants, pt=100, ct=50):
    return [{"variant": v, "mean_prompt_tokens": pt, "mean_completion_tokens": ct} for v in variants]


def test_local_cost_amortises_hardware_power_and_ops():
    (c,) = cost_per_1000(_ops("C"), COSTS)
    fixed = 3600 / 36 + 0.35 * 730 * 0.30 + 4 * 80
    assert c["fixed_eur_per_month"] == pytest.approx(fixed)
    assert c["eur_per_1000@1000"] == pytest.approx(fixed)
    assert c["eur_per_1000@50000"] == pytest.approx(fixed / 50)


def test_cloud_token_cost():
    (b,) = cost_per_1000(_ops("B"), COSTS)
    assert b["eur_per_reply"] == pytest.approx((100 * 2 + 50 * 8) / 1e6)
    assert b["eur_per_1000@1000"] == b["eur_per_1000@50000"] == pytest.approx(0.6)


def test_vm_cost_is_fixed_per_month():
    (e,) = cost_per_1000(_ops("E"), COSTS)
    assert e["fixed_eur_per_month"] == 730 and e["eur_per_1000@1000"] == 730


def test_missing_prices_give_none_not_zero():
    rows = {r["variant"]: r for r in cost_per_1000(_ops("A", "D", "FAKE"), COSTS)}
    assert all(r["eur_per_1000@1000"] is None for r in rows.values())


def test_break_even_against_B():
    rows = {r["variant"]: r for r in cost_per_1000(_ops("B", "C"), COSTS)}
    assert rows["C"]["break_even_replies_per_month_vs_B"] == pytest.approx(
        rows["C"]["fixed_eur_per_month"] / rows["B"]["eur_per_reply"])
    assert "break_even_replies_per_month_vs_B" not in rows["B"]


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


def _op(v, p95, n=40):
    return {"variant": v, "p95_s": p95, "n": n}


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


def test_redaction_quality_loss_is_A_minus_B():
    gates = evaluate_gates([], [_q("A", 4.5), _q("B", 4.2)], None, G)
    g = _status(gates, "Quality loss", "B")
    assert g["value"] == pytest.approx(0.3) and g["status"] == "fail"


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
    return [Complaint(id="s1", source="synthetic", language="en", text="synthetic text"),
            Complaint(id="r1", source="real", language="en", text="REAL CUSTOMER TEXT")]


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
    for h in ("## Gates", "## Operations", "## Cost per 1,000", "## Quality", "## Privacy", "## Effort",
              "### Quality by language"):
        assert h in md, h


def test_markdown_on_empty_db(con):
    md = render_markdown(build(con, costs=COSTS, hours_csv=None))
    assert "No scores yet" in md and "No redaction eval stored" in md


# -- display names --------------------------------------------------------------------
from complaintexp.report import display_names  # noqa: E402

from .conftest import make_variant  # noqa: E402


def _ops_with(**models):
    return [{"variant": k, "model_ids": [m] if m else []} for k, m in models.items()]


def test_variants_are_named_after_the_reported_model():
    names = display_names(["C", "D"], _ops_with(C="mistral-small3.2", D="qwen3:8b"), {})
    assert names == {"C": "mistral-small3.2", "D": "qwen3:8b"}


def test_shared_model_is_disambiguated_by_hosting():
    vs = {"A": make_variant("A", model="gpt-x", label="Frontier / Azure EU Data Zone / raw text"),
          "B": make_variant("B", model="gpt-x", label="Frontier / Azure private endpoint / pseudonymised")}
    names = display_names(["A", "B"], _ops_with(A="gpt-x", B="gpt-x"), vs)
    assert names == {"A": "gpt-x (Azure EU Data Zone, raw text)", "B": "gpt-x (Azure private endpoint, pseudonymised)"}


def test_unrun_variant_falls_back_to_config_then_label():
    vs = {"A": make_variant("A", model="${AZURE_DEPLOYMENT_FRONTIER}", label="Frontier / Azure"),
          "C": make_variant("C", model="mistral-small3.2")}
    assert display_names(["A", "C"], [], vs) == {"A": "Frontier / Azure", "C": "mistral-small3.2"}


def test_names_are_always_unique():
    vs = {k: make_variant(k, model="same", label="same") for k in "XY"}
    names = display_names(["X", "Y"], [], vs)
    assert len(set(names.values())) == 2


def test_markdown_shows_model_names_not_letters(con):
    insert_run(con, "C", "s1")
    insert_score(con, "judgements", "C", "s1", 4)
    md = render_markdown(build(con, costs=COSTS, hours_csv=None, variants={}))
    assert "| m-C |" in md and "| C |" not in md
