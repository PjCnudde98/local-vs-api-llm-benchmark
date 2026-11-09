"""Variant/cost/rubric/gate configuration (config.py)."""

import pytest

from complaintexp.config import load_costs, load_gates, load_rubric, load_variants


@pytest.fixture
def variants_yaml(tmp_path):
    p = tmp_path / "variants.yaml"
    p.write_text(
        """
defaults: {temperature: 0.2, max_tokens: 800, json_mode: true}
variants:
  X:
    label: x
    provider: openai_compatible
    base_url: ${CEXP_TEST_URL}
    model: ${CEXP_TEST_MODEL}
    reasoning_effort: none
    max_tokens: 100
judge:
  label: j
  provider: fake
  model: judge-1
  temperature: 0.0
""",
        encoding="utf-8",
    )
    return p


def test_env_vars_are_expanded(variants_yaml, monkeypatch):
    monkeypatch.setenv("CEXP_TEST_URL", "http://localhost:9/v1")
    monkeypatch.setenv("CEXP_TEST_MODEL", "tiny")
    v = load_variants(variants_yaml)["X"]
    assert (v.base_url, v.model) == ("http://localhost:9/v1", "tiny")


def test_variant_overrides_defaults_and_keeps_the_rest(variants_yaml):
    v = load_variants(variants_yaml)["X"]
    assert v.max_tokens == 100 and v.temperature == 0.2 and v.json_mode is True


def test_unknown_keys_go_to_extra(variants_yaml):
    assert load_variants(variants_yaml)["X"].extra == {"reasoning_effort": "none"}


def test_judge_is_loaded_as_JUDGE(variants_yaml):
    j = load_variants(variants_yaml)["JUDGE"]
    assert j.model == "judge-1" and j.temperature == 0.0


def test_safe_defaults_for_missing_flags(variants_yaml):
    v = load_variants(variants_yaml)["X"]
    assert v.cloud is False and v.allow_real_data is False and v.pii == "none"


# -- the shipped config files: invariants the experiment depends on ----------
def test_shipped_cloud_variants_never_allow_real_data():
    for k, v in load_variants().items():
        if v.cloud:
            assert not v.allow_real_data, f"cloud variant {k} must keep allow_real_data: false"


def test_shipped_variants_share_generation_settings():
    """Only model, hosting and PII handling may differ (plan: one prompt, one setting)."""
    vs = [v for k, v in load_variants().items() if k in {"A", "B", "C", "D", "E"}]
    assert len({(v.temperature, v.max_tokens, v.json_mode) for v in vs}) == 1


def test_shipped_B_and_E_pseudonymise():
    vs = load_variants()
    assert vs["B"].pii == "pseudonymise" and vs["E"].pii == "pseudonymise"


def test_shipped_rubric_has_scale_and_criteria():
    r = load_rubric()
    assert r["scale"] == [1, 5] and len(r["criteria"]) >= 3


def test_shipped_costs_have_volumes():
    assert load_costs()["volumes_per_month"]


def test_shipped_gates_match_docs():
    """config/gates.yaml must state the same thresholds as docs/gates.md."""
    from complaintexp import ROOT

    g = load_gates()
    doc = (ROOT / "docs" / "gates.md").read_text(encoding="utf-8")
    assert g["quality"]["rubric_mean_min"] == 4.0 and ">= 4.0" in doc
    assert g["quality"]["hallucination_rate_max"] == 0.02 and "< 2%" in doc
    assert g["privacy"]["redaction_recall_min"] == 0.98 and ">= 98%" in doc
    assert g["operations"]["p95_latency_s_max"] == 20 and "< 20 s" in doc
