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


# -- lineup expansion ------------------------------------------------------------
LINEUP = """
defaults: {temperature: 0.2, max_tokens: 800, max_tokens_by_tier: {1: 300}}
routes:
  local: {pii: none, cloud: false, allow_real_data: true}
  pseudonymise: {pii: pseudonymise, cloud: true, allow_real_data: false}
models:
  small:
    label: Small 1B
    size_class: small
    provider: openai_compatible
    model: ${CEXP_TEST_SMALL:-small:1b}
    routes: [local]
    reasoning:
      off: {reasoning_effort: none}
      on: {reasoning_effort: high}
  api:
    label: Api
    size_class: cloud
    provider: anthropic
    model: api-1
    temperature: null
    routes: [local, pseudonymise]
variants:
  hybrid/x: {kind: hybrid, label: h, router: local/small, easy: local/small, hard: pseudonymise/api, hard_tiers: [5]}
judges:
  JUDGE: {label: j, provider: fake, model: judge-1}
"""


@pytest.fixture
def lineup(tmp_path, monkeypatch):
    monkeypatch.delenv("CEXP_TEST_SMALL", raising=False)
    p = tmp_path / "lineup.yaml"
    p.write_text(LINEUP, encoding="utf-8")
    return load_variants(p)


def test_models_expand_into_route_and_reasoning_variants(lineup):
    assert {"local/small", "local/small+think", "local/api", "pseudonymise/api", "hybrid/x", "JUDGE"} == set(lineup)


def test_on_and_off_keys_survive_yaml_booleans(lineup):
    assert lineup["local/small"].reasoning == "off" and lineup["local/small+think"].reasoning == "on"
    assert lineup["local/small+think"].request == {"reasoning_effort": "high"}


def test_route_sets_privacy_flags(lineup):
    v = lineup["pseudonymise/api"]
    assert (v.route, v.pii, v.cloud, v.allow_real_data) == ("pseudonymise", "pseudonymise", True, False)
    assert v.model_key == "api" and v.model_label == "Api" and v.temperature is None


def test_env_default_syntax(lineup, monkeypatch):
    assert lineup["local/small"].model == "small:1b"


def test_tier_budgets_are_ints(lineup):
    assert lineup["local/small"].max_tokens_by_tier == {1: 300} and lineup["local/small"].tokens_for(1) == 300


def test_hybrid_variant(lineup):
    h = lineup["hybrid/x"]
    assert h.route == "hybrid" and h.hybrid["hard"] == "pseudonymise/api" and h.hybrid["hard_tiers"] == [5]


def test_unknown_route_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("routes: {}\nmodels:\n  m: {provider: fake, model: m, routes: [nowhere]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown route"):
        load_variants(p)


def test_select_by_glob_and_key(lineup):
    from complaintexp.config import select

    assert select(lineup, ["local/*"]) == ["local/small", "local/small+think", "local/api"]
    assert select(lineup, ["hybrid/x", "local/small"]) == ["hybrid/x", "local/small"]
    with pytest.raises(KeyError):
        select(lineup, ["nothing/*"])
    assert "JUDGE" not in select(lineup, ["*"])


# -- the shipped config files: invariants the experiment depends on ----------
def test_shipped_cloud_variants_never_allow_real_data():
    for k, v in load_variants().items():
        if v.cloud:
            assert not v.allow_real_data, f"cloud variant {k} must keep allow_real_data: false"


def test_shipped_local_variants_are_local():
    for k, v in load_variants().items():
        if v.route == "local":
            assert not v.cloud and v.pii == "none", k


def test_shipped_variants_share_generation_settings():
    """Within a tier every variant gets the same token budget and output mode (plan: one prompt, one setting).

    Temperature may only be absent (the API rejects it), never different.
    """
    vs = [v for k, v in load_variants().items() if k != "FAKE" and not k.startswith("JUDGE") and v.route != "hybrid"]
    assert len({(tuple(sorted(v.max_tokens_by_tier.items())), v.reasoning_extra_tokens, v.json_mode) for v in vs}) == 1
    assert {v.temperature for v in vs} <= {0.2, None}


def test_shipped_lineup_covers_every_size_class_and_route():
    vs = load_variants()
    assert {v.size_class for v in vs.values()} >= {"cloud", "xl", "large", "medium", "small"}
    assert {v.route for v in vs.values()} >= {"local", "pseudonymise", "abstract", "raw", "hybrid"}


def test_shipped_xl_class_has_one_model_per_requested_family():
    xl = {v.family for v in load_variants().values() if v.size_class == "xl"}
    assert xl == {"kimi", "glm", "deepseek", "qwen"}


def test_shipped_references_resolve():
    vs = load_variants()
    for k, v in vs.items():
        if v.abstractor:
            assert vs[v.abstractor].route == "local", k
        for role in ("router", "easy", "hard"):
            if v.hybrid.get(role):
                assert v.hybrid[role] in vs, (k, role)


def test_shipped_local_models_have_a_cost_profile():
    hw = load_costs()["hardware"]
    for k, v in load_variants().items():
        if v.route == "local" and v.provider != "fake":
            assert v.hardware in hw, k


def test_shipped_rubric_has_scale_and_criteria():
    r = load_rubric()
    assert r["scale"] == [1, 5] and len(r["criteria"]) >= 3


def test_shipped_costs_have_volumes():
    assert load_costs()["volumes_per_month"]


def test_shipped_tier_thresholds_match_docs():
    from complaintexp import ROOT

    g = load_gates()["tiers"]
    doc = (ROOT / "docs" / "gates.md").read_text(encoding="utf-8")
    assert [m["threshold"] for m in g[1]] == [0.95] and "| 1. Classify | label accuracy | >= 95% |" in doc
    assert [m["threshold"] for m in g[5]] == [0.90, 0.98] and "| 5. Policy reasoning with tools | decision accuracy | >= 90% |" in doc
    assert g[6][1] == {"metric": "missed_fact_rate", "op": "<", "threshold": 0.05} and "missed key facts | < 5% |" in doc


def test_shipped_gates_match_docs():
    """config/gates.yaml must state the same thresholds as docs/gates.md."""
    from complaintexp import ROOT

    g = load_gates()
    doc = (ROOT / "docs" / "gates.md").read_text(encoding="utf-8")
    assert g["quality"]["rubric_mean_min"] == 4.0 and ">= 4.0" in doc
    assert g["quality"]["hallucination_rate_max"] == 0.02 and "< 2%" in doc
    assert g["privacy"]["redaction_recall_min"] == 0.98 and ">= 98%" in doc
    assert g["operations"]["p95_latency_s_max"] == 20 and "< 20 s" in doc
