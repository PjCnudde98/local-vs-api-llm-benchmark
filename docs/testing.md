# Testing

The experiment is only as trustworthy as the harness, so the harness has its own test suite.
It runs in about 5 seconds, needs no network and no model server, and never touches
`results/experiment.sqlite`.

```bash
uv sync --extra dev --extra redaction
uv run pytest                                  # everything
uv run pytest -m "not ner"                     # skip the spaCy/Presidio tests
uv run pytest --cov=complaintexp --cov-report=term-missing
```

Tests marked `ner` need the `redaction` extra plus the spaCy models; they skip themselves
when those are missing.

## Rules

- **No network.** An autouse fixture (`tests/conftest.py::_no_network`) makes every socket
  connect raise. A test that forgot to inject a fake fails instead of calling Azure or Ollama.
- **Hand-rolled fakes, no mocking library.** `ScriptedClient` has the same
  `complete(system, user, tier)` and `chat(system, messages, tier)` contract as `LLMClient`,
  returns or raises from a script, and records every call and conversation so tests can check
  what would have been sent. `runner.run_variant` takes `client=` (the variant itself) and
  `clients=` (by key: abstractor, router, hybrid targets); `judge.judge_variant` and
  `generate.generate` take `client=`. The Claude and OpenAI request builders are pure
  functions, tested by recording the kwargs they would send.
- **One temp database per test** (`con` fixture). Rows are inserted with the `insert_run` and
  `insert_score` helpers.
- **Behaviour, not implementation.** Tests assert on what lands in the DB, what was sent and
  what the report says, not on internal calls.
- **Shipped config is tested too.** Invariants the experiment depends on (cloud routes refuse
  real data, every variant shares the per-tier generation settings, the XL class has one model
  per requested family, every local model has a cost profile, abstractor and hybrid references
  resolve, `gates.yaml` matches `gates.md`) fail the build if someone edits a YAML file carelessly.
- **The sample set covers every tier**, so the FAKE end-to-end test exercises the whole ladder.

## What is covered

| File | Covers |
|---|---|
| `test_schemas.py` | dataset loading, required fields (tier included), bad source/language/tier/PII, gold validation per tier, `received_on` format and its requirement for tiers 5 and 6, duplicates across files, error locations |
| `test_config.py` | env-var expansion with `${VAR:-default}`, models x routes x reasoning expansion (YAML `on`/`off` booleans), hybrid variants, glob selection; invariants of the shipped YAML files |
| `test_pseudonymise.py` | every structured identifier format, nl/fr/en/uk addresses merged into one placeholder, a city after a street absorbed, dates, amounts and words that must *not* be redacted, consistent placeholders across a conversation, restore, overlaps; NER names in nl/fr/en; NER leaves JSON field names alone |
| `test_prompting.py` | base prompt shared by every tier's task prompt, the date received as today, tier-6 thread and policy pages, the abstract route's replacement text, output contract per tier, blind normalisation |
| `test_llm.py` | OpenAI-compatible request shape (token budget per tier and reasoning, extra_body, temperature omitted, JSON mode, multi-turn), Claude request shape (no temperature, thinking, effort), usage parsing, the fake model per tier, a hosted OpenAI-compatible endpoint reading its own API key, the shipped `JUDGE_OPENAI` |
| `test_runner.py` | the real-data matrix, resumability, retries, parse failures kept, prompt-as-sent logging, placeholder restore, identical system prompt per tier, tier 1/2 automatic scoring, the tier-5 tool loop (lookups, invalid calls, no decision, pseudonymised args and results), the abstract route, leakage counting, hybrid routing and its real-data rule, warm-up of every model involved |
| `test_judge.py` | self-judging refused (hybrid targets too), real data never to a cloud judge, variant hidden, only draft tiers judged, tier-6 consistency and missed facts, judge choice, invalid verdicts skipped and logged, resumability |
| `test_evaluate.py` | redaction recall and the partial-leak rule, stored redaction runs, blind export (draft tiers only, hidden labels, deterministic, variant filter, a draft cap balanced over variants), score import round trip, whole-sheet rejection |
| `test_generate.py` | generated rows load as valid records per tier, harness-chosen gold for tiers 1 and 5, generator gold for tiers 2 and 6, invalid gold dropped, stray PII annotations dropped, seeded determinism, appending, a received date inside the configured range told to the generator, the tier-5 rule quoted, tier-6 pages that resolve the contradiction plus one distractor |
| `test_report.py` | percentiles, bootstrap CI, operations (draft-tier latency), cost per reply and per sendable reply, USD conversion, capacity cliff, break-even vs cheapest API, abstractor box; gate semantics, privacy loss per route; every tier metric, malformed answers as failures, failure point, economic failure; display names; real-text hiding; judge agreement (pairs of sources, humans averaged, the gap by drafting family); break-even volume (within one box, "never" when a full box costs more, "always" against the abstract route's box), capacity estimated from run latency, hardware-only profiles |
| `test_store.py` | migration of a v0.2 results database |
| `test_html_report.py` | embedded data round-trips, hostile text cannot break out of the script tag, no external URLs, real text excluded by default, dark mode, frontier embedded, cost-against-volume data and section |
| `test_cli.py` | `validate`, `variants` with globs, unknown variant, a full FAKE pipeline over all six tiers (run, redaction-eval, export-blind, report as HTML/JSON/Markdown), `--tiers`, `rescore`, `throughput`, resume, rejected score sheet exit code |

Line coverage is 96%. What is left is CLI glue and SDK client construction for commands that
need a live model or API key.

## Bugs fixed when this suite was written (2026-11-06)

Each now has a regression test.

- Errored runs (timeouts, rate limits) were stored and then skipped forever on rerun. They are now retried.
- Belgian landline numbers (`09 123 45 67`, `+32 9 123 45 67`) were not redacted, and dates such as `05.11.2026` were redacted as phone numbers.
- A judge reply with a non-numeric hallucination flag crashed the whole judge run instead of skipping one draft.
- `import-scores` accepted scores outside 1-5 and wrote half a sheet before failing on a bad row.
- Addresses leaked: NER redacted only the city, so street, house number and postcode reached the cloud prompt. Address recall on the sample went from 50% to 100%.
- `export-blind --variants D` still picked complaints that only other variants had drafted.

Found while building v0.3 (2026-11-09), each with a regression test:

- YAML 1.1 reads the keys `on:` and `off:` as booleans, which collapsed every reasoning on/off
  pair into one variant.
- A city right after an English street ("14 Elm Grove, Bristol") leaked when NER missed it.

## Adding a test

Put it in the file for the module it exercises, use the `con`, `sample` and `ps` fixtures,
and fake the model with `ScriptedClient([...])`. If you add a privacy rule, add the case that
must be blocked *and* the one that must still pass.
