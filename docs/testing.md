# Testing

The experiment is only as trustworthy as the harness, so the harness has its own test suite.
It runs in about 6 seconds, needs no network and no model server, and never touches
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
- **Hand-rolled fakes, no mocking library.** `ScriptedClient` has the same `complete(system, user)`
  contract as `LLMClient`, returns or raises from a script, and records every call so tests
  can check what would have been sent. `runner.run_variant`, `judge.judge_variant` and
  `generate.generate` take an optional `client=` for this.
- **One temp database per test** (`con` fixture). Rows are inserted with the `insert_run` and
  `insert_score` helpers.
- **Behaviour, not implementation.** Tests assert on what lands in the DB, what was sent and
  what the report says, not on internal calls.
- **Shipped config is tested too.** Invariants the experiment depends on (cloud variants refuse
  real data, A-E share generation settings, `gates.yaml` matches `gates.md`) fail the build if
  someone edits a YAML file carelessly.

## What is covered

| File | Covers |
|---|---|
| `test_schemas.py` | dataset loading, required fields, bad source/language/PII, duplicates across files, error locations |
| `test_config.py` | env-var expansion, defaults merge, `extra` passthrough; invariants of the shipped YAML files |
| `test_pseudonymise.py` | every structured identifier format (BE/NL/FR IBAN, BE/FR mobile and landline, national number, order/customer ids), nl/fr/en/uk addresses merged into one placeholder, dates, amounts, years and ordinary words that must *not* be redacted, consistent placeholders, restore (incl. `_1` vs `_10`), overlaps; NER names in nl/fr/en |
| `test_prompting.py` | prompt assembly, JSON parsing of wrapped/thinking/broken output, blind normalisation |
| `test_llm.py` | request shape per provider (settings, token parameter name, JSON mode, `reasoning_effort`), usage parsing |
| `test_runner.py` | the real-data matrix (cloud x allow_real_data), resumability, retry of errored rows, parse failures kept, prompt-as-sent logging, placeholder restore, identical system prompt across variants, warm-up |
| `test_judge.py` | self-judging refused, real data never to a cloud judge, variant hidden from judge, invalid verdicts skipped and logged, resumability |
| `test_evaluate.py` | redaction recall and partial-leak rule, stored redaction runs, blind export (hidden labels, deterministic, variant filter), score import round trip, whole-sheet rejection of invalid scores |
| `test_generate.py` | generated rows load as valid records, stray PII annotations dropped, failures logged, seeded determinism, appending |
| `test_report.py` | nearest-rank percentiles, bootstrap CI, operations counts, every cost formula and break-even, quality and splits, gate semantics (inclusive vs strict, human over judge, not measured, FAKE not gated, small-sample note), effort log, real-text hiding |
| `test_html_report.py` | embedded data round-trips, hostile text cannot break out of the script tag, no external URLs, real text excluded by default, dark mode |
| `test_cli.py` | `validate`, unknown variant, a full FAKE pipeline (run, redaction-eval, export-blind, report as HTML/JSON/Markdown), resume, rejected score sheet exit code |

Line coverage is 98%. What is left is CLI glue for commands that need a live model (`judge`, `generate`).

## Bugs fixed when this suite was written (2026-11-06)

Each now has a regression test.

- Errored runs (timeouts, rate limits) were stored and then skipped forever on rerun. They are now retried.
- Belgian landline numbers (`09 123 45 67`, `+32 9 123 45 67`) were not redacted, and dates such as `05.11.2026` were redacted as phone numbers.
- A judge reply with a non-numeric hallucination flag crashed the whole judge run instead of skipping one draft.
- `import-scores` accepted scores outside 1-5 and wrote half a sheet before failing on a bad row.
- Addresses leaked: NER redacted only the city, so street, house number and postcode reached the cloud prompt. Address recall on the sample went from 50% to 100%.
- `export-blind --variants D` still picked complaints that only other variants had drafted.

## Adding a test

Put it in the file for the module it exercises, use the `con`, `sample` and `ps` fixtures,
and fake the model with `ScriptedClient([...])`. If you add a privacy rule, add the case that
must be blocked *and* the one that must still pass.
