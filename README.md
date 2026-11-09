# On-prem vs cloud LLM for complaint replies

Test harness for the experiment plan "on-prem vs cloud LLM for complaint answering"
(https://claude.ai/artifact/NuoBbtVYJRhM49BrSuaxwv). It runs the same complaints through
the same prompt on every variant, so model and hosting are the only differences.

- Full experiment spec (variants, data, metric definitions, gates, threats to validity): [docs/experiment.md](docs/experiment.md)
- Decision thresholds: [docs/gates.md](docs/gates.md) and `config/gates.yaml`
- How the harness is tested: [docs/testing.md](docs/testing.md)

| Variant | Model | Hosting | PII handling |
|---|---|---|---|
| A | Frontier | Azure EU Data Zone | raw text, synthetic data only |
| B | Frontier | Azure private endpoint | local pseudonymisation first |
| C | Mid-size open (20-35B) | local GPU | none needed |
| D | Small open (7-9B) | laptop | none needed |
| E (optional) | same as C | GPU VM in own Azure subscription | pseudonymised |

## Setup

```bash
uv sync --extra dev --extra redaction      # redaction = Presidio + spaCy
uv run python -m spacy download en_core_web_md nl_core_news_md fr_core_news_md
cp .env.example .env                        # fill in Azure + local endpoints
uv run pytest
```

Without the `redaction` extra the pseudonymiser only catches emails, IBANs, phones,
national numbers and order IDs. Names and addresses stay in. `cexp run` warns about this.

## Workflow (maps to the plan's phases)

```bash
uv run cexp validate --data data/synthetic/sample.jsonl         # check data + PII annotations
uv run cexp generate --variant A --n 150                        # LLM-draft synthetic set -> agent review
uv run cexp redaction-eval --data data/real/real.jsonl          # recall per PII type, no model calls; stored for the report
uv run cexp run --variants C D --warmup --data data/real/real.jsonl      # real set: local variants only
uv run cexp run --variants A B C D --warmup --data data/synthetic/*.jsonl
uv run cexp judge --variants A B C D --data data/synthetic/*.jsonl
uv run cexp export-blind --n 60                                 # shuffled sheet + separate key
uv run cexp import-scores results/blind_review.csv --key results/blind_review.key.csv --rater anna
uv run cexp report --html --out results/report.md               # gates, latency, cost, scores, privacy, effort
```

`cexp report --html` writes `results/report.html`: a single offline page with every gate,
filters by variant/language/type/difficulty, charts, and every variant's draft side by side
per complaint. Real complaint text is left out unless you pass `--include-real-text`.
`--json` writes the underlying data.

`cexp run --variants FAKE` is a no-network dry run.

## Safeguards built in

- Real complaints (`source: "real"`) are refused by any variant with `cloud: true` or
  `allow_real_data: false`, and by a cloud judge. See `runner.check_allowed`.
- The run log stores the prompt exactly as sent (pseudonymised for B), never raw text for cloud runs.
- Runs are resumable and keyed by (variant, complaint): re-running skips finished work and retries errors.
- A judge model equal to the variant's model is rejected.
- `data/real/` and `results/` are git-ignored.
- The HTML/JSON report embeds real complaint text only with `--include-real-text`.
- The test suite blocks all network access.

## Data format

JSONL, one complaint per line: `id, source, language, text, type, difficulty, facts, pii, gold_reply, sent_reply`.
`pii` is `[{"type": "PERSON", "text": "Jan Peeters"}]`, hand-annotated by text.

## Before the first scored run

1. Replace `prompts/policy.md` (placeholder) and freeze prompts at gate 3.
2. Pick concrete models in `.env` (shortlist 2-3 per size class; test Dutch and French).
3. Fill in `config/costs.yaml` (prices, hardware quote); `null` shows as n/a.
4. Adjust `DEFAULT_ID_PATTERNS` in `src/complaintexp/pseudonymise.py` to your order/customer ID formats.
5. Log hours per task in `hours_log.csv`; effort is one of the experiment's results.

## Not built yet

- Pairwise preference export/import (plan: local wins or ties >= 40% vs B).
- Edit-distance against sent replies for the real set.
- Placeholder-integrity metric for B/E (placeholders the model lost or rewrote).
- Azure-side check of abuse-monitoring and retention settings (manual, before sending anything).
