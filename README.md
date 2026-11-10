# Local vs API LLMs for complaint handling

Test harness for the experiment plan "on-prem vs cloud LLM for complaint answering"
(https://claude.ai/artifact/NuoBbtVYJRhM49BrSuaxwv). Every model answers the same complaints
with the same prompt, so model, hosting and data handling are the only differences.

The experiment answers one main question: **where do locally hosted models stop being good
enough, and where do they break?** Every model climbs the same six-tier ladder, from
classification to long-context reasoning. The report shows the first tier each model fails,
for quality or for cost.

| Tier | Task | Pass threshold |
|---|---|---|
| 1 | Classify type, language, urgency, sentiment | 95% label accuracy |
| 2 | Extract order number, product, remedy, deadline | F1 0.95; 99% valid JSON |
| 3 | Draft a reply to a simple complaint | rubric 4.0 |
| 4 | Draft a reply to an angry, multi-issue complaint | rubric 4.0; 98% policy compliant |
| 5 | Decide the remedy with lookup tools | 90% correct; 98% valid tool calls |
| 6 | Resolve a long email thread | 95% consistent; < 5% key facts missed |

| Size class | Models | Runs on |
|---|---|---|
| Cloud | Claude Sonnet 5.5, Claude Haiku 5.5, mid-tier GPT | API (pseudonymised, abstracted or raw) |
| XL local | GLM-5.3, DeepSeek-V4.1-Flash, Kimi K2.6, Qwen3.8-Flash-Next | server with 512 GB RAM + GPUs |
| Large local | Mistral Small 4 | 80 GB+ GPU |
| Medium local | Qwen3.6-27B, Gemma 4 26B A4B, gpt-oss-20b | one 24-48 GB GPU |
| Small local | Gemma 4 12B, Qwen3 8B | laptop |

Each runs with reasoning off and on. `cexp variants` lists all 42 variants.

- Full experiment spec (questions, routes, ladder, data, metric definitions, gates, threats to validity): [docs/experiment.md](docs/experiment.md)
- Models, XL server sizing, serving commands and thinking switches: [docs/models.md](docs/models.md)
- Decision thresholds: [docs/gates.md](docs/gates.md) and `config/gates.yaml`
- How the harness is tested: [docs/testing.md](docs/testing.md)

## Setup

```bash
uv sync --extra dev --extra redaction      # redaction = Presidio + spaCy
uv run python -m spacy download en_core_web_md nl_core_news_md fr_core_news_md
cp .env.example .env                        # API keys, local server URLs, model names
uv run pytest
uv run cexp run --variants FAKE             # dry run, no network
```

Without the `redaction` extra the pseudonymiser only catches emails, IBANs, phones, national
numbers, order IDs and addresses. Names stay in. `cexp run` warns about this.

## Workflow (maps to the plan's phases)

```bash
uv run cexp validate --data data/synthetic/sample.jsonl          # data, gold answers, PII annotations
uv run cexp generate --n 25                                       # 25 synthetic complaints per tier -> agent review
uv run cexp redaction-eval --data data/real/real.jsonl           # recall per PII type, no model calls

# Runs: select variants by key or glob. Real data only ever reaches local variants.
uv run cexp run --variants 'local/qwen3*' 'local/gemma*' --warmup --data data/synthetic/*.jsonl
uv run cexp run --variants 'local/glm-5.3*' --warmup --data data/synthetic/*.jsonl   # one XL model at a time
uv run cexp run --variants 'pseudonymise/*' 'abstract/*' 'raw/*' --warmup --data data/synthetic/*.jsonl
uv run cexp run --variants 'local/*' --data data/real/real.jsonl
uv run cexp throughput --variants local/qwen3.6-27b --concurrency 1 2 4 8   # capacity for the cost model

uv run cexp judge --variants '*' --judge JUDGE --data data/synthetic/*.jsonl     # Claude judge, tiers 3/4/6
uv run cexp judge --variants '*' --judge JUDGE_GPT --data data/synthetic/*.jsonl # second judge, other family (Azure)
#   no Azure: --judge JUDGE_OPENAI with OPENAI_API_KEY and OPENAI_JUDGE_MODEL in .env
uv run cexp export-blind --drafts 60                             # small human overlap sample, balanced over variants
uv run cexp import-scores results/blind_review.csv --key results/blind_review.key.csv --rater anna
uv run cexp report --html --out results/report.md                # frontier, gates, cost, quality, privacy
```

`cexp report --html` writes `results/report.html`: a single offline page with the capability
frontier (one line per model across the six tiers, threshold marked), every gate, filters by
variant, tier, route and language, charts, cost per sendable reply, the monthly volume from
which each local option is cheaper than each API route, how far the judges agree, and every
model's answer side by side per complaint. Real complaint text is left out unless you pass `--include-real-text`.

## Routes

| Route | What leaves your infrastructure | Real data |
|---|---|---|
| `local` | nothing | yes, after DPO sign-off |
| `pseudonymise` | text with identifiers replaced by placeholders | no |
| `abstract` | a case summary written by a local model from pseudonymised text | no |
| `raw` | the full text (quality ceiling, synthetic data only) | no |
| `hybrid` | easy complaints stay local, hard ones go to an API with preprocessing | only if routed locally |

## Safeguards built in

- Real complaints (`source: "real"`) are refused by any variant with `cloud: true` or
  `allow_real_data: false`, by a hybrid's cloud target and by a cloud judge. See `runner.check_allowed`.
- The run log stores the prompt exactly as sent (pseudonymised or abstracted for cloud routes),
  and counts annotated PII still visible in it.
- Runs are resumable and keyed by (variant, complaint): re-running skips finished work and retries errors.
- A judge model equal to the variant's model is rejected.
- `data/real/`, `results/` and `.env` are git-ignored.
- The test suite blocks all network access.

## Before the first scored run

1. `prompts/policy.md` is a policy written for this experiment (fictional retailer). Replace it
   if you test for a real company, update the tier-5 cases and tier-6 rule numbers in
   `data/synthetic/scenarios.yaml` to match, and freeze prompts at gate 3.
2. Serve the local models and set names and URLs in `.env` ([docs/models.md](docs/models.md)).
   Send one request per variant to check the thinking switch works.
3. Lock model versions and quants at gate 1 and write them down.
4. Fill in `config/costs.yaml` (prices, hardware quotes, exchange rate); `null` shows as n/a.
   Hardware values marked illustrative are estimates so the break-even chart has something to show.
5. Adjust `DEFAULT_ID_PATTERNS` in `src/complaintexp/pseudonymise.py` to your order/customer ID formats.
6. Log hours per task in `hours_log.csv`; effort is one of the experiment's results.

## Not built yet

- Pairwise preference export/import (plan: local wins or ties >= 40% vs pseudonymised API).
- Edit distance against sent replies for the real set.
- Placeholder-integrity metric (placeholders the model lost or rewrote).
- Azure/Foundry check of abuse-monitoring and retention settings (manual, before sending anything).
