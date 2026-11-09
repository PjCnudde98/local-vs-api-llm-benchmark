# Experiment specification: on-prem vs cloud LLM for complaint replies

This document describes the experiment end to end: the question, the variants, the data,
how every number in the report is computed, and what can make those numbers wrong.
Plan of record: <https://claude.ai/artifact/NuoBbtVYJRhM49BrSuaxwv>. Decision thresholds:
[gates.md](gates.md) and `config/gates.yaml`. How the code is tested: [testing.md](testing.md).

## 1. Question

Can a model we host ourselves draft customer complaint replies well enough, fast enough and
cheaply enough to replace or complement a frontier model hosted on Azure, given that real
complaints contain personal data?

The harness answers this with five comparable variants that see the same complaints, the
same prompt and the same generation settings. Model, hosting and PII handling are the only
things that change.

## 2. Variants

| Key | Model class | Hosting | PII handling | Real data allowed | Configured model (today) |
|---|---|---|---|---|---|
| A | Frontier | Azure EU Data Zone | raw text | no | `AZURE_DEPLOYMENT_FRONTIER` (not set) |
| B | Frontier | Azure private endpoint | pseudonymised locally first | no | same as A |
| C | Mid-size open (20-35B) | local GPU / Ollama | none needed | yes | `mistral-small3.2` |
| D | Small open (7-9B) | laptop / Ollama | none needed | yes | `qwen3:8b` (thinking off) |
| E (optional) | same model as C | GPU VM in own Azure subscription | pseudonymised | no | same as C |
| FAKE | none | in-process | none | yes | `fake-1`: echoes input, for dry runs |
| JUDGE | another model family than A | Azure | raw (synthetic only) | no | `AZURE_DEPLOYMENT_JUDGE` (not set) |

Source: `config/variants.yaml`, with model names from `.env`.

**Held constant across A-E** (enforced by `tests/test_config.py::test_shipped_variants_share_generation_settings`
and `tests/test_runner.py::test_all_variants_get_the_identical_system_prompt`):

- System prompt `prompts/system.md` with policy `prompts/policy.md` inlined.
- User message layout (`prompting.build_user_message`): complaint text, language tag, customer facts as JSON.
- `temperature: 0.2`, `max_tokens: 800`, JSON output mode.
- Output contract: `{"draft_reply": ..., "agent_note": ...}`.

Allowed per-variant differences: endpoint, model, the token parameter name
(`max_tokens` vs `max_completion_tokens`) and `reasoning_effort` (used to switch off
qwen3 thinking, which otherwise spends the token budget before the answer).

## 3. Data

### 3.1 Sets

| Set | File | Source | May go to | Status |
|---|---|---|---|---|
| Sample | `data/synthetic/sample.jsonl` | hand-written, 6 complaints (2 nl, 2 fr, 2 en) | every variant | in use |
| Generated | `data/synthetic/generated.jsonl` | `cexp generate`, LLM-written from `scenarios.yaml` | every variant after agent review | not generated yet |
| Real | `data/real/*.jsonl` | historical complaints | local variants only (C, D) | not loaded; git-ignored |

Generated rows carry `"status": "needs_review"`. A support agent checks realism before they count.

### 3.2 Record format (JSONL, one complaint per line)

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique across all loaded files |
| `source` | yes | `synthetic` or `real`; drives the real-data safeguard |
| `language` | yes | `nl`, `fr` or `en` |
| `text` | yes | the complaint as the customer wrote it |
| `type` | no | delivery, damage, billing, refund, staff, repeat |
| `difficulty` | no | simple, angry, rambling, multi_issue, legal, out_of_policy |
| `facts` | no | structured customer facts given to the model (order status, dates, amounts) |
| `pii` | no | hand annotations `[{"type": "PERSON", "text": "Jan Peeters"}]`, matched by text, not offsets |
| `gold_reply` | no | an experienced agent's reply (target: ~30 complaints) |
| `sent_reply` | no | real set only: the reply that was actually sent |

`load_dataset` refuses duplicate ids, unknown sources or languages, missing required fields and
malformed `pii`, and reports `file:line`. `cexp validate` also checks that every PII
annotation occurs verbatim in the text.

### 3.3 Scenario axes for generation

`data/synthetic/scenarios.yaml`: language weights (nl 0.5, fr 0.3, en 0.2), six complaint
types, six difficulty styles, 3-5 PII patterns per complaint and an optional sensitive context
(health, bereavement, financial hardship). Generation is seeded (`--seed`), so the same seed
gives the same scenario mix.

## 4. Procedure

Phases follow the plan. Commands are run from the project root.

| Phase | Step | Command |
|---|---|---|
| 0. Setup | install, models, `.env` | `uv sync --extra dev --extra redaction` |
| 1. Data | validate sample, generate + review synthetic set | `cexp validate`, `cexp generate --n 150` |
| 2. Privacy | measure redaction recall (no model calls) | `cexp redaction-eval --data ...` |
| Gate | DPO approval and Azure settings checked | manual |
| 3. Drafting | run all variants on the synthetic set; C/D also on real | `cexp run --variants A B C D --warmup` |
| Gate | prompts frozen | manual: no edits to `prompts/` after this |
| 4. Scoring | LLM judge; blind human scoring by 2-3 agents | `cexp judge`, `cexp export-blind`, `cexp import-scores` |
| 5. Readout | report against the gates | `cexp report --html --out results/report.md` |

Effort is a result too: every person logs hours in `hours_log.csv`
(`date,person,phase,task_category,variant,hours,notes`).

### 4.1 What happens to one complaint (`runner.run_variant`)

1. **Safeguard.** `check_allowed` raises for a `real` complaint if the variant has
   `cloud: true` or `allow_real_data: false`. Blocked complaints are counted, never sent, never stored.
2. **Prompt.** System prompt plus `build_user_message(complaint)`.
3. **Pseudonymise** (B and E only). `Pseudonymiser.redact` replaces identifiers with
   consistent placeholders (`<PERSON_1>`, `<EMAIL_1>`) and keeps the mapping in memory.
4. **Call.** One chat-completions request; latency is timed around this call only.
5. **Parse.** Strict JSON first, then the first `{...}` block. Anything else is a parse failure
   (stored, counted as a failure, not retried).
6. **Restore.** Placeholders in the draft and note are mapped back to the real values.
7. **Log.** One row per (variant, complaint) in `results/experiment.sqlite`, including the
   prompt exactly as sent (pseudonymised for B/E) and the raw output.

Runs are resumable: successful rows are skipped on a rerun, rows that ended in an error
(timeout, rate limit, server down) are retried.

## 5. Metrics: exact definitions

All metrics come from `report.build()`. The Markdown and HTML reports are two views of the same dict.

### 5.1 Operations

| Metric | Definition |
|---|---|
| Time to draft | Client-side wall clock around the chat-completions call (`time.perf_counter`). Includes network and server queueing. Excludes prompt building and pseudonymisation (pseudonymisation time is not measured yet). |
| p50 / p95 | Nearest-rank percentile over runs without an error, parse failures included: the k-th smallest value with k = ceil(q * n). No interpolation. |
| Failure rate | (errors + parse failures) / all stored rows for that variant. |
| Tokens | `prompt_tokens` / `completion_tokens` as reported by the server, averaged over runs without an error. |
| Model id | The model string the server reports, stored per row to catch silent model swaps. |

`cexp run --warmup` sends one untimed dummy request first, so a local server's model-load
time does not land in the first complaint's latency. Use it for every scored run.

### 5.2 Quality

Rubric (`config/rubric.yaml`), shared by human raters and the LLM judge, integers 1-5:
factual correctness, policy compliance, tone and empathy, language quality, completeness.
Plus a hallucination flag (0/1).

| Metric | Definition |
|---|---|
| Per-criterion mean | Mean over all score rows (one row per rater per draft, or per judge model per draft). |
| Overall | Mean of the per-criterion means. |
| 95% CI | Percentile bootstrap of the mean of per-row overall scores; 2,000 resamples, seed 0. Shown only for n >= 2. |
| Hallucination rate | Rows flagged 1 / rows with a flag. |
| Edit effort | Median `edit_minutes` from human sheets: how long the rater would need to make the draft sendable. |
| Splits | Overall per language, complaint type and difficulty (and p50 latency per split). |

Blind scoring: `export-blind` shuffles drafts of all variants with a fixed seed (42), strips
formatting that could reveal the model (`normalise_for_blind`: markdown bold, headings,
quotes, extra blank lines) and writes the variant key to a separate file. `import-scores`
validates the entire sheet before writing anything: unknown blind ids, scores outside 1-5,
a hallucination flag other than 0/1 or a negative edit time reject the import.

The judge never sees which variant wrote a draft, is refused when its model equals the
variant's model, and never receives real complaints when it runs in the cloud.

### 5.3 Privacy

| Metric | Definition |
|---|---|
| Redaction recall, per PII type | Share of hand-annotated values that do not survive `Pseudonymiser.redact`. A value leaks when the full value, or any of its tokens of 4+ characters, still appears in the redacted text (case-insensitive). Partial leaks count: "Kerkstraat 12, 9000 Gent" with only the street redacted still leaks "9000" and "Gent". Tokens under 4 characters ("Jan", "12") are ignored because they match ordinary text. |
| Quality loss from redaction | Overall(A) - Overall(B), using the same score source for both. |

`cexp redaction-eval` stores each run in the `redaction_eval` table; the report uses the latest one.
Without the `redaction` extra the pseudonymiser runs regex-only (emails, IBANs, phones,
national numbers, order and customer ids, addresses) and the report marks the gate "regex only".

**Addresses** are regex-based in both modes, because spaCy tags at most the city and never the
street, house number or postcode. Recognised forms: Dutch/Flemish street + number
(`Kerkstraat 12`, `Grote Markt 1`, `Stationsplein 4 bus 2`), French number-first and number-last
(`12 rue de la Paix`, `avenue Louise 54 bte 3`), English (`221B Baker Street`), BE/FR/NL
postcode + city (`9000 Gent`, `75002 Paris`, `1012 LG Amsterdam`) and UK postcodes. Adjacent
pieces, including a city found by NER, merge into one `<ADDRESS_n>` placeholder, so the model
sees `<ADDRESS_1>` instead of street and city as separate fragments.

### 5.4 Cost per 1,000 replies

Inputs in `config/costs.yaml`; `null` stays `n/a` in the report rather than a made-up number.

- **Cloud per token (A, B):** (mean prompt tokens x input price + mean completion tokens x output price) / 1e6 per reply, x 1,000.
- **VM (E):** EUR/hour x hours/month, spread over the monthly volume.
- **Local (C, D):** hardware / amortisation months + kW x hours/month x EUR/kWh + ops hours x EUR/hour, spread over the monthly volume.
- **Break-even vs B:** local fixed EUR/month / B's EUR per reply = replies per month above which local is cheaper.

Reported at the volumes in `volumes_per_month` (1,000 and 50,000).

## 6. Gates and how they are evaluated

Thresholds: [gates.md](gates.md) (rationale) and `config/gates.yaml` (machine-readable; a test
keeps both in sync). `report.evaluate_gates` applies them as follows.

- `>=` gates pass at the threshold (rubric 4.0 passes); `<` gates are strict (p95 of exactly 20 s fails).
- Gates apply to A-E only; FAKE is reported but never gated.
- Quality gates use human scores when a variant has them, otherwise the LLM judge. The source is shown next to every value.
- Edit effort uses human scores only.
- No data means **not measured**, never pass or fail.
- Any gate based on fewer than 30 items carries a "small sample: treat as indicative" note.
- Redaction recall is gated per direct-identifier type (`PERSON, LOCATION, EMAIL, PHONE, IBAN, NATIONAL_ID`). Order and customer ids are reported but not gated.
- The pairwise gate shows "not measured" until pairwise export exists.

## 7. Safeguards

- Real complaints cannot reach a cloud variant or a cloud judge (`runner.check_allowed`, `judge_variant`). Tested for every combination of `cloud` and `allow_real_data`.
- Cloud variants ship with `allow_real_data: false`; a test fails if that changes.
- The run log stores the prompt as sent. For B and E that is the pseudonymised text.
- `data/real/` and `results/` are git-ignored.
- The HTML/JSON report embeds complaint and draft text for synthetic complaints only. Real text needs `--include-real-text`, and the page then says it must not be shared.
- The test suite blocks all network access, so no test can call Azure or a local server by accident.

## 8. Storage (`results/experiment.sqlite`)

| Table | Key | Content |
|---|---|---|
| `runs` | (variant, complaint_id) | metadata, prompt sent, raw output, draft, note, parse flag, latency, tokens, error |
| `judgements` | (variant, complaint_id, judge_model) | scores JSON, hallucination flag |
| `human_scores` | (variant, complaint_id, rater) | scores JSON, hallucination flag, edit minutes |
| `redaction_eval` | (ts, pii_type) | n, leaked, missed values, NER on/off, dataset |

## 9. Outputs

| Output | Command | Content |
|---|---|---|
| Markdown | `cexp report --out results/report.md` | gates, operations, cost, quality with CI and splits, privacy, effort |
| Interactive HTML | `cexp report --html` (writes `results/report.html`) | everything above, plus filters (variant, language, type, difficulty, data source, score source), a latency strip chart, per-criterion quality chart, breakdown heat table, redaction chart, effort chart and a side-by-side draft explorer. Single file, no network, light and dark theme. |
| JSON | `cexp report --json results/report.json` | the raw report dict, for further analysis |

Both reports label variants by **model name**, not by letter: the model id the server
reported, else the configured model, else the variant's label. When two variants run the
same model (A/B, C/E), the hosting and PII handling from the label is appended, e.g.
`gpt-x (Azure private endpoint, pseudonymised)`. The letters stay in the database and JSON.

The HTML recomputes charts and tables in the browser from the embedded per-draft records
when you change a filter. The gates table always reflects the full data set.

## 10. Reproducibility

- Seeds: generation `--seed` (default 1), blind export 42, bootstrap 0 (Python) / 7 (browser).
- Every row stores the server-reported model id and a UTC timestamp.
- Prompts and policy are versioned files; freeze them at gate 3 and record the commit.
- Generation settings come from `variants.yaml` defaults and are identical for A-E.
- Local models are not bit-reproducible at temperature 0.2. Treat single drafts as samples, not fixed outputs.

## 11. Threats to validity

- **Sample size.** 6 complaints per variant is a smoke test. Rubric means need roughly 60+ items per variant before a 0.2-point difference means anything; check the CI column.
- **Cold start.** C's first call in the current data took 41.9 s against 15.7-28.7 s for the rest: model loading. With n = 6 that one call sets p95. Rerun with `--warmup` before reading the latency gate.
- **Test hardware.** C and D ran through Ollama on one machine, one request at a time. Production latency depends on the GPU and on concurrency, which this harness does not load-test.
- **Judge bias.** An LLM judge tends to favour its own model family and longer answers. Use another family than A, and check it against human scores on an overlap sample before trusting judge-only results.
- **Synthetic vs real.** Generated complaints are cleaner than real ones. The real set (local variants only) is the check.
- **Blinding.** Normalisation removes markdown, but sign-off style and length can still give a model away.
- **Redaction recall** is measured on the annotated sample only. Annotation misses are invisible to the metric.
- **Address patterns** need a recognisable street word (`-straat`, `-laan`, `rue`, `Street`, ...)
  or a postcode. A bare `Damrak 1` with no postcode is missed. A capitalised word plus a
  4-digit number that is not a year (`1500 Brussels`) can be over-redacted; years are only
  treated as postcodes after a comma, a colon or at a line start.
- **Placeholders.** If a model rewrites a placeholder (`<PERSON 1>`, `PERSON_1`), restore leaves it in the draft. The raw output is kept so this can be counted.

## 12. Status on 2026-11-06

- Runs: C (`mistral-small3.2`) and D (`qwen3:8b`) on the 6 sample complaints, no errors or parse failures.
- Latency: C p50 20.0 s / p95 41.9 s; D p50 11.4 s / p95 14.0 s. C's p95 includes a cold start, but its slowest warm draft (28.7 s) also misses the < 20 s gate, so C fails on this machine either way. D passes.
- Redaction (NER on): 100% for every annotated type, including addresses. Until 2026-11-06 only the city was redacted and "Kerkstraat 12, 9000 Gent" leaked street and postcode (LOCATION 50%); address patterns fixed this. Regex-only mode still misses names and lone cities, so B must run with NER. The sample has only 2 addresses: measure again on the generated set.
- Not yet run: A, B, E (no Azure deployment), LLM judge, human scoring; costs and hours not filled in.

## 13. Not built yet

- Pairwise preference export/import (gate: local wins or ties >= 40% vs B).
- Edit distance against `sent_reply` for the real set.
- Azure-side check of abuse monitoring and retention settings (manual, before any cloud call).
- Placeholder-integrity metric (placeholders lost or mangled by B/E).
