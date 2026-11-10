# Experiment specification: local vs API LLMs for complaint handling

This document describes the experiment end to end: the questions, the lineup, the task ladder,
the data, how every number in the report is computed, and what can make those numbers wrong.
Plan of record: <https://claude.ai/artifact/NuoBbtVYJRhM49BrSuaxwv>. Decision thresholds:
[gates.md](gates.md) and `config/gates.yaml`. Models, hardware and serving: [models.md](models.md).
How the code is tested: [testing.md](testing.md).

## 1. Questions

The experiment builds the same complaint assistant on locally hosted open-weight models and on
API models, and answers five questions:

| Question | Answered by |
|---|---|
| **Quality:** where do local models start to perform worse, and where do they break? | the capability frontier: pass/fail per model on six tiers of increasing difficulty |
| **Effort:** how many person-days does each setup take to run and keep running? | `hours_log.csv` |
| **Cost:** what does a sendable reply cost per option, at low and high volume? | cost model with measured tokens, throughput and edit time |
| **Privacy:** what does each route protect, and is pseudonymisation or abstraction good enough? | redaction recall, residual leakage per run, quality loss per route |
| **Operations:** latency, reliability, capacity | p50/p95, failure rate, replies per hour |

The output is a decision framework: for a data class and a task tier, which route is allowed,
which models pass, and which of those is cheapest at the expected volume.

## 2. Lineup: models x routes x reasoning

A **variant** is one model, reached through one data-handling route, with reasoning on or off.
Its key is `<route>/<model>[+think]`, for example `local/qwen3.6-27b` or
`pseudonymise/haiku-5.5+think`. `cexp variants` lists all of them.

### 2.1 Models

Five size classes. Full table with parameters and hardware: [models.md](models.md).

| Size class | Models | Hardware |
|---|---|---|
| Cloud | Claude Sonnet 5.5, Claude Haiku 5.5, newest mid-tier GPT on Azure | API |
| XL local | GLM-5.3, DeepSeek-V4.1-Flash-1017, Kimi K2.6, Qwen3.8-Flash-Next | server with 512 GB RAM + GPUs |
| Large local | Mistral Small 4 | 80 GB+ GPU |
| Medium local | Qwen3.6-27B, Gemma 4 26B A4B, gpt-oss-20b | one 24-48 GB GPU |
| Small local | Gemma 4 12B, Qwen3 8B | laptop GPU or CPU |

The XL class tests the best open-weight model of the Kimi, GLM, DeepSeek and Qwen families
that fits on the big server. Their flagships (up to 2.8T parameters) do not fit;
[models.md](models.md#xl-class-the-128-gb-server) lists what was chosen instead, and why.

### 2.2 Routes

| Route | Plan route | What leaves your infrastructure | Real data allowed | Used by |
|---|---|---|---|---|
| `local` | R0 fully local | nothing | yes (after DPO sign-off) | every local model |
| `own_tenant` | R1 own-tenant GPU | pseudonymised text to a VM in your subscription | no | optional, any local model |
| `abstract` | R2 abstract + API | a case summary written locally from pseudonymised text | no | every cloud model |
| `pseudonymise` | R3 pseudonymise + API | text with identifiers replaced by placeholders | no | every cloud model |
| `raw` | R4/R5 raw + API | the full text; synthetic data only | no | every cloud model (quality ceiling) |
| `hybrid` | hybrid | depends on routing | only if the routed-to variant allows it | `hybrid/default` |

- **Pseudonymise:** `Pseudonymiser.redact` replaces identifiers with consistent placeholders
  (`<PERSON_1>`, `<ORDER_1>`). Placeholders in the answer are mapped back to the real values.
- **Abstract:** first pseudonymise the complaint, then the abstractor (a local model,
  `local/qwen3.6-27b` by default) writes a neutral case summary: issue, facts, request, tone,
  no identifiers, placeholders kept. Only the summary and the (pseudonymised) customer facts go
  to the API. So abstract is pseudonymise plus summarisation. Expect it to cost quality on
  extraction and long-context tiers: that cost is what the route measures.
- **Hybrid:** tiers 5 and 6 always go to `hard` (an API variant with preprocessing). For tiers
  3 and 4 a local router model answers "easy" or "hard". Tiers 1 and 2 stay on `easy`. Set
  `easy` and `hard` in `config/variants.yaml` after the first results (plan: best local model
  plus the cheapest API model that passes the hard tiers).

### 2.3 Held constant

Within a tier every variant gets the same system prompt (`prompts/system.md` with
`prompts/policy.md` inlined, plus the tier's task prompt from `prompts/tasks/`), the same user
message layout, the same output contract, the same token budget (`max_tokens_by_tier`, plus
6,000 tokens when reasoning is on) and JSON output. Enforced by
`tests/test_config.py::test_shipped_variants_share_generation_settings` and
`tests/test_runner.py::test_all_variants_get_the_identical_system_prompt`.

Allowed per-variant differences: endpoint, model, the reasoning switch, the token parameter
name (`max_tokens` vs `max_completion_tokens`) and temperature. The Claude API and GPT
reasoning models reject temperature, so they run at the provider default; all other variants
send 0.2.

## 3. Task complexity ladder

Every model runs the same six tiers. The **failure point** of a model is the first tier where
it misses a threshold. The result is a capability frontier per size class, not one average.

| Tier | Task | Example | Output contract | Scored by |
|---|---|---|---|---|
| 1. Classify | type, language, urgency, sentiment | "Is this delivery, billing or damage?" | `{"type", "language", "urgency", "sentiment"}` | automatic, vs `gold` labels |
| 2. Extract | order number, product, requested remedy, deadline | "ORD-6120458, coffee machine, replacement, 2026-11-20" | `{"order_number", "product", "requested_remedy", "deadline"}` | automatic, vs `gold.fields` |
| 3. Simple draft | reply to one issue with a clear policy match | late delivery inside the refund window | `{"draft_reply", "agent_note"}` | rubric (judge + humans) |
| 4. Complex draft | angry, multi-issue or mixed-language complaints | damaged item, rude installer and billing error | `{"draft_reply", "agent_note"}` | rubric + policy compliance |
| 5. Policy reasoning with tools | look up order and contacts, decide the remedy within policy | voucher or refund? | tool calls, then `{"final": {"decision", "justification", "agent_note"}}` | automatic, vs `gold.decision`, plus tool-call validity |
| 6. Long context | resolve a long email thread plus retrieved policy pages | 12 messages with a contradiction about what was promised | `{"draft_reply", "agent_note"}` | judge: consistency + key facts, plus rubric |

Run every model with reasoning off and on: extra reasoning often moves a small model one tier
up, at a latency cost. The two variants show both sides.

**Tier 5 tools.** `get_order(order_number)` and `get_contact_history(customer_id)` answer from
the complaint's `tool_data`, so runs are deterministic. Tools are called through a plain JSON
protocol (`{"tool": ..., "args": {...}}` per turn, up to 6 turns), not each server's native
tool-calling API. That keeps the instruction identical for every model and serving stack, so
the score measures the model, not a parser. On cloud routes every tool result is pseudonymised
with the complaint's mapping, and placeholders in tool arguments are restored before lookup.

## 4. Data

### 4.1 Sets

| Set | File | Source | May go to | Status |
|---|---|---|---|---|
| Sample | `data/synthetic/sample.jsonl` | hand-written, 13 complaints covering all six tiers and nl/fr/en | every variant | in use (smoke test) |
| Generated | `data/synthetic/generated.jsonl` | `cexp generate`, 25 per tier = 150 | every variant after agent review | not generated yet |
| Real | `data/real/*.jsonl` | ~50 historical complaints with sent replies | local variants only | not loaded; git-ignored |

Generated rows carry `"status": "needs_review"`. A support agent checks realism, and every
gold answer, before they count. Around 30 complaints also get a `gold_reply` from an
experienced agent, to anchor scoring and calibrate the judge.

### 4.2 Record format (JSONL, one complaint per line)

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | unique across all loaded files |
| `source` | yes | `synthetic` or `real`; drives the real-data safeguard |
| `language` | yes | `nl`, `fr` or `en` |
| `tier` | yes | 1-6 |
| `received_on` | tiers 5, 6 | `YYYY-MM-DD` the complaint arrived. The prompt shows it next to the language and the model treats it as today: the policy's time windows count up to it |
| `text` | yes | the complaint as the customer wrote it; for tier 6 the whole thread |
| `type` | no | delivery, damage, billing, refund, staff, repeat |
| `difficulty` | no | simple, angry, rambling, multi_issue, legal, out_of_policy, mixed_language |
| `facts` | no | structured customer facts given to the model |
| `pii` | no | hand annotations `[{"type": "PERSON", "text": "Jan Peeters"}]`, matched by text |
| `gold` | tiers 1, 2, 5, 6 | tier 1: `{type, language, urgency, sentiment}`; tier 2: `{fields: {order_number, product, requested_remedy, deadline}}` (null when absent); tier 5: `{decision}`; tier 6: `{key_facts: [...]}` |
| `tool_data` | tier 5 | `{"orders": {<order number>: {...}}, "contacts": {<customer id>: [...]}}` |
| `context_docs` | tier 6 | retrieved policy pages |
| `gold_reply` | no | an experienced agent's reply |
| `sent_reply` | no | real set only: the reply that was actually sent |

`load_dataset` refuses duplicate ids, unknown sources, languages or tiers, missing fields,
malformed `pii` and invalid gold for the tier, and reports `file:line`. `cexp validate` also
checks that every PII annotation occurs verbatim in the text.

### 4.3 Generation

`cexp generate --tiers 1 2 3 4 5 6 --n 25` writes 25 complaints per tier from
`data/synthetic/scenarios.yaml`. The harness picks language, type, difficulty and 3-5 PII
patterns, and where it can, the answer key:

- Tier 1: the harness picks urgency and sentiment; the model writes a matching complaint.
- Tier 2: the model plants the fields and returns them; the agent checks them.
- Every row: the harness picks `received_on`, a working day in `received_between`, and tells
  the generator that dates must fit it.
- Tier 5: the harness picks a case from `scenarios.yaml` (situation, policy rule, correct
  decision) and quotes the rule; the model writes the complaint and the tool data that make that
  decision correct. **Update the cases and rule numbers whenever `prompts/policy.md` changes.**
- Tier 6: the model writes an 8-14 message thread with a planted contradiction, resolved under
  the rules listed for that contradiction in `scenarios.yaml`, and lists the key facts. Those
  rules become the retrieved pages, plus one unrelated rule.

Rows whose gold fails validation are dropped and logged. Generation is seeded (`--seed`).

## 5. Procedure

| Phase | Step | Command |
|---|---|---|
| 0. Setup | install, `.env`, serve local models | `uv sync --extra dev --extra redaction`; [models.md](models.md) |
| 1. Data | validate, generate, agent review | `cexp validate`, `cexp generate` |
| Gate | DPO approval, Azure/Foundry settings checked, model versions locked | manual |
| 2. Privacy | redaction recall, no model calls | `cexp redaction-eval --data ...` |
| 3. Runs | every variant on the synthetic set; local variants also on the real set | `cexp run --variants 'local/*' --warmup` and so on |
| 3b. Capacity | replies/hour per local model | `cexp throughput --variants 'local/glm-5.3'` |
| Gate | prompts frozen | manual: no edits to `prompts/` after this |
| 4. Scoring | LLM judges, blind human scoring by 2-3 agents | `cexp judge --judge JUDGE`, `--judge JUDGE_GPT`, `cexp export-blind`, `cexp import-scores` |
| 5. Readout | report against the gates | `cexp report --html --out results/report.md` |

Effort is a result too: every person logs hours in `hours_log.csv`
(`date,person,phase,task_category,variant,hours,notes`).

### 5.1 What happens to one complaint (`runner.process`)

1. **Safeguard.** `check_allowed` raises for a real complaint if the variant (for hybrid: the
   routed-to variant) is cloud or not cleared. Blocked complaints are counted, never sent, never stored.
2. **Route.** Hybrid picks the variant that answers.
3. **Preprocess.** Pseudonymise (pseudonymise, abstract, own_tenant); then for abstract, the local summary.
4. **Call.** One request, or the tool loop for tier 5.
5. **Parse** against the tier's output contract (strict JSON, then the first `{...}` block).
   Anything else is a parse failure: stored, counted as a failure and a failed task, not retried.
6. **Restore** placeholders in the answer; score tiers 1, 2 and 5 against gold.
7. **Count leakage.** For cloud routes: how many annotated PII values are still visible in
   everything that was sent (`pii_leaked` of `pii_annotated`).
8. **Log** one row per (variant, complaint) in `results/experiment.sqlite`, including the
   prompt exactly as sent and the raw output.

Runs are resumable: successful rows are skipped on a rerun, rows that ended in an error
(timeout, rate limit, server down) are retried. `--tiers` limits a run to some tiers.

## 6. Metrics: exact definitions

All metrics come from `report.build()`. The Markdown and HTML reports are two views of the same dict.

### 6.1 Tier metrics (capability frontier)

Computed per (variant, tier) over answered runs; errors (timeouts) are excluded and show in the
failure rate instead. A malformed answer counts as a failed task.

| Metric | Definition |
|---|---|
| Label accuracy (T1) | Mean over answers of (labels equal to gold / 4), after case and accent normalisation. Malformed = 0. |
| Field F1 (T2) | Micro F1 over all fields: a field is a true positive when predicted and gold are both non-null and match (normalised; product also matches when one contains the other), a false positive when predicted but wrong or not in gold, a false negative when in gold but missing or wrong. |
| Valid JSON rate (T2) | Answers that parse and have all four keys / answers. |
| Rubric average (T3, T4) | Mean per-draft overall score (mean of criteria, averaged over raters or judges). Human scores when any exist for that (variant, tier), else the judge. Malformed drafts count as the lowest score (1). |
| Policy compliance rate (T4) | Drafts with a policy-compliance score >= 4 / drafts. Malformed = not compliant. |
| Decision accuracy (T5) | Final decisions equal to gold / runs. No decision within 6 turns = wrong. |
| Valid tool-call rate (T5) | Calls to a known tool with exactly its required string arguments / all tool calls. |
| Factual consistency rate (T6) | Mean of the judge's `consistent` flag (no contradiction with the thread or with written promises). Malformed = 0. |
| Missed key facts (T6) | Key facts the judge marks as missed or wrong / key facts, over judged drafts. |

The **main metric** of a tier is its first one. The frontier chart draws main metric /
threshold per tier, so 100% is the threshold for every tier.

**Economic failure** (local variants, per volume): the tier passes, but local cost per reply at
that volume, using the tier's own token counts, exceeds the cheapest API variant
(pseudonymise or abstract) that passes the same tier. Once human edit times exist, edit time x
agent rate is added to both sides.

### 6.2 Operations

| Metric | Definition |
|---|---|
| Time to draft | Preprocessing (pseudonymisation wall time, the abstractor's or router's model time) plus model time (one call, or the sum over the tool loop). |
| p50 / p95 (gated) | Nearest-rank percentile over tiers 3 and 4 (`operations.latency_tiers`), parse failures included, errors excluded. `p95_all_s` covers every tier. |
| Failure rate | (errors + parse failures) / all stored rows for that variant. |
| Tokens | Server-reported prompt/completion tokens, averaged over runs without an error; summed over turns for tier 5. Also kept per tier. |
| Model id | The model string the server reports, stored per row to catch silent model swaps. |
| Replies per hour | `cexp throughput`: completed requests / wall clock x 3600 at each concurrency level. The report uses the highest value whose p95 still passes the latency gate. |

`cexp run --warmup` sends one untimed dummy request to every model a variant uses, so model
load time does not land in the first latency.

### 6.3 Quality (rubric)

Rubric (`config/rubric.yaml`), shared by human raters and the LLM judges, integers 1-5:
factual correctness, policy compliance, tone and empathy, language quality, completeness.
Plus a hallucination flag (0/1). Variant-level metrics (all scored drafts of a variant):

| Metric | Definition |
|---|---|
| Per-criterion mean | Mean over all score rows. |
| Overall | Mean of the per-criterion means. |
| 95% CI | Percentile bootstrap of the mean of per-row overall scores; 2,000 resamples, seed 0. |
| Hallucination rate | Rows flagged 1 / rows with a flag. |
| Edit effort | Median `edit_minutes` from human sheets. |

Two judge families are configured: Claude Opus 5.5 (`JUDGE`) and GPT, through an Azure
deployment (`JUDGE_GPT`) or the OpenAI API (`JUDGE_OPENAI`, key in `OPENAI_API_KEY`). Judges
favour their own family, so run both and compare them with the human overlap sample
(`cexp export-blind --drafts 60`: a sheet of 60 drafts spread evenly over the variants).
The report's **judge agreement** table compares every pair of score sources (each judge, and the
humans averaged per draft) on the drafts both scored: mean of each, signed and absolute
difference, Pearson r, how often the hallucination flags match, and the signed difference split
by the drafting model's family. A judge that favours its own family shows a larger gap on that
family's drafts. A judge
is refused when its model equals the variant's model (for hybrid: either routed-to model), and
never receives real complaints when it runs in the cloud. For tier 6 the judge also gets the
numbered key facts and returns `consistent` and `missed_facts`.

Blind scoring: `export-blind` takes drafts of tiers 3, 4 and 6 only, shuffles them with a fixed
seed (42), strips formatting that could reveal the model and writes the variant key to a
separate file. `import-scores` validates the whole sheet before writing anything.

### 6.4 Privacy

| Metric | Definition |
|---|---|
| Redaction recall, per PII type | Share of hand-annotated values that do not survive `Pseudonymiser.redact`. A value leaks when the full value, or any of its tokens of 4+ characters, still appears (case-insensitive). Partial leaks count. |
| Residual leakage, per variant | For cloud routes: annotated values still visible in everything sent (prompt, summary, tool results) / annotated values. Shows what abstraction adds over pseudonymisation. |
| Quality loss from preprocessing | Overall(raw) - Overall(pseudonymise) and Overall(raw) - Overall(abstract), same model, same score source. |

Addresses are regex-based in both NER modes: Dutch/Flemish, French and English street forms,
BE/FR/NL postcode + city and UK postcodes. Adjacent pieces merge into one `<ADDRESS_n>`, and a
city right after a street (", Bristol") is absorbed into it. Without the `redaction` extra the
pseudonymiser runs regex-only and the report marks the recall gate "regex only".

### 6.5 Cost per sendable reply

Inputs in `config/costs.yaml`; `null` stays `n/a` rather than a made-up number.

- **Pay per token** (pseudonymise, abstract, raw): (mean prompt tokens x input price + mean completion tokens x output price) / 1e6, in EUR (USD prices x `usd_to_eur`).
- **Local box** (local route, the abstractor's box, the hybrid's easy box): hardware / amortisation months + hosting + ops hours x rate, per month.
- **Capacity cliff:** servers needed = ceil(volume / (replies per hour x 730 h x `max_utilisation`)). Without a throughput measurement one server is assumed, and the report says so.
- **Hybrid:** the share of replies routed to `hard` x its token cost, plus the easy box.
- **Per sendable reply** = cost per reply + median edit minutes x `agent_eur_per_hour` / 60.
- **Break-even:** local fixed cost per month / the cheapest API variant's cost per reply.

Reported at the volumes in `volumes_per_month` (1,000 and 50,000).

**Cost against volume** (`report.volume_view`, report section "At what volume does local pay
off?"). For every local option and every API route allowed for personal data (pseudonymise,
abstract) it computes the monthly volume from which the local option costs no more than the API:

- Local options: each measured local variant on its hardware profile, and every other hardware
  profile in `costs.yaml` with complete inputs ("hardware only": no model measured on it, so its
  capacity is unknown and one box is assumed to keep up).
- Capacity of one box: replies/hour from `cexp throughput` when measured, otherwise 3,600 / mean
  run latency (one request at a time: a lower bound, since a GPU server batches requests), x 730 h
  x `max_utilisation`.
- Break-even: the lowest volume where boxes x fixed cost <= tokens x price (+ the abstractor's box
  for the abstract route). If one full box still costs more per reply than the API, the answer is
  "never": more boxes only repeat that cost. If the API route already pays for a box as big
  (the abstract route's summariser), the answer is "always".
- Quality is not part of the number. The HTML page filters both sides to setups that pass every
  tier up to the one you pick, and draws cost per reply against volume (log-log) with the
  break-even points and one minute of agent time for scale. Agent edit time is left out: it is the
  same on both sides at equal quality.

## 7. Gates and how they are evaluated

Thresholds: [gates.md](gates.md) and `config/gates.yaml` (a test keeps both in sync).

- `>=` gates pass at the threshold (rubric 4.0 passes); `<` gates are strict (p95 of exactly 20 s fails).
- Variant gates apply to every variant except `ungated_variants` (FAKE).
- Quality gates use human scores when a variant has them, otherwise the LLM judge. The source is shown.
- No data means **not measured**, never pass or fail.
- Fewer than 30 items for a variant gate, or fewer than 25 for a tier, carries a "small sample: treat as indicative" note.
- Redaction recall is gated per direct-identifier type (`PERSON, LOCATION, EMAIL, PHONE, IBAN, NATIONAL_ID`).
- The pairwise gate shows "not measured" until pairwise export exists.

## 8. Safeguards

- Real complaints cannot reach a cloud variant, a cloud judge or a hybrid's cloud target (`runner.check_allowed`).
- Cloud routes ship with `allow_real_data: false`; a test fails if that changes.
- The run log stores what was sent: pseudonymised or abstracted text for cloud routes.
- `data/real/` and `results/` are git-ignored.
- The HTML/JSON report embeds complaint and answer text for synthetic complaints only, unless `--include-real-text`.
- The test suite blocks all network access.

## 9. Storage (`results/experiment.sqlite`)

| Table | Key | Content |
|---|---|---|
| `runs` | (variant, complaint_id) | tier, route, model, size class, reasoning, prompt sent, preprocessing output, raw and parsed answer, automatic scores, tool calls, leakage, latency, tokens, error |
| `judgements` | (variant, complaint_id, judge_model) | rubric scores, hallucination flag, tier-6 consistency and missed facts |
| `human_scores` | (variant, complaint_id, rater) | rubric scores, hallucination flag, edit minutes |
| `redaction_eval` | (ts, pii_type) | n, leaked, missed values, NER on/off, dataset |
| `throughput` | (ts, variant, concurrency) | requests, wall time, replies per hour, p50/p95 |

Databases from the first version are migrated on open (new columns added, old rows kept).
Old rows have no tier, so they appear in the variant tables but not in the frontier.

## 10. Outputs

| Output | Command | Content |
|---|---|---|
| Markdown | `cexp report --out results/report.md` | frontier, gates, operations, cost, quality with CI and splits, privacy, effort |
| Interactive HTML | `cexp report --html` | the same, plus the frontier chart (one line per variant across tiers, threshold at 100%), filters (variant, tier, route, language, type, difficulty, source, score source), latency strip chart, quality chart, breakdown heat table, cost table and a side-by-side answer explorer with gold answers |
| JSON | `cexp report --json results/report.json` | the raw report dict |

Reports name variants by model and route, e.g. "Claude Haiku 5.5 (pseudonymised)" or
"Qwen3 8B (local, thinking)". Colours follow the size class.

## 11. Reproducibility

- Seeds: generation `--seed` (default 1), blind export 42, bootstrap 0 (Python) / 7 (browser).
- Every row stores the server-reported model id and a UTC timestamp.
- Lock model versions at gate 1 ([models.md](models.md#locking-versions)); freeze prompts at gate 3 and record the commit.
- Local models are not bit-reproducible at temperature 0.2. Treat single answers as samples.

## 12. Threats to validity

- **Sample size.** 13 sample complaints are a smoke test. 25 per tier gives a coarse pass/fail;
  rubric means need ~60 human-scored items per variant before a 0.5-point difference is visible.
- **Quantisation.** XL models run quantised, Kimi K2.6 at ~2 bits. A failure can be the quant,
  not the model. Record the quant and, if a tier is close, rerun at a higher precision.
- **Test hardware.** Latency and throughput depend on GPU, RAM speed and how many MoE experts
  sit in RAM. Record the hardware. One machine serving one model is not a production cluster.
- **Temperature.** Claude and GPT reasoning models run at their default temperature, the others at 0.2.
- **JSON tool protocol.** Tier 5 measures tool use through a text protocol. Native tool-calling
  APIs can do better or worse per model.
- **Judge bias.** A judge favours its own family and longer answers. Use both judges, and check
  them against human scores on an overlap sample before trusting judge-only results.
- **Synthetic vs real.** Generated complaints are cleaner than real ones. The real set (local
  variants only) is the check. Generated gold answers are only as good as the agent review.
- **Blinding.** Normalisation removes markdown, but sign-off style and length can still give a model away.
- **Redaction recall** is measured on annotated complaints only; annotation misses are invisible.
- **Address patterns** need a street word or a postcode. A bare `Damrak 1` is missed.
- **Placeholders.** A model that rewrites a placeholder (`<PERSON 1>`) leaves it in the draft. The raw output is kept.
- **Policy.** `prompts/policy.md` is written for this experiment (a fictional Belgian retailer),
  not a real company's policy. Tiers 4-6 measure compliance with it. Swap in a real policy, and
  re-check every gold answer, if the result must hold for one company.
- **Hardware prices.** The values marked illustrative in `config/costs.yaml` are estimates; the
  break-even volumes move with them, and ops hours dominate the fixed cost.

## 13. Status on 2026-11-09

- Harness v0.3: lineup of 20 models/routes x reasoning modes (40 variants), task ladder, abstract
  and hybrid routes, tool loop, throughput, cost per sendable reply, frontier report.
- Data: 13 hand-written sample complaints across all tiers. Generated set not produced yet.
- Redaction (NER on): 100% recall for every annotated type on the sample, after fixing a city
  leak after English street addresses ("14 Elm Grove, Bristol").
- Earlier runs (2026-11-06, v0.2): `mistral-small3.2` and `qwen3:8b` on the 6 drafting samples.
  Those rows have no tier and are kept only for reference.
- Not yet run: anything on the new lineup.

### Run 1 (2026-11-09) and what changed after it (2026-11-10)

Run 1: 15 variants (Claude Sonnet 5.5 and Haiku 5.5 on the raw, pseudonymise and abstract routes,
thinking on and off; Qwen3 8B and Mistral Small 3.2 on a laptop) on the 13 sample complaints,
judged by Claude Opus 5.5 only. Its database is kept for reference; rerun everything for run 2,
because prompts, policy and gold changed.

| Problem found in run 1 | Change |
|---|---|
| spaCy tagged JSON field names (`delivered_on`, `shipping_charged_eur`) as PERSON, so the pseudonymised tool result and facts block lost what the value meant (sample-502, sample-006) | NER hits inside JSON object keys are ignored; regex identifiers still apply (`pseudonymise._json_key_ranges`) |
| No "today" in the prompt, so "more than 3 working days late" could not be decided (sample-501) | `received_on` per complaint, shown in the prompt header; system rule: count windows up to that date; required for tiers 5 and 6 |
| Tier 6 gold demanded a refund that the placeholder policy forbade, and the task prompt said both "promises count" and "policy decides" (14 of 15 variants failed) | New policy with rule 5 (the registration date of a return counts) and rule 10 (written commitments are honoured); task prompt resolves contradictions with the policy; sample-601 gets the rules that decide it as its retrieved pages |
| Placeholder policy | `prompts/policy.md` v2026-11-10: 12 rules plus definitions (working days, "unused", latest request counts); tier-5 cases renumbered, two added (warranty escalation, written commitment) |
| Generated tier-6 threads got three random policy lines as retrieved pages | Each contradiction lists the rules that resolve it; they become the pages, plus one distractor |
| One judge, same family as every cloud candidate | `JUDGE_OPENAI` (GPT through the OpenAI API) next to `JUDGE_GPT` (Azure); judge-agreement table; `export-blind --drafts` for a small balanced human sample |
| The cost table answered "what at 1,000 and 50,000 a month", not "from what volume" | Cost-against-volume section with break-even volumes; illustrative hardware prices filled in |

## 14. Not built yet

- Pairwise preference export/import (gate: local wins or ties >= 40% vs pseudonymised API).
- Edit distance against `sent_reply` for the real set.
- Placeholder-integrity metric (placeholders lost or mangled).
- Azure/Foundry check of abuse monitoring and retention settings (manual, before any cloud call).
