# Gates and decision thresholds

Fixed before any results are seen (from the experiment plan). Edit here, not after the fact.
The machine-readable copy is `config/gates.yaml`; `tests/test_config.py` fails if the two disagree.
How `cexp report` evaluates each gate: [experiment.md, section 7](experiment.md#7-gates-and-how-they-are-evaluated).

## Variant gates

Applied to every variant (FAKE excluded), on all its scored drafts.

| Dimension | Metric | Threshold |
|---|---|---|
| Quality | Rubric average (1-5), blind, 2-3 agents | >= 4.0 |
| Quality | Pairwise: local wins or ties vs pseudonymised API | >= 40% |
| Quality | Hallucination rate | < 2% |
| Quality | Edit effort, median | < 2 min |
| Privacy | Redaction recall, direct identifiers | >= 98% |
| Privacy | Quality loss from preprocessing (raw minus pseudonymised, raw minus abstract) | < 0.2 points |
| Operations | p95 time to draft, tiers 3 and 4 | < 20 s |
| Effort / Cost | hours logged; EUR per sendable reply at 1,000 and 50,000/month | recorded, no threshold |

## Task complexity ladder

Every model runs the same six tiers. A model passes a tier when every metric of that tier
passes. Its **failure point** is the first tier it fails. The first metric of each tier is
its main metric, drawn in the capability-frontier chart.

| Tier | Metric | Threshold |
|---|---|---|
| 1. Classify | label accuracy | >= 95% |
| 2. Extract | field-level F1 | >= 95% |
| 2. Extract | valid JSON rate | >= 99% |
| 3. Simple draft | rubric average | >= 4.0 |
| 4. Complex draft | rubric average | >= 4.0 |
| 4. Complex draft | policy compliance rate (policy score >= 4) | >= 98% |
| 5. Policy reasoning with tools | decision accuracy | >= 90% |
| 5. Policy reasoning with tools | valid tool-call rate | >= 98% |
| 6. Long context | factual consistency rate | >= 95% |
| 6. Long context | missed key facts | < 5% |

25 complaints per tier (150 in total). A tier result on fewer items is marked
"small sample: treat as indicative".

A model fails a tier in one of two ways, and the report shows both:

- **Quality failure:** it misses a threshold above.
- **Economic failure** (local models only): it passes, but at a given monthly volume its cost
  per reply exceeds that of the cheapest API variant (pseudonymise or abstract route) that
  passes the same tier. Agent edit time is added on both sides once human edit times exist.

## Gate checklist

DPO approval and Azure/Foundry settings (before real data or any cloud call) -> data set
reviewed, including every gold answer -> model versions locked -> prompts frozen (before blind
scoring) -> readout.

Open questions to settle first: DPO conditions for real data; Azure subscription, region and
deployment; Claude hosting route (Foundry is Global or US Data Zone only); GPU availability;
which agents give ~6 h each; monthly volumes for the cost comparison.
