# Gates and decision thresholds

Fixed before any results are seen (from the experiment plan). Edit here, not after the fact.
The machine-readable copy is `config/gates.yaml`; `tests/test_config.py` fails if the two disagree.
How `cexp report` evaluates each gate: [experiment.md, section 6](experiment.md#6-gates-and-how-they-are-evaluated).

| Dimension | Metric | Threshold |
|---|---|---|
| Quality | Rubric average (1-5), blind, 2-3 agents | >= 4.0 |
| Quality | Pairwise: local wins or ties vs B | >= 40% |
| Quality | Hallucination rate | < 2% |
| Quality | Edit effort, median | < 2 min |
| Privacy | Redaction recall, direct identifiers | >= 98% |
| Privacy | Quality loss from redaction (B minus A) | < 0.2 points |
| Operations | p95 time to draft | < 20 s |
| Effort / Cost | hours logged; EUR per 1,000 replies at 1,000 and 50,000/month | recorded, no threshold |

Gate checklist: DPO approval and Azure settings (before real data / any cloud call) ->
data set reviewed -> prompts frozen (before blind scoring) -> readout.

Open questions to settle first: DPO conditions for real data; Azure subscription/region/deployment;
GPU availability; which agents give ~6 h each; monthly volumes for the cost comparison.
