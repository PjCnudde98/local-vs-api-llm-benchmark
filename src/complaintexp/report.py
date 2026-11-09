"""Aggregate results: operations, cost per 1,000 replies, quality, privacy, gates, effort.

`build()` turns the SQLite store into one plain dict. `render_markdown()` and
`html.render_html()` are two views of that same dict, so the Markdown and HTML
reports can never disagree.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ROOT, __version__
from .config import load_costs, load_gates, load_rubric
from .schemas import Complaint

DEFAULT_HOURS_LOG = ROOT / "hours_log.csv"
SPLIT_DIMENSIONS = ("language", "type", "difficulty")


# -- small statistics helpers -------------------------------------------------
def _pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (no interpolation): p95 of 20 values is the 19th."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def _mean(values: list[float]) -> float | None:
    return st.fmean(values) if values else None


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of the mean. Deterministic (fixed seed). None if n < 2."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(st.fmean(rng.choices(values, k=n)) for _ in range(n_resamples))
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))]
    return lo, hi


# -- operations ---------------------------------------------------------------
def operations(con: sqlite3.Connection) -> list[dict]:
    out = []
    for (v,) in con.execute("SELECT DISTINCT variant FROM runs ORDER BY variant").fetchall():
        rows = con.execute("SELECT * FROM runs WHERE variant=?", (v,)).fetchall()
        ok = [r for r in rows if r["error"] is None]
        lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
        errors = [r for r in rows if r["error"] is not None]
        parse_fail = [r for r in ok if not r["parse_ok"]]
        out.append(
            dict(
                variant=v, n=len(rows), n_ok=len(ok) - len(parse_fail),
                errors=len(errors), parse_failures=len(parse_fail),
                failure_rate=(len(errors) + len(parse_fail)) / len(rows),
                p50_s=_pct(lat, 0.5), p95_s=_pct(lat, 0.95),
                mean_s=_mean(lat), max_s=max(lat) if lat else None,
                mean_prompt_tokens=st.fmean([r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [0]),
                mean_completion_tokens=st.fmean([r["completion_tokens"] for r in ok if r["completion_tokens"]] or [0]),
                model_ids=sorted({r["model_id"] for r in rows if r["model_id"]}),
            )
        )
    return out


# -- cost ---------------------------------------------------------------------
def cost_per_1000(ops: list[dict], costs: dict | None = None) -> list[dict]:
    """Cloud: tokens x list price (or VM hours). Local: amortised hardware + power + ops hours."""
    costs = costs if costs is not None else load_costs()
    volumes = costs["volumes_per_month"]
    rows = []
    for o in ops:
        v = o["variant"]
        row: dict[str, Any] = {"variant": v, **{f"eur_per_1000@{vol}": None for vol in volumes}}
        if v in (costs.get("local") or {}):
            c = costs["local"][v]
            if c.get("hardware_eur") is not None:
                fixed = (
                    c["hardware_eur"] / c["amortisation_months"]
                    + c["power_watts"] / 1000 * c["hours_per_month"] * c["eur_per_kwh"]
                    + c["ops_hours_per_month"] * c["eur_per_ops_hour"]
                )
                row["fixed_eur_per_month"] = fixed
                for vol in volumes:
                    row[f"eur_per_1000@{vol}"] = fixed / vol * 1000
        elif v in (costs.get("cloud") or {}):
            c = costs["cloud"][v]
            if c.get("input_per_1m") is not None and c.get("output_per_1m") is not None:
                per_reply = (
                    o["mean_prompt_tokens"] * c["input_per_1m"]
                    + o["mean_completion_tokens"] * c["output_per_1m"]
                ) / 1e6
                row["eur_per_reply"] = per_reply
                for vol in volumes:
                    row[f"eur_per_1000@{vol}"] = per_reply * 1000
            elif c.get("vm_eur_per_hour") is not None:
                fixed = c["vm_eur_per_hour"] * c["hours_per_month"]
                row["fixed_eur_per_month"] = fixed
                for vol in volumes:
                    row[f"eur_per_1000@{vol}"] = fixed / vol * 1000
        rows.append(row)
    # Break-even: local fixed cost vs the pay-per-token variant B.
    b = next((r for r in rows if r["variant"] == "B" and r.get("eur_per_reply")), None)
    if b:
        for r in rows:
            if r.get("fixed_eur_per_month") and r["variant"] != "B":
                r["break_even_replies_per_month_vs_B"] = r["fixed_eur_per_month"] / b["eur_per_reply"]
    return rows


# -- quality ------------------------------------------------------------------
SCORE_TABLES = (("human", "human_scores"), ("llm_judge", "judgements"))


def _score_rows(con: sqlite3.Connection, table: str) -> list[dict]:
    """Score rows joined with run metadata (language/type/difficulty)."""
    q = f"""SELECT s.*, r.language, r.type, r.difficulty, r.source
            FROM {table} s LEFT JOIN runs r
              ON r.variant = s.variant AND r.complaint_id = s.complaint_id"""
    out = []
    for r in con.execute(q).fetchall():
        d = dict(r)
        d["scores"] = json.loads(d.pop("scores_json"))
        d["overall"] = st.fmean(d["scores"].values()) if d["scores"] else None
        out.append(d)
    return out


def _summarise(rows: list[dict]) -> dict:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, val in r["scores"].items():
            acc[k].append(val)
    per = {k: st.fmean(v) for k, v in acc.items()}
    overalls = [r["overall"] for r in rows if r["overall"] is not None]
    hall = [r["hallucination"] for r in rows if r.get("hallucination") is not None]
    edits = [r["edit_minutes"] for r in rows if r.get("edit_minutes") is not None]
    ci = bootstrap_ci(overalls)
    return dict(
        n=len(rows),
        overall=st.fmean(per.values()) if per else None,
        ci95=list(ci) if ci else None,
        per_criterion=per,
        hallucination_rate=(sum(hall) / len(hall)) if hall else None,
        edit_minutes_median=st.median(edits) if edits else None,
        n_edit=len(edits),
    )


def quality(con: sqlite3.Connection) -> list[dict]:
    out = []
    for src, table in SCORE_TABLES:
        rows = _score_rows(con, table)
        by_v: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_v[r["variant"]].append(r)
        for v in sorted(by_v):
            out.append(dict(source=src, variant=v, **_summarise(by_v[v])))
    return out


def splits(con: sqlite3.Connection) -> list[dict]:
    """Quality and latency per language / complaint type / difficulty."""
    out = []
    for src, table in SCORE_TABLES:
        rows = _score_rows(con, table)
        for dim in SPLIT_DIMENSIONS:
            groups: dict[tuple, list[float]] = defaultdict(list)
            for r in rows:
                if r["overall"] is not None:
                    groups[(r["variant"], r.get(dim) or "unknown")].append(r["overall"])
            for (v, val), xs in sorted(groups.items()):
                out.append(dict(metric=f"overall_{src}", dimension=dim, value=val, variant=v,
                                n=len(xs), result=st.fmean(xs)))
    run_rows = con.execute("SELECT * FROM runs WHERE error IS NULL AND latency_s IS NOT NULL").fetchall()
    for dim in SPLIT_DIMENSIONS:
        groups = defaultdict(list)
        for r in run_rows:
            groups[(r["variant"], r[dim] or "unknown")].append(r["latency_s"])
        for (v, val), xs in sorted(groups.items()):
            out.append(dict(metric="p50_latency_s", dimension=dim, value=val, variant=v,
                            n=len(xs), result=_pct(xs, 0.5)))
    return out


# -- privacy ------------------------------------------------------------------
def latest_redaction_eval(con: sqlite3.Connection) -> dict | None:
    row = con.execute("SELECT MAX(ts) FROM redaction_eval").fetchone()
    if not row or row[0] is None:
        return None
    ts = row[0]
    rows = con.execute("SELECT * FROM redaction_eval WHERE ts=? ORDER BY pii_type", (ts,)).fetchall()
    return dict(
        ts=ts, dataset=rows[0]["dataset"], has_ner=bool(rows[0]["has_ner"]),
        types=[
            dict(type=r["pii_type"], n=r["n"], leaked=r["leaked"],
                 recall=1 - r["leaked"] / r["n"] if r["n"] else None,
                 missed=json.loads(r["missed_json"]))
            for r in rows
        ],
    )


# -- effort -------------------------------------------------------------------
def effort(hours_csv: Path | None) -> dict:
    """Hours logged per phase / task category / variant (plan: effort is a result)."""
    empty = dict(total_hours=0.0, entries=0, by_phase={}, by_task_category={}, by_variant={}, by_person={})
    if hours_csv is None or not Path(hours_csv).exists():
        return empty
    out = empty | {k: defaultdict(float) for k in ("by_phase", "by_task_category", "by_variant", "by_person")}
    with Path(hours_csv).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                h = float(r.get("hours") or 0)
            except ValueError:
                continue
            out["total_hours"] += h
            out["entries"] += 1
            out["by_phase"][r.get("phase") or "unspecified"] += h
            out["by_task_category"][r.get("task_category") or "unspecified"] += h
            out["by_variant"][r.get("variant") or "shared"] += h
            out["by_person"][r.get("person") or "unspecified"] += h
    return {k: dict(v) if isinstance(v, defaultdict) else v for k, v in out.items()}


# -- gates --------------------------------------------------------------------
def _gate(gate, dimension, variant, value, op, threshold, source=None, n=None, note=""):
    if value is None:
        status = "not_measured"
    elif op == ">=":
        status = "pass" if value >= threshold else "fail"
    else:  # "<": strictly below
        status = "pass" if value < threshold else "fail"
    return dict(gate=gate, dimension=dimension, variant=variant, value=value, op=op,
                threshold=threshold, status=status, source=source, n=n, note=note)


def evaluate_gates(ops: list[dict], qual: list[dict], redaction: dict | None, gates: dict) -> list[dict]:
    gq, gp, go = gates["quality"], gates["privacy"], gates["operations"]
    gated = gates.get("gated_variants") or []
    pref = gates.get("score_source_preference") or ["human", "llm_judge"]
    q_by = {(q["source"], q["variant"]): q for q in qual}
    present = sorted({o["variant"] for o in ops} | {q["variant"] for q in qual})
    variants = [v for v in present if v in gated]
    small = "small sample: treat as indicative"
    out = []

    def pick(v):
        for src in pref:
            if (src, v) in q_by and q_by[(src, v)]["overall"] is not None:
                return q_by[(src, v)]
        return None

    for v in variants:
        q = pick(v)
        src, n = (q["source"], q["n"]) if q else (None, None)
        note = small if n is not None and n < 30 else ""
        out.append(_gate("Rubric average (1-5)", "Quality", v, q and q["overall"], ">=",
                         gq["rubric_mean_min"], src, n, note))
        out.append(_gate("Hallucination rate", "Quality", v, q and q["hallucination_rate"], "<",
                         gq["hallucination_rate_max"], src, n, note))
        h = q_by.get(("human", v))
        out.append(_gate("Edit effort, median (min)", "Quality", v, h and h["edit_minutes_median"], "<",
                         gq["edit_minutes_median_max"], "human", h and h["n_edit"]))
    for v in variants:
        if v in ("A", "B"):
            continue
        out.append(_gate("Pairwise: wins or ties vs pseudonymised frontier", "Quality", v, None, ">=",
                         gq["pairwise_local_win_or_tie_min"], note="pairwise export not built yet"))
    if redaction:
        for t in gp["direct_identifier_types"]:
            row = next((r for r in redaction["types"] if r["type"] == t), None)
            notes = [] if redaction["has_ner"] else ["regex only: NER was off"]
            if row and row["n"] < 30:
                notes.append(small)
            note = "; ".join(notes)
            out.append(_gate(f"Redaction recall: {t}", "Privacy", "pseudonymiser",
                             row and row["recall"], ">=", gp["redaction_recall_min"],
                             "redaction-eval", row and row["n"], note))
    else:
        out.append(_gate("Redaction recall, direct identifiers", "Privacy", "pseudonymiser", None, ">=",
                         gp["redaction_recall_min"], note="run `cexp redaction-eval`"))
    loss, src, n = None, None, None
    for s in pref:
        a, b = q_by.get((s, "A")), q_by.get((s, "B"))
        if a and b and a["overall"] is not None and b["overall"] is not None:
            loss, src, n = a["overall"] - b["overall"], s, min(a["n"], b["n"])
            break
    out.append(_gate("Quality loss from redaction (raw minus pseudonymised frontier)", "Privacy", "B", loss, "<",
                     gp["redaction_quality_loss_max"], src, n))
    o_by = {o["variant"]: o for o in ops}
    for v in variants:
        o = o_by.get(v)
        n = o and o["n"]
        out.append(_gate("p95 time to draft (s)", "Operations", v, o and o["p95_s"], "<",
                         go["p95_latency_s_max"], "runs", n, small if n and n < 30 else ""))
    return out


# -- per-complaint records (feeds the interactive HTML) ------------------------
def records(
    con: sqlite3.Connection,
    complaints: list[Complaint] | None,
    include_real_text: bool,
    include_text: bool = True,
) -> tuple[list[dict], dict[str, dict]]:
    """One record per (variant, complaint) run, with judge and human scores attached.

    Free text (complaint, draft, note) is included for synthetic complaints only,
    unless `include_real_text` is set: real complaint text must not end up in a
    shareable file by accident. `include_text=False` leaves out all free text.
    """
    by_id = {c.id: c for c in complaints or []}
    judge = defaultdict(list)
    for r in _score_rows(con, "judgements"):
        judge[(r["variant"], r["complaint_id"])].append(r)
    human = defaultdict(list)
    for r in _score_rows(con, "human_scores"):
        human[(r["variant"], r["complaint_id"])].append(r)

    def agg(rows):
        if not rows:
            return None
        acc = defaultdict(list)
        for r in rows:
            for k, v in r["scores"].items():
                acc[k].append(v)
        hall = [r["hallucination"] for r in rows if r.get("hallucination") is not None]
        edits = [r["edit_minutes"] for r in rows if r.get("edit_minutes") is not None]
        return dict(scores={k: st.fmean(v) for k, v in acc.items()}, n=len(rows),
                    hallucination=_mean(hall), edit_minutes=_mean(edits))

    out, texts = [], {}
    for r in con.execute("SELECT * FROM runs ORDER BY complaint_id, variant").fetchall():
        show = include_text and (r["source"] != "real" or include_real_text)
        key = (r["variant"], r["complaint_id"])
        out.append(dict(
            variant=r["variant"], complaint_id=r["complaint_id"], source=r["source"],
            language=r["language"] or "unknown", type=r["type"] or "unknown",
            difficulty=r["difficulty"] or "unknown", model_id=r["model_id"], ts=r["ts"],
            latency_s=r["latency_s"], parse_ok=bool(r["parse_ok"]), error=r["error"],
            prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
            judge=agg(judge.get(key)), human=agg(human.get(key)),
            draft=r["draft"] if show else None, agent_note=r["agent_note"] if show else None,
        ))
        if r["complaint_id"] not in texts:
            c = by_id.get(r["complaint_id"])
            texts[r["complaint_id"]] = dict(
                text=(c.text if c else None) if show else None,
                facts=(c.facts if c else None) if show else None,
                pii_types=sorted({p["type"] for p in c.pii}) if c else [],
                redacted=not show,
            )
    return out, texts


# -- display names ----------------------------------------------------------------
def display_names(keys: list[str], ops: list[dict], variants: dict | None) -> dict[str, str]:
    """Show variants by model name, not by experiment letter.

    Name source, first available: the model id the server reported in the runs,
    the configured model, the configured label, the key. Variants that share a
    model (A/B, C/E) get their hosting and PII handling from the label appended,
    e.g. "gpt-4o (Azure EU Data Zone, raw text)".
    """
    variants = variants or {}
    reported = {o["variant"]: ", ".join(o["model_ids"]) for o in ops if o.get("model_ids")}
    base, detail = {}, {}
    for k in keys:
        v = variants.get(k)
        model = getattr(v, "model", "") or ""
        label = getattr(v, "label", "") or ""
        base[k] = reported.get(k) or (model if model and "${" not in model else "") or label or k
        detail[k] = ", ".join(p.strip() for p in label.split("/")[1:]) or k
    counts: dict[str, int] = defaultdict(int)
    for k in keys:
        counts[base[k]] += 1
    names = {k: f"{base[k]} ({detail[k]})" if counts[base[k]] > 1 else base[k] for k in keys}
    if len(set(names.values())) < len(names):  # still ambiguous: fall back to the key
        names = {k: f"{n} [{k}]" for k, n in names.items()}
    return names


# -- assemble -----------------------------------------------------------------
def build(
    con: sqlite3.Connection,
    *,
    costs: dict | None = None,
    gates: dict | None = None,
    rubric: dict | None = None,
    hours_csv: Path | None = DEFAULT_HOURS_LOG,
    complaints: list[Complaint] | None = None,
    include_real_text: bool = False,
    include_text: bool = True,
    variants: dict | None = None,
) -> dict:
    """Everything the reports show. `variants` (from config) is only used for display names."""
    if variants is None:
        try:
            from .config import load_variants

            variants = load_variants()
        except Exception:  # unreadable config must not break the report
            variants = {}
    costs = costs if costs is not None else load_costs()
    gates = gates if gates is not None else load_gates()
    rubric = rubric if rubric is not None else load_rubric()
    ops = operations(con)
    qual = quality(con)
    red = latest_redaction_eval(con)
    recs, texts = records(con, complaints, include_real_text, include_text)
    gate_rows = evaluate_gates(ops, qual, red, gates)
    keys = sorted({o["variant"] for o in ops} | {q["variant"] for q in qual}
                  | {g["variant"] for g in gate_rows if g["variant"] in variants})
    return dict(
        names=display_names(keys, ops, variants),
        meta=dict(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            harness_version=__version__,
            n_runs=len(recs),
            n_complaints=len(texts),
            variants=[o["variant"] for o in ops],
            include_real_text=include_real_text,
            include_text=include_text,
        ),
        rubric=rubric,
        volumes=costs["volumes_per_month"],
        operations=ops,
        costs=cost_per_1000(ops, costs),
        quality=qual,
        splits=splits(con),
        redaction=red,
        gates=gate_rows,
        effort=effort(hours_csv),
        records=recs,
        complaints=texts,
    )


# -- markdown view ------------------------------------------------------------
def _f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def render_markdown(data: dict) -> str:
    m = data["meta"]
    names = data.get("names", {})
    nm = lambda k: names.get(k, k)  # noqa: E731
    lines = ["# Experiment report", "",
             f"Generated {m['generated_at']} - {m['n_runs']} runs over {m['n_complaints']} complaints, "
             f"models: {', '.join(nm(v) for v in m['variants']) or 'none'}.", ""]

    lines += ["## Gates", "", "| dimension | gate | applies to | value | threshold | status | source (n) | note |",
              "|---|---|---|---|---|---|---|---|"]
    icon = {"pass": "PASS", "fail": "FAIL", "not_measured": "not measured"}
    for g in data["gates"]:
        pct = any(w in g["gate"] for w in ("rate", "recall", "Pairwise"))
        val = "n/a" if g["value"] is None else (f"{g['value']:.1%}" if pct else f"{g['value']:.2f}")
        thr = f"{g['threshold']:.0%}" if pct else f"{g['threshold']:g}"
        src = f"{g['source']} ({g['n']})" if g["source"] and g["n"] is not None else ""
        lines.append(f"| {g['dimension']} | {g['gate']} | {nm(g['variant'])} | {val} | {g['op']} {thr} | "
                     f"{icon[g['status']]} | {src} | {g['note']} |")

    lines += ["", "## Operations", "",
              "| model | n | p50 s | p95 s | errors | parse failures | failure rate | mean in tok | mean out tok |",
              "|---|---|---|---|---|---|---|---|---|"]
    for o in data["operations"]:
        lines.append(
            f"| {nm(o['variant'])} | {o['n']} | {_f(o['p50_s'])} | "
            f"{_f(o['p95_s'])} | {o['errors']} | {o['parse_failures']} | {o['failure_rate']:.1%} | "
            f"{o['mean_prompt_tokens']:.0f} | {o['mean_completion_tokens']:.0f} |"
        )

    vols = data["volumes"]
    lines += ["", "## Cost per 1,000 replies (EUR)", "",
              "| model | " + " | ".join(f"@ {v:,}/month" for v in vols) + " | break-even vs pay-per-token cloud (replies/month) |",
              "|---|" + "---|" * (len(vols) + 1)]
    for r in data["costs"]:
        cells = [_f(r[f"eur_per_1000@{v}"]) for v in vols]
        lines.append(f"| {nm(r['variant'])} | " + " | ".join(cells) +
                     f" | {_f(r.get('break_even_replies_per_month_vs_B'), 0)} |")
    lines += ["", "(n/a = fill in config/costs.yaml)", "", "## Quality", ""]
    q = data["quality"]
    if not q:
        lines.append("No scores yet. Run `cexp judge` and/or `cexp import-scores`.")
    else:
        crit = list(data["rubric"]["criteria"])
        lines += ["| source | model | n | overall (1-5) | 95% CI | hallucination rate | edit min (median) | "
                  + " | ".join(crit) + " |", "|---" * (7 + len(crit)) + "|"]
        for r in q:
            hr = "n/a" if r["hallucination_rate"] is None else f"{r['hallucination_rate']:.1%}"
            ci = "n/a" if not r["ci95"] else f"{r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}"
            per = " | ".join(_f(r["per_criterion"].get(c)) for c in crit)
            lines.append(f"| {r['source']} | {nm(r['variant'])} | {r['n']} | {_f(r['overall'])} | {ci} | {hr} | "
                         f"{_f(r['edit_minutes_median'], 1)} | {per} |")
        for dim in SPLIT_DIMENSIONS:
            rows = [s for s in data["splits"] if s["dimension"] == dim and s["metric"].startswith("overall")]
            if not rows:
                continue
            lines += ["", f"### Quality by {dim}", "", "| source | model | " + dim + " | n | overall |",
                      "|---|---|---|---|---|"]
            for s in rows:
                lines.append(f"| {s['metric'].removeprefix('overall_')} | {nm(s['variant'])} | {s['value']} | "
                             f"{s['n']} | {_f(s['result'])} |")

    lines += ["", "## Privacy: redaction recall", ""]
    red = data["redaction"]
    if not red:
        lines.append("No redaction eval stored. Run `cexp redaction-eval`.")
    else:
        lines += [f"Latest run {red['ts']} on {red['dataset']}, NER {'on' if red['has_ner'] else 'OFF'}.", "",
                  "| PII type | n | leaked | recall |", "|---|---|---|---|"]
        for t in red["types"]:
            lines.append(f"| {t['type']} | {t['n']} | {t['leaked']} | {t['recall']:.1%} |")

    e = data["effort"]
    lines += ["", "## Effort", "", f"{e['total_hours']:.1f} hours logged in {e['entries']} entries."]
    if e["by_phase"]:
        lines += ["", "| phase | hours |", "|---|---|"] + [f"| {k} | {v:.1f} |" for k, v in sorted(e["by_phase"].items())]
    return "\n".join(lines) + "\n"


def render(con: sqlite3.Connection, **kwargs) -> str:
    """Markdown report (kept for backwards compatibility)."""
    return render_markdown(build(con, **kwargs))
