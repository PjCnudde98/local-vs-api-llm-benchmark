"""Aggregate results: capability frontier, operations, cost, quality, privacy, gates, effort.

`build()` turns the SQLite store into one plain dict. `render_markdown()` and
`html_report.render_html()` are two views of that same dict, so the Markdown
and HTML reports can never disagree.
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
from .tiers import DRAFT_TIERS, TIERS

DEFAULT_HOURS_LOG = ROOT / "hours_log.csv"
SPLIT_DIMENSIONS = ("tier", "language", "type", "difficulty", "route")
API_ROUTES = ("pseudonymise", "abstract")  # the API routes allowed for personal data (plan: S2 -> R3 or R2)
PAY_PER_TOKEN_ROUTES = ("pseudonymise", "abstract", "raw")


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


def _first(rows, key):
    return next((r[key] for r in rows if r[key] is not None), None)


# -- operations ---------------------------------------------------------------
def operations(con: sqlite3.Connection, latency_tiers: list[int] | None = None) -> list[dict]:
    """Per variant: counts, failure rate, latency, tokens, residual PII leakage.

    p50/p95 are over `latency_tiers` (time to first draft) when rows carry a
    tier; `p95_all_s` covers every tier, the tool loop included.
    """
    latency_tiers = latency_tiers or [3, 4]
    out = []
    for (v,) in con.execute("SELECT DISTINCT variant FROM runs ORDER BY variant").fetchall():
        rows = con.execute("SELECT * FROM runs WHERE variant=?", (v,)).fetchall()
        ok = [r for r in rows if r["error"] is None]
        lat_all = [r["latency_s"] for r in ok if r["latency_s"] is not None]
        lat = [r["latency_s"] for r in ok if r["latency_s"] is not None
               and (r["tier"] is None or r["tier"] in latency_tiers)]
        errors = [r for r in rows if r["error"] is not None]
        parse_fail = [r for r in ok if not r["parse_ok"]]
        tok = defaultdict(lambda: ([], []))
        for r in ok:
            tok[r["tier"]][0].append(r["prompt_tokens"] or 0)
            tok[r["tier"]][1].append(r["completion_tokens"] or 0)
        ann = sum(r["pii_annotated"] or 0 for r in ok if r["pii_leaked"] is not None)
        leaked = sum(r["pii_leaked"] or 0 for r in ok if r["pii_leaked"] is not None)
        o = dict(
            variant=v, n=len(rows), n_ok=len(ok) - len(parse_fail),
            errors=len(errors), parse_failures=len(parse_fail),
            failure_rate=(len(errors) + len(parse_fail)) / len(rows),
            p50_s=_pct(lat, 0.5), p95_s=_pct(lat, 0.95), n_latency=len(lat),
            p95_all_s=_pct(lat_all, 0.95), mean_s=_mean(lat_all), max_s=max(lat_all) if lat_all else None,
            mean_prompt_tokens=st.fmean([r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [0]),
            mean_completion_tokens=st.fmean([r["completion_tokens"] for r in ok if r["completion_tokens"]] or [0]),
            tokens_by_tier={t: (st.fmean(p), st.fmean(c)) for t, (p, c) in tok.items() if t is not None},
            model_ids=sorted({r["model_id"] for r in rows if r["model_id"]}),
            route=_first(rows, "route"), model_key=_first(rows, "model_key"),
            size_class=_first(rows, "size_class"), reasoning=_first(rows, "reasoning"),
            hardware=_first(rows, "hardware"),
            pii_annotated=ann, pii_leaked=leaked, pii_leak_rate=leaked / ann if ann else None,
        )
        routed = [r for r in ok if r["routed_to"]]
        if routed:
            targets = defaultdict(int)
            for r in routed:
                targets[r["routed_to"]] += 1
            o["routed"] = dict(targets)
        out.append(o)
    return out


# -- cost ---------------------------------------------------------------------
def _price(costs: dict, model_key: str | None) -> tuple[float, float] | None:
    """EUR per 1M input and output tokens, or None while the price list is empty."""
    p = (costs.get("models") or {}).get(model_key or "")
    if not p or p.get("input_per_1m") is None or p.get("output_per_1m") is None:
        return None
    fx = costs.get("usd_to_eur") if p.get("currency", "EUR") == "USD" else 1.0
    if fx is None:
        return None
    return p["input_per_1m"] * fx, p["output_per_1m"] * fx


def fixed_per_month(costs: dict, hardware: str | None) -> float | None:
    """Fixed EUR per month of one local box (or own-tenant VM)."""
    h = (costs.get("hardware") or {}).get(hardware or "")
    if not h:
        return None
    if "vm_eur_per_hour" in h:
        return None if h["vm_eur_per_hour"] is None else h["vm_eur_per_hour"] * h.get("hours_per_month", 730)
    need = ("hardware_eur", "amortisation_months", "hosting_eur_per_month", "ops_hours_per_month", "eur_per_ops_hour")
    if any(h.get(k) is None for k in need):
        return None
    return (h["hardware_eur"] / h["amortisation_months"] + h["hosting_eur_per_month"]
            + h["ops_hours_per_month"] * h["eur_per_ops_hour"])


def best_throughput(con: sqlite3.Connection, p95_max: float) -> dict[str, dict]:
    """Per variant: highest measured replies/hour whose p95 latency passes the gate (latest run)."""
    out: dict[str, dict] = {}
    rows = con.execute("SELECT * FROM throughput ORDER BY ts").fetchall()
    latest: dict[tuple, Any] = {}
    for r in rows:
        latest[(r["variant"], r["concurrency"])] = r
    for (v, _), r in latest.items():
        if r["p95_s"] is None or r["p95_s"] >= p95_max or not r["replies_per_hour"]:
            continue
        if v not in out or r["replies_per_hour"] > out[v]["replies_per_hour"]:
            out[v] = dict(replies_per_hour=r["replies_per_hour"], concurrency=r["concurrency"], p95_s=r["p95_s"])
    return out


def _components(o: dict, costs: dict, variants: dict) -> tuple[float | None, str | None, list[str]]:
    """(variable EUR per reply at the variant's mean tokens, hardware profile with a fixed cost, notes)."""
    route, notes = o.get("route"), []
    v = variants.get(o["variant"])

    def variable(model_key, pt, ct):
        p = _price(costs, model_key)
        return None if p is None else (pt * p[0] + ct * p[1]) / 1e6

    if route in ("local", "own_tenant"):
        return 0.0, o.get("hardware") or getattr(v, "hardware", None), notes
    if route in PAY_PER_TOKEN_ROUTES:
        var = variable(o.get("model_key"), o["mean_prompt_tokens"], o["mean_completion_tokens"])
        hw = None
        if route == "abstract" and v is not None and v.abstractor in variants:
            hw = variants[v.abstractor].hardware
            notes.append(f"includes the abstractor's box ({hw})")
        return var, hw, notes
    if route == "hybrid" and v is not None:
        hard, easy = variants.get(v.hybrid.get("hard")), variants.get(v.hybrid.get("easy"))
        share = (o.get("routed") or {}).get(getattr(hard, "key", ""), 0) / max(1, o["n"] - o["errors"])
        var = variable(getattr(hard, "model_key", None), o["mean_prompt_tokens"], o["mean_completion_tokens"])
        notes.append(f"{share:.0%} of replies routed to {getattr(hard, 'key', '?')}")
        return (None if var is None else var * share), getattr(easy, "hardware", None), notes
    return None, None, ["no cost model for this variant"]


def costs_table(ops: list[dict], qual: list[dict], costs: dict, variants: dict,
                throughput: dict[str, dict]) -> list[dict]:
    """Cost per reply and per sendable reply (model cost + agent edit time) at each volume.

    Local fixed cost scales in steps: servers needed = ceil(volume / (replies per
    hour x 730 h x max_utilisation)), from measured throughput. Without a
    throughput measurement one server is assumed.
    """
    volumes = costs["volumes_per_month"]
    edit = {q["variant"]: q["edit_minutes_median"] for q in qual
            if q["source"] == "human" and q["edit_minutes_median"] is not None}
    rows = []
    for o in ops:
        var, hw, notes = _components(o, costs, variants)
        fixed = fixed_per_month(costs, hw) if hw else 0.0
        if hw and fixed is None:
            notes.append(f"hardware profile {hw!r} has empty inputs")
        tp = throughput.get(o["variant"])
        row: dict[str, Any] = dict(variant=o["variant"], eur_per_reply_variable=var, fixed_eur_per_month=fixed,
                                   hardware=hw, edit_minutes=edit.get(o["variant"]),
                                   replies_per_hour=tp and tp["replies_per_hour"], notes=notes)
        if hw and not tp:
            notes.append("throughput not measured: one server assumed")
        for vol in volumes:
            servers = 1
            if hw and tp:
                cap = tp["replies_per_hour"] * 730 * costs.get("max_utilisation", 1.0)
                servers = max(1, math.ceil(vol / cap))
            per = None if var is None or fixed is None else var + fixed * servers / vol
            row[f"servers@{vol}"] = servers if hw else None
            row[f"eur_per_1000@{vol}"] = None if per is None else per * 1000
            em = edit.get(o["variant"])
            row[f"eur_per_sendable@{vol}"] = (None if per is None or em is None
                                              else per + em * costs.get("agent_eur_per_hour", 0) / 60)
        rows.append(row)
    api = [r for r in rows if r["eur_per_reply_variable"] and r["fixed_eur_per_month"] == 0
           and (o_route(ops, r["variant"]) in API_ROUTES)]
    if api:
        cheapest = min(api, key=lambda r: r["eur_per_reply_variable"])
        for r in rows:
            if r["fixed_eur_per_month"] and r["eur_per_reply_variable"] == 0:
                r["break_even_replies_per_month"] = r["fixed_eur_per_month"] / cheapest["eur_per_reply_variable"]
                r["break_even_vs"] = cheapest["variant"]
    return rows


VOLUME_GRID = [100 * 10 ** (e / 4) for e in range(0, 21)]  # 100 to 10 million replies/month, 4 steps per decade


def _monthly(opt: dict, vol: float) -> float | None:
    """Monthly EUR of one option at a volume: boxes x fixed cost, plus tokens for API routes."""
    if opt["fixed_eur_per_month"] is None or opt["eur_per_reply_variable"] is None:
        return None
    cap = opt.get("capacity_per_month")
    boxes = max(1, math.ceil(vol / cap)) if cap else 1
    return boxes * opt["fixed_eur_per_month"] + opt["eur_per_reply_variable"] * vol


def _break_even(local: dict, api: dict, max_boxes: int = 1000) -> tuple[float | None, str]:
    """Lowest monthly volume from which `local` costs no more than `api`, and why (or why never).

    Local cost is flat per box and steps up when a box is full; API cost grows with
    every reply. If the API line has not overtaken one box by the time that box is
    full, it never overtakes: a full box is the cheapest a local reply gets.
    """
    fl, cap = local["fixed_eur_per_month"], local.get("capacity_per_month")
    fa, va = api["fixed_eur_per_month"] or 0.0, api["eur_per_reply_variable"]
    if fl is None or va is None:
        return None, "inputs missing"
    if not va:
        return None, "the API option has no per-reply cost"
    for n in range(1, (max_boxes if cap else 1) + 1):
        lo = (n - 1) * cap if cap else 0.0
        vol = max(lo, (n * fl - fa) / va, 0.0)
        if cap is None or vol <= n * cap:
            note = "" if cap else "capacity unknown: assumes one box keeps up"
            return vol, note
    full = fl / cap
    return None, (f"never: a full box costs EUR {full:.4f} per reply, "
                  f"the API EUR {va:.4f}")


def volume_view(ops: list[dict], cost_rows: list[dict], costs: dict, front: dict,
                variants: dict) -> dict:
    """Cost against monthly volume: where each local option becomes cheaper than each API route.

    Local options are the measured local variants (their box, with capacity from a
    throughput run, else estimated from the run latency at concurrency 1) and every
    other hardware profile in costs.yaml with complete inputs (capacity unknown until a
    model is measured on it). API options are the routes allowed for personal data.
    `quality_through`: the last tier of an unbroken run of passed tiers from tier 1,
    so the page can compare only options good enough for the tiers you need.
    """
    util = costs.get("max_utilisation", 1.0)
    o_by = {o["variant"]: o for o in ops}
    through = {}
    for s in front.get("summary") or []:
        k = 0
        while k + 1 in s["passed"]:
            k += 1
        through[s["variant"]] = k
    local, api = [], []
    for r in cost_rows:
        o, route = o_by.get(r["variant"], {}), o_route(ops, r["variant"])
        base = dict(key=r["variant"], kind="variant", hardware=r["hardware"],
                    fixed_eur_per_month=r["fixed_eur_per_month"], eur_per_reply_variable=r["eur_per_reply_variable"],
                    quality_through=through.get(r["variant"]))
        if route in ("local", "own_tenant"):
            if r.get("replies_per_hour"):
                cap, src = r["replies_per_hour"] * 730 * util, "measured throughput"
            elif o.get("mean_s"):
                cap, src = 3600 / o["mean_s"] * 730 * util, "run latency, one request at a time (lower bound)"
            else:
                cap, src = None, "unknown"
            local.append(dict(base, capacity_per_month=cap, capacity_source=src))
        elif route in API_ROUTES and r["eur_per_reply_variable"] is not None:
            api.append(dict(base, capacity_per_month=None))
    measured = {x["hardware"] for x in local}
    for hw in costs.get("hardware") or {}:
        fixed = fixed_per_month(costs, hw)
        if fixed is not None and hw not in measured:
            local.append(dict(key=f"hardware:{hw}", kind="hardware", hardware=hw, fixed_eur_per_month=fixed,
                              eur_per_reply_variable=0.0, quality_through=None,
                              capacity_per_month=None, capacity_source="no model measured on it yet"))
    for x in local:
        cap = x["capacity_per_month"]
        x["eur_per_reply_at_full_box"] = x["fixed_eur_per_month"] / cap if cap and x["fixed_eur_per_month"] else None
    pairs = []
    for x in local:
        for a in api:
            vol, note = _break_even(x, a)
            pairs.append(dict(local=x["key"], api=a["key"], volume=vol, note=note))
    grid = [round(v) for v in VOLUME_GRID]
    curves = {x["key"]: [_monthly(x, v) for v in grid] for x in local + api}
    return dict(grid=grid, local=local, api=api, break_even=pairs, curves=curves,
                volumes=costs["volumes_per_month"], max_utilisation=util,
                agent_eur_per_hour=costs.get("agent_eur_per_hour"))


def o_route(ops: list[dict], variant: str) -> str | None:
    return next((o.get("route") for o in ops if o["variant"] == variant), None)


# -- quality ------------------------------------------------------------------
SCORE_TABLES = (("human", "human_scores"), ("llm_judge", "judgements"))


def _score_rows(con: sqlite3.Connection, table: str) -> list[dict]:
    """Score rows joined with run metadata (tier/language/type/difficulty/route)."""
    q = f"""SELECT s.*, r.language, r.type, r.difficulty, r.source, r.tier, r.route
            FROM {table} s LEFT JOIN runs r
              ON r.variant = s.variant AND r.complaint_id = s.complaint_id"""
    out = []
    for r in con.execute(q).fetchall():
        d = dict(r)
        d["scores"] = json.loads(d.pop("scores_json"))
        d["extra"] = json.loads(d["extra_json"]) if d.get("extra_json") else None
        d.pop("extra_json", None)
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


def judge_agreement(con: sqlite3.Connection, variants: dict | None = None) -> list[dict]:
    """How far the score sources agree on the drafts they both scored.

    Sources: every judge model on its own, and the human raters (averaged per draft).
    Per pair: the mean overall of each, the mean signed and absolute difference, Pearson r
    and how often the hallucination flags match. `by_family` splits the signed difference
    by the drafting model's family: a judge that favours its own family shows a larger gap
    on that family's drafts than on the others.
    """
    per: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in _score_rows(con, "judgements"):
        per[r["judge_model"]][(r["variant"], r["complaint_id"])] = r
    human: dict[tuple, list[dict]] = defaultdict(list)
    for r in _score_rows(con, "human_scores"):
        human[(r["variant"], r["complaint_id"])].append(r)
    for k, rs in human.items():
        hall = [x["hallucination"] for x in rs if x.get("hallucination") is not None]
        per["human"][k] = dict(overall=st.fmean(x["overall"] for x in rs),
                               hallucination=round(st.fmean(hall)) if hall else None)
    family = {k: getattr(v, "family", "") or "unknown" for k, v in (variants or {}).items()}
    out = []
    sources = sorted(per, key=lambda s: (s == "human", s))
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            both = sorted(set(per[a]) & set(per[b]))
            if not both:
                continue
            xa = [per[a][k]["overall"] for k in both]
            xb = [per[b][k]["overall"] for k in both]
            diff = [p - q for p, q in zip(xa, xb)]
            try:
                r = st.correlation(xa, xb) if len(both) >= 3 else None
            except st.StatisticsError:  # one side gave every draft the same score
                r = None
            flags = [(per[a][k].get("hallucination"), per[b][k].get("hallucination")) for k in both]
            flags = [(p, q) for p, q in flags if p is not None and q is not None]
            fam: dict[str, list[float]] = defaultdict(list)
            for k, d in zip(both, diff):
                fam[family.get(k[0], "unknown")].append(d)
            out.append(dict(a=a, b=b, n=len(both), mean_a=st.fmean(xa), mean_b=st.fmean(xb),
                            mean_diff=st.fmean(diff), mean_abs_diff=st.fmean(abs(d) for d in diff), pearson_r=r,
                            hallucination_agreement=(sum(p == q for p, q in flags) / len(flags)) if flags else None,
                            by_family={f: dict(n=len(ds), mean_diff=st.fmean(ds)) for f, ds in sorted(fam.items())}))
    return out


def splits(con: sqlite3.Connection) -> list[dict]:
    """Quality and latency per tier / language / complaint type / difficulty / route."""
    out = []
    for src, table in SCORE_TABLES:
        rows = _score_rows(con, table)
        for dim in SPLIT_DIMENSIONS:
            groups: dict[tuple, list[float]] = defaultdict(list)
            for r in rows:
                if r["overall"] is not None:
                    groups[(r["variant"], str(r.get(dim) or "unknown"))].append(r["overall"])
            for (v, val), xs in sorted(groups.items()):
                out.append(dict(metric=f"overall_{src}", dimension=dim, value=val, variant=v,
                                n=len(xs), result=st.fmean(xs)))
    run_rows = con.execute("SELECT * FROM runs WHERE error IS NULL AND latency_s IS NOT NULL").fetchall()
    for dim in SPLIT_DIMENSIONS:
        groups = defaultdict(list)
        for r in run_rows:
            groups[(r["variant"], str(r[dim] or "unknown"))].append(r["latency_s"])
        for (v, val), xs in sorted(groups.items()):
            out.append(dict(metric="p50_latency_s", dimension=dim, value=val, variant=v,
                            n=len(xs), result=_pct(xs, 0.5)))
    return out


# -- capability frontier (task complexity ladder) ---------------------------------
def _cmp(value, op, threshold) -> str:
    if value is None:
        return "not_measured"
    return "pass" if (value >= threshold if op == ">=" else value < threshold) else "fail"


def tier_metrics(con: sqlite3.Connection, gates: dict, rubric: dict) -> dict[tuple[str, int], dict]:
    """Every tier metric per (variant, tier), from answered runs (errors excluded).

    A malformed answer counts as a failed task: label accuracy 0, all gold fields
    missed, wrong decision, lowest rubric score, not compliant, inconsistent.
    Rubric metrics use human scores for a (variant, tier) when any exist, else the judge.
    """
    lo = rubric["scale"][0]
    min_ok = gates.get("policy_compliant_min_score", 4)
    pref = gates.get("score_source_preference") or ["human", "llm_judge"]
    scores: dict[str, dict[tuple, list[dict]]] = {}
    for src, table in SCORE_TABLES:
        d = defaultdict(list)
        for r in _score_rows(con, table):
            d[(r["variant"], r["complaint_id"])].append(r)
        scores[src] = d
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for r in con.execute("SELECT * FROM runs WHERE error IS NULL AND tier IS NOT NULL").fetchall():
        groups[(r["variant"], r["tier"])].append(r)
    out: dict[tuple[str, int], dict] = {}
    for (v, tier), rows in groups.items():
        m: dict[str, Any] = {"n": len(rows)}
        met = [json.loads(r["metrics_json"]) if r["metrics_json"] else {} for r in rows]
        if tier == 1:
            m["label_accuracy"] = _mean([x.get("label_accuracy", 0.0) for x in met])
        elif tier == 2:
            tp, fp, fn = (sum(x.get(k, 0) for x in met) for k in ("tp", "fp", "fn"))
            m["field_f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
            m["valid_json_rate"] = _mean([r["parse_ok"] for r in rows])
        elif tier == 5:
            m["decision_accuracy"] = _mean([x.get("decision_correct", 0) for x in met])
            calls = sum(r["tool_calls"] or 0 for r in rows)
            m["valid_tool_call_rate"] = sum(r["valid_tool_calls"] or 0 for r in rows) / calls if calls else None
        if tier in DRAFT_TIERS:
            src = next((s for s in pref if any((v, r["complaint_id"]) in scores[s] for r in rows)), None)
            m["score_source"] = src
            if src:
                overall, compliant, n_scored = [], [], 0
                for r in rows:
                    srows = scores[src].get((v, r["complaint_id"]))
                    if not r["parse_ok"]:
                        overall.append(lo)
                        compliant.append(0)
                    elif srows:
                        n_scored += 1
                        overall.append(st.fmean(x["overall"] for x in srows))
                        pc = [x["scores"].get("policy_compliance") for x in srows]
                        pc = [p for p in pc if p is not None]
                        if pc:
                            compliant.append(int(st.fmean(pc) >= min_ok))
                m["n_scored"] = n_scored
                if n_scored:
                    m["rubric_mean"] = _mean(overall)
                    if tier == 4:
                        m["policy_compliance_rate"] = _mean(compliant)
            if tier == 6:
                cons, missed, total = [], 0, 0
                for r in rows:
                    ex = [x["extra"] for x in scores["llm_judge"].get((v, r["complaint_id"]), []) if x["extra"]]
                    if not r["parse_ok"]:
                        cons.append(0)
                        continue
                    if ex:
                        cons.append(_mean([e["consistent"] for e in ex]))
                        missed += _mean([len(e["missed_facts"]) for e in ex])
                        total += ex[0]["n_key_facts"]
                if total:
                    m["factual_consistency_rate"] = _mean(cons)
                    m["missed_fact_rate"] = missed / total
        out[(v, tier)] = m
    return out


def frontier(con: sqlite3.Connection, gates: dict, rubric: dict, cost_rows: list[dict] | None = None,
             ops: list[dict] | None = None, costs: dict | None = None, variants: dict | None = None) -> dict:
    """Pass/fail per variant per tier, each variant's failure point, and economic failures.

    Quality failure: a tier metric misses its threshold.
    Economic failure (local variants): the tier passes, but at a given volume the
    local cost per reply (per sendable reply when edit time is known for both)
    exceeds the cheapest API variant (pseudonymise/abstract route) that passes the same tier.
    """
    spec = {int(t): ms for t, ms in (gates.get("tiers") or {}).items()}
    small_n = gates.get("items_per_tier", 25)
    data = tier_metrics(con, gates, rubric)
    variants_present = sorted({v for v, _ in data})
    cells = []
    for v in variants_present:
        for tier in sorted(spec):
            m = data.get((v, tier))
            metrics = []
            for s in spec[tier]:
                val = m.get(s["metric"]) if m else None
                metrics.append(dict(metric=s["metric"], value=val, op=s["op"], threshold=s["threshold"],
                                    status=_cmp(val, s["op"], s["threshold"])))
            sts = [x["status"] for x in metrics]
            status = "fail" if "fail" in sts else "pass" if all(x == "pass" for x in sts) else "not_measured"
            main = metrics[0]
            ratio = (main["value"] / main["threshold"] if main["value"] is not None and main["op"] == ">="
                     and main["threshold"] else None)
            cells.append(dict(variant=v, tier=tier, status=status, metrics=metrics, n=m["n"] if m else 0,
                              n_scored=(m or {}).get("n_scored"), score_source=(m or {}).get("score_source"),
                              main_ratio=ratio,
                              note="small sample: treat as indicative" if m and m["n"] < small_n else ""))
    summary = []
    by_v = defaultdict(list)
    for c in cells:
        by_v[c["variant"]].append(c)
    for v, cs in by_v.items():
        fails = [c["tier"] for c in cs if c["status"] == "fail"]
        passed = [c["tier"] for c in cs if c["status"] == "pass"]
        summary.append(dict(variant=v, breaks_at=min(fails) if fails else None, passed=passed,
                            measured=[c["tier"] for c in cs if c["status"] != "not_measured"]))
    if ops and costs is not None and variants is not None:
        _economic(cells, ops, costs, variants, cost_rows or [])
    return dict(cells=cells, summary=summary, tiers={t: TIERS[t] for t in spec})


def _economic(cells: list[dict], ops: list[dict], costs: dict, variants: dict, cost_rows: list[dict]) -> None:
    """Mark local cells that pass on quality but lose to the cheapest passing API variant on cost."""
    o_by = {o["variant"]: o for o in ops}
    c_by = {r["variant"]: r for r in cost_rows}
    agent = costs.get("agent_eur_per_hour", 0) / 60

    def per_reply(v, tier, vol):
        o, c = o_by.get(v), c_by.get(v)
        if not o or not c or c["fixed_eur_per_month"] is None:
            return None
        pt, ct = o["tokens_by_tier"].get(tier, (o["mean_prompt_tokens"], o["mean_completion_tokens"]))
        if o.get("route") in PAY_PER_TOKEN_ROUTES:
            p = _price(costs, o.get("model_key"))
            if p is None:
                return None
            var = (pt * p[0] + ct * p[1]) / 1e6
        elif o.get("route") in ("local", "own_tenant"):
            var = 0.0
        else:
            return None
        servers = c.get(f"servers@{vol}") or 1
        return var + c["fixed_eur_per_month"] * servers / vol

    for tier in {c["tier"] for c in cells}:
        passing = [c for c in cells if c["tier"] == tier and c["status"] == "pass"]
        api = [c for c in passing if o_by.get(c["variant"], {}).get("route") in API_ROUTES]
        for cell in passing:
            if o_by.get(cell["variant"], {}).get("route") not in ("local", "own_tenant"):
                continue
            cell["economic"] = {}
            for vol in costs["volumes_per_month"]:
                local = per_reply(cell["variant"], tier, vol)
                rivals = [(per_reply(a["variant"], tier, vol), a["variant"]) for a in api]
                rivals = [r for r in rivals if r[0] is not None]
                if local is None or not rivals:
                    cell["economic"][vol] = None
                    continue
                best, who = min(rivals)
                le, ae = c_by[cell["variant"]].get("edit_minutes"), c_by[who].get("edit_minutes")
                if le is not None and ae is not None:
                    local, best = local + le * agent, best + ae * agent
                cell["economic"][vol] = dict(local_eur=local, api_eur=best, api_variant=who,
                                             status="fail" if local > best else "pass")


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
    return dict(gate=gate, dimension=dimension, variant=variant, value=value, op=op,
                threshold=threshold, status=_cmp(value, op, threshold), source=source, n=n, note=note)


def _model_and_route(key: str) -> tuple[str, str]:
    route, _, rest = key.partition("/")
    return rest, route


def evaluate_gates(ops: list[dict], qual: list[dict], redaction: dict | None, gates: dict) -> list[dict]:
    gq, gp, go = gates["quality"], gates["privacy"], gates["operations"]
    ungated = set(gates.get("ungated_variants") or [])
    pref = gates.get("score_source_preference") or ["human", "llm_judge"]
    q_by = {(q["source"], q["variant"]): q for q in qual}
    present = sorted({o["variant"] for o in ops} | {q["variant"] for q in qual})
    variants = [v for v in present if v not in ungated]
    routes = {o["variant"]: o.get("route") for o in ops}
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
        if routes.get(v) not in ("local", "hybrid"):
            continue
        out.append(_gate("Pairwise: wins or ties vs pseudonymised API", "Quality", v, None, ">=",
                         gq["pairwise_local_win_or_tie_min"], note="pairwise export not built yet"))
    if redaction:
        for t in gp["direct_identifier_types"]:
            row = next((r for r in redaction["types"] if r["type"] == t), None)
            notes = [] if redaction["has_ner"] else ["regex only: NER was off"]
            if row and row["n"] < 30:
                notes.append(small)
            out.append(_gate(f"Redaction recall: {t}", "Privacy", "pseudonymiser",
                             row and row["recall"], ">=", gp["redaction_recall_min"],
                             "redaction-eval", row and row["n"], "; ".join(notes)))
    else:
        out.append(_gate("Redaction recall, direct identifiers", "Privacy", "pseudonymiser", None, ">=",
                         gp["redaction_recall_min"], note="run `cexp redaction-eval`"))
    # Quality loss from privacy preprocessing: raw minus pseudonymised (and abstract), same model, same source.
    raw_keys = [v for v in present if routes.get(v) == "raw" or v.startswith("raw/")]
    for raw_key in raw_keys:
        model, _ = _model_and_route(raw_key)
        for route in API_ROUTES:
            other = f"{route}/{model}"
            if other not in present:
                continue
            loss, src, n = None, None, None
            for s in pref:
                a, b = q_by.get((s, raw_key)), q_by.get((s, other))
                if a and b and a["overall"] is not None and b["overall"] is not None:
                    loss, src, n = a["overall"] - b["overall"], s, min(a["n"], b["n"])
                    break
            out.append(_gate(f"Quality loss from {route} (raw minus {route})", "Privacy", other, loss, "<",
                             gp["redaction_quality_loss_max"], src, n))
    if not raw_keys:
        out.append(_gate("Quality loss from pseudonymisation (raw minus pseudonymised)", "Privacy",
                         "pseudonymise", None, "<", gp["redaction_quality_loss_max"],
                         note="needs the same model on the raw and pseudonymise routes"))
    o_by = {o["variant"]: o for o in ops}
    for v in variants:
        o = o_by.get(v)
        n = o and o["n_latency"]
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

    Free text (complaint, answer, note) is included for synthetic complaints only,
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
        extras = [r["extra"] for r in rows if r.get("extra")]
        return dict(scores={k: st.fmean(v) for k, v in acc.items()}, n=len(rows),
                    hallucination=_mean(hall), edit_minutes=_mean(edits),
                    consistent=_mean([e["consistent"] for e in extras]) if extras else None)

    out, texts = [], {}
    for r in con.execute("SELECT * FROM runs ORDER BY complaint_id, variant").fetchall():
        show = include_text and (r["source"] != "real" or include_real_text)
        key = (r["variant"], r["complaint_id"])
        out.append(dict(
            variant=r["variant"], complaint_id=r["complaint_id"], source=r["source"],
            tier=r["tier"], route=r["route"] or "unknown", size_class=r["size_class"] or "unknown",
            model_key=r["model_key"], reasoning=r["reasoning"], routed_to=r["routed_to"],
            language=r["language"] or "unknown", type=r["type"] or "unknown",
            difficulty=r["difficulty"] or "unknown", model_id=r["model_id"], ts=r["ts"],
            latency_s=r["latency_s"], parse_ok=bool(r["parse_ok"]), error=r["error"],
            prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"],
            metrics=json.loads(r["metrics_json"]) if r["metrics_json"] else None,
            tool_calls=r["tool_calls"], valid_tool_calls=r["valid_tool_calls"],
            pii_annotated=r["pii_annotated"], pii_leaked=r["pii_leaked"],
            judge=agg(judge.get(key)), human=agg(human.get(key)),
            draft=r["draft"] if show else None, agent_note=r["agent_note"] if show else None,
        ))
        if r["complaint_id"] not in texts:
            c = by_id.get(r["complaint_id"])
            texts[r["complaint_id"]] = dict(
                text=(c.text if c else None) if show else None,
                facts=(c.facts if c else None) if show else None,
                gold=(c.gold if c else None) if show else None,
                pii_types=sorted({p["type"] for p in c.pii}) if c else [],
                redacted=not show,
            )
    return out, texts


# -- display names ----------------------------------------------------------------
ROUTE_LABEL = {"local": "local", "pseudonymise": "pseudonymised", "abstract": "abstracted", "raw": "raw text",
               "own_tenant": "own-tenant VM", "hybrid": "hybrid"}


def display_names(keys: list[str], ops: list[dict], variants: dict | None) -> dict[str, str]:
    """Show variants by model name and route, e.g. "Claude Haiku 5.5 (pseudonymised)".

    Name source, first available: the configured model label, the model id the
    server reported, the configured label, the key. Reasoning on adds ", thinking".
    """
    variants = variants or {}
    reported = {o["variant"]: ", ".join(o["model_ids"]) for o in ops if o.get("model_ids")}
    names = {}
    for k in keys:
        v = variants.get(k)
        if v is not None and v.route == "hybrid":
            easy, hard = variants.get(v.hybrid.get("easy")), variants.get(v.hybrid.get("hard"))
            names[k] = (f"Hybrid: {getattr(easy, 'model_label', '?') or '?'} / "
                        f"{getattr(hard, 'model_label', '?') or '?'}")
            continue
        label = getattr(v, "model_label", "") or ""
        model = getattr(v, "model", "") or ""
        base = label or reported.get(k) or (model if model and "${" not in model else "") or \
            getattr(v, "label", "") or k
        detail = [ROUTE_LABEL.get(getattr(v, "route", ""), "")] if v is not None and v.route else []
        if v is not None and v.reasoning == "on":
            detail.append("thinking")
        detail = [d for d in detail if d]
        names[k] = f"{base} ({', '.join(detail)})" if detail else base
    if len(set(names.values())) < len(names):  # still ambiguous: fall back to the key
        names = {k: f"{n} [{k}]" for k, n in names.items()}
    return names


def lineup(con: sqlite3.Connection, keys: list[str], variants: dict) -> dict:
    """Who took part: model id and hosting per variant, the abstractor and the judges actually used."""
    models: dict[str, dict] = {}
    for k in keys:
        v = variants.get(k)
        if v is None or not hasattr(v, "model"):
            continue
        models[k] = dict(model_label=v.model_label or v.label, model_id=v.model, hosting=v.hosting,
                         hardware=v.hardware, abstractor=v.abstractor if v.route == "abstract" else None)
    judge_ids = [r[0] for r in con.execute("SELECT DISTINCT judge_model FROM judgements ORDER BY 1")]
    labels = {getattr(v, "model", None): v.label for k, v in variants.items()
              if k.startswith("JUDGE") and hasattr(v, "label")}
    return dict(variants=models, judges=[{"model_id": j, "label": labels.get(j, j)} for j in judge_ids])


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
    """Everything the reports show. `variants` (from config) gives names and the cost model's routes."""
    if variants is None:
        try:
            from .config import load_variants

            variants = load_variants()
        except Exception:  # unreadable config must not break the report
            variants = {}
    costs = costs if costs is not None else load_costs()
    gates = gates if gates is not None else load_gates()
    rubric = rubric if rubric is not None else load_rubric()
    ops = operations(con, (gates.get("operations") or {}).get("latency_tiers"))
    qual = quality(con)
    red = latest_redaction_eval(con)
    tp = best_throughput(con, gates["operations"]["p95_latency_s_max"])
    cost_rows = costs_table(ops, qual, costs, variants, tp)
    recs, texts = records(con, complaints, include_real_text, include_text)
    gate_rows = evaluate_gates(ops, qual, red, gates)
    front = frontier(con, gates, rubric, cost_rows, ops, costs, variants)
    keys = sorted({o["variant"] for o in ops} | {q["variant"] for q in qual}
                  | {g["variant"] for g in gate_rows if g["variant"] in variants})
    meta_by = {o["variant"]: {k: o.get(k) for k in ("route", "size_class", "model_key", "reasoning")} for o in ops}
    return dict(
        names=display_names(keys, ops, variants),
        variant_meta=meta_by,
        lineup=lineup(con, keys, variants),
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
        costs=cost_rows,
        volume=volume_view(ops, cost_rows, costs, front, variants),
        throughput=tp,
        quality=qual,
        judge_agreement=judge_agreement(con, variants),
        splits=splits(con),
        redaction=red,
        gates=gate_rows,
        frontier=front,
        effort=effort(hours_csv),
        records=recs,
        complaints=texts,
    )


# -- markdown view ------------------------------------------------------------
def _f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def _metric_value(m: dict) -> str:
    if m["value"] is None:
        return "n/a"
    return f"{m['value']:.2f}" if m["metric"] == "rubric_mean" else f"{m['value']:.0%}"


def render_markdown(data: dict) -> str:
    m = data["meta"]
    names = data.get("names", {})
    nm = lambda k: names.get(k, k)  # noqa: E731
    lines = ["# Experiment report", "",
             f"Generated {m['generated_at']} - {m['n_runs']} runs over {m['n_complaints']} complaints, "
             f"{len(m['variants'])} variants.", ""]

    fr = data.get("frontier") or {}
    if fr.get("cells"):
        tiers = sorted({c["tier"] for c in fr["cells"]})
        lines += ["## Capability frontier", "",
                  "Each cell: status and main metric. A model's failure point is the first tier it fails.", "",
                  "| model | " + " | ".join(f"T{t} {fr['tiers'][t]}" for t in tiers) + " | breaks at |",
                  "|---|" + "---|" * (len(tiers) + 1)]
        summ = {s["variant"]: s for s in fr["summary"]}
        icon = {"pass": "PASS", "fail": "FAIL", "not_measured": "-"}
        for v in sorted({c["variant"] for c in fr["cells"]}):
            cs = {c["tier"]: c for c in fr["cells"] if c["variant"] == v}
            cells = []
            for t in tiers:
                c = cs[t]
                txt = icon[c["status"]]
                if c["status"] != "not_measured":
                    txt += " " + " / ".join(_metric_value(x) for x in c["metrics"])
                econ = [vol for vol, e in (c.get("economic") or {}).items() if e and e["status"] == "fail"]
                if econ:
                    txt += f" (econ. fail @ {', '.join(f'{x:,}' for x in econ)})"
                cells.append(txt)
            b = summ[v]["breaks_at"]
            lines.append(f"| {nm(v)} | " + " | ".join(cells) + f" | {'T' + str(b) if b else '-'} |")

    lines += ["", "## Gates", "", "| dimension | gate | applies to | value | threshold | status | source (n) | note |",
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
              "| model | n | p50 s (draft) | p95 s (draft) | p95 s (all tiers) | errors | parse failures | "
              "failure rate | mean in tok | mean out tok | PII leaked |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in data["operations"]:
        leak = "" if o["pii_leak_rate"] is None else f"{o['pii_leaked']}/{o['pii_annotated']}"
        lines.append(
            f"| {nm(o['variant'])} | {o['n']} | {_f(o['p50_s'])} | {_f(o['p95_s'])} | {_f(o['p95_all_s'])} | "
            f"{o['errors']} | {o['parse_failures']} | {o['failure_rate']:.1%} | "
            f"{o['mean_prompt_tokens']:.0f} | {o['mean_completion_tokens']:.0f} | {leak} |"
        )

    vols = data["volumes"]
    lines += ["", "## Cost (EUR)", "",
              "| model | " + " | ".join(f"per 1,000 @ {v:,}/month" for v in vols) + " | "
              + " | ".join(f"per sendable reply @ {v:,}" for v in vols)
              + " | fixed / month | break-even (replies/month) | notes |",
              "|---|" + "---|" * (2 * len(vols) + 3)]
    for r in data["costs"]:
        cells = [_f(r[f"eur_per_1000@{v}"]) for v in vols] + [_f(r[f"eur_per_sendable@{v}"], 3) for v in vols]
        be = r.get("break_even_replies_per_month")
        lines.append(f"| {nm(r['variant'])} | " + " | ".join(cells) +
                     f" | {_f(r['fixed_eur_per_month'])} | {_f(be, 0)}"
                     f"{' vs ' + nm(r['break_even_vs']) if be else ''} | {'; '.join(r['notes'])} |")
    lines += ["", "(n/a = fill in config/costs.yaml; per sendable reply needs human edit times)", ""]
    lines += _volume_markdown(data.get("volume"), nm)
    lines += ["", "## Quality", ""]
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

    ja = data.get("judge_agreement") or []
    lines += ["", "## Judge agreement", ""]
    if not ja:
        lines.append("Only one score source so far. Run a second judge (`cexp judge --judge JUDGE_GPT` or "
                     "`JUDGE_OPENAI`) and score a small human sample (`cexp export-blind --drafts 60`).")
    else:
        lines += ["Same drafts, two score sources. Difference = first minus second. A larger difference on "
                  "one model family's drafts than on the others points to a judge favouring that family.", "",
                  "| first | second | n | mean first | mean second | difference | mean abs. difference | "
                  "Pearson r | same hallucination flag | difference by drafting family |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for j in ja:
            fam = "; ".join(f"{f} {x['mean_diff']:+.2f} (n={x['n']})" for f, x in j["by_family"].items())
            ha = "n/a" if j["hallucination_agreement"] is None else f"{j['hallucination_agreement']:.0%}"
            lines.append(f"| {j['a']} | {j['b']} | {j['n']} | {j['mean_a']:.2f} | {j['mean_b']:.2f} | "
                         f"{j['mean_diff']:+.2f} | {j['mean_abs_diff']:.2f} | {_f(j['pearson_r'])} | {ha} | {fam} |")

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


def _vol(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.0f}"


def _volume_markdown(vv: dict | None, nm) -> list[str]:
    """Break-even volume per local option against each API route."""
    if not vv or not vv["local"] or not vv["api"]:
        return ["### When local becomes cheaper", "",
                "Needs at least one local option with a fixed cost and one API route with a token price."]
    lab = lambda k: k.removeprefix("hardware:") + " (hardware only)" if k.startswith("hardware:") else nm(k)  # noqa: E731
    be = {(p["local"], p["api"]): p for p in vv["break_even"]}
    out = ["### When local becomes cheaper", "",
           "Monthly volume from which a local option costs no more than the API route: fixed cost per box "
           "(hardware amortisation, hosting, ops hours) against tokens x price. Quality is not part of this: "
           "check the frontier for which options pass the tiers you need. Agent edit time is left out (the "
           "same on both sides at equal quality).", "",
           "| local option | fixed EUR / month / box | capacity per box (replies/month) | capacity from | "
           "EUR per reply, full box | " + " | ".join(f"vs {nm(a['key'])}" for a in vv["api"]) + " |",
           "|---|---|---|---|---|" + "---|" * len(vv["api"])]
    for x in vv["local"]:
        cells = []
        for a in vv["api"]:
            p = be[(x["key"], a["key"])]
            cells.append("always" if p["volume"] == 0 else _vol(p["volume"]) if p["volume"] is not None
                         else (p["note"].split(":")[0] or "n/a"))
        out.append(f"| {lab(x['key'])} | {_f(x['fixed_eur_per_month'], 0)} | {_vol(x['capacity_per_month'])} | "
                   f"{x['capacity_source']} | {_f(x['eur_per_reply_at_full_box'], 4)} | " + " | ".join(cells) + " |")
    out += ["", "API routes per reply: " + "; ".join(f"{nm(a['key'])} EUR {a['eur_per_reply_variable']:.5f}"
                                                   + (f" + EUR {a['fixed_eur_per_month']:,.0f}/month abstractor box"
                                                      if a["fixed_eur_per_month"] else "") for a in vv["api"]) + ".",
            "\"never\" = even a fully used box costs more per reply than the API; \"always\" = the API route "
            "already pays for a box of that cost (the abstractor). Capacity uses "
            f"max_utilisation {vv['max_utilisation']:g}."]
    return out


def render(con: sqlite3.Connection, **kwargs) -> str:
    """Markdown report (kept for backwards compatibility)."""
    return render_markdown(build(con, **kwargs))
