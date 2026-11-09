"""Redaction recall, blind human-scoring export/import."""

from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import load_rubric
from .prompting import normalise_for_blind
from .pseudonymise import Pseudonymiser
from .schemas import Complaint
from .tiers import DRAFT_TIERS


# -- privacy: redaction recall ------------------------------------------------
def leaks(value: str, redacted: str) -> bool:
    """True if the value, or any identifying part of it, survives redaction.

    Partial leaks count: "Kerkstraat 12, 9000 Gent" with only "Gent" redacted still
    leaks the street. Identifying parts are tokens of 4+ characters; shorter ones
    ("Jan", "12") are skipped because they match ordinary text.
    """
    low = redacted.lower()
    if value.lower() in low:
        return True
    tokens = re.findall(r"[\w.@+-]+", value.lower())
    return any(len(t) >= 4 and t in low for t in tokens)


def redaction_recall(complaints: list[Complaint], ps: Pseudonymiser) -> dict[str, dict]:
    """Per PII type: share of hand-annotated values that no longer appear after redaction.

    A value counts as leaked if it, or any identifying part of it, is still
    present (case-insensitive) in the redacted text; see `leaks`. Only
    complaints with a `pii` annotation are scored.
    """
    total: dict[str, int] = defaultdict(int)
    leaked: dict[str, int] = defaultdict(int)
    misses: dict[str, list[str]] = defaultdict(list)
    for c in complaints:
        if not c.pii:
            continue
        redacted, _ = ps.redact(c.text, c.language)
        for ann in c.pii:
            t, value = ann["type"], ann["text"]
            total[t] += 1
            if leaks(value, redacted):
                leaked[t] += 1
                misses[t].append(f"{c.id}: {value}")
    return {
        t: {
            "n": total[t],
            "leaked": leaked[t],
            "recall": 1 - leaked[t] / total[t],
            "missed": misses[t],
        }
        for t in sorted(total)
    }


def save_redaction_eval(
    con: sqlite3.Connection, result: dict[str, dict], dataset: str, has_ner: bool
) -> str:
    """Store one redaction-eval run so `cexp report` can gate on it. Returns its timestamp."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for t, r in result.items():
        con.execute(
            "INSERT OR REPLACE INTO redaction_eval VALUES (?,?,?,?,?,?,?)",
            (ts, dataset, int(has_ner), t, r["n"], r["leaked"], json.dumps(r["missed"], ensure_ascii=False)),
        )
    con.commit()
    return ts


# -- quality: blind human scoring ---------------------------------------------
def export_blind(
    con: sqlite3.Connection,
    complaints: list[Complaint],
    out_csv: Path,
    key_csv: Path,
    n: int = 60,
    variants: list[str] | None = None,
    seed: int = 42,
) -> int:
    """Write a shuffled review sheet (no variant labels) plus a separate key file.

    Only draft tiers (3, 4, 6) are exported: tiers 1, 2 and 5 are scored automatically.
    """
    rubric = load_rubric()
    by_id = {c.id: c for c in complaints}
    rng = random.Random(seed)
    vfilter = f" AND (tier IS NULL OR tier IN ({','.join(str(t) for t in DRAFT_TIERS)}))"
    vargs: list = []
    if variants:
        vfilter += f" AND variant IN ({','.join('?' * len(variants))})"
        vargs = list(variants)
    ids = sorted(
        r[0] for r in con.execute(
            "SELECT DISTINCT complaint_id FROM runs WHERE parse_ok=1" + vfilter, vargs
        )
    )
    ids = [i for i in ids if i in by_id]
    rng.shuffle(ids)
    ids = ids[:n]
    rows = []
    for cid in ids:
        q = "SELECT variant, draft FROM runs WHERE complaint_id=? AND parse_ok=1" + vfilter
        for r in con.execute(q + " ORDER BY variant", [cid, *vargs]):
            rows.append((cid, r["variant"], normalise_for_blind(r["draft"])))
    rng.shuffle(rows)

    criteria = list(rubric["criteria"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f, key_csv.open(
        "w", newline="", encoding="utf-8"
    ) as k:
        w = csv.writer(f)
        kw = csv.writer(k)
        w.writerow(["blind_id", "language", "tier", "complaint", "draft", *criteria, "hallucination(0/1)", "edit_minutes"])
        kw.writerow(["blind_id", "complaint_id", "variant"])
        for i, (cid, variant, draft) in enumerate(rows, 1):
            bid = f"B{i:04d}"
            c = by_id[cid]
            w.writerow([bid, c.language, c.tier, c.text, draft, *[""] * (len(criteria) + 2)])
            kw.writerow([bid, cid, variant])
    return len(rows)


class ScoreSheetError(ValueError):
    pass


def import_scores(con: sqlite3.Connection, scored_csv: Path, key_csv: Path, rater: str) -> int:
    """Load a filled-in blind sheet. Rows without all criteria scored are skipped.

    The whole sheet is validated before anything is written: an unknown
    blind_id, a score outside the rubric scale, a hallucination flag other than
    0/1 or a negative edit time aborts the import with every problem listed.
    """
    with key_csv.open(encoding="utf-8") as f:
        key = {r["blind_id"]: r for r in csv.DictReader(f)}
    rubric = load_rubric()
    criteria = list(rubric["criteria"])
    lo, hi = rubric["scale"]
    rows, problems = [], []
    with scored_csv.open(encoding="utf-8") as f:
        for line, r in enumerate(csv.DictReader(f), 2):
            if not all((r.get(c) or "").strip() for c in criteria):
                continue  # unscored row
            bid = r.get("blind_id", "")
            if bid not in key:
                problems.append(f"line {line}: blind_id {bid!r} not in key file")
                continue
            try:
                scores = {c: float(r[c]) for c in criteria}
            except ValueError:
                problems.append(f"line {line}: non-numeric score")
                continue
            bad = [f"{c}={v:g}" for c, v in scores.items() if not lo <= v <= hi]
            if bad:
                problems.append(f"line {line}: score outside {lo}-{hi}: {', '.join(bad)}")
            h = (r.get("hallucination(0/1)") or "").strip()
            if h not in ("", "0", "1"):
                problems.append(f"line {line}: hallucination must be 0 or 1, got {h!r}")
            em = (r.get("edit_minutes") or "").strip()
            try:
                em_v = float(em) if em else None
            except ValueError:
                em_v = -1
            if em_v is not None and em_v < 0:
                problems.append(f"line {line}: edit_minutes must be a number >= 0, got {em!r}")
            k = key[bid]
            rows.append((k["variant"], k["complaint_id"], rater, json.dumps(scores),
                         int(h) if h in ("0", "1") else None, em_v))
    if problems:
        raise ScoreSheetError("score sheet rejected, nothing imported:\n  " + "\n  ".join(problems))
    con.executemany("INSERT OR REPLACE INTO human_scores VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)
