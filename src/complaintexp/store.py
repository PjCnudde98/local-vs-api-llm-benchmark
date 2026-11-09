"""SQLite store for runs, judgements and human scores."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import ROOT

DEFAULT_DB = ROOT / "results" / "experiment.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    variant TEXT NOT NULL,
    complaint_id TEXT NOT NULL,
    source TEXT NOT NULL,
    language TEXT, type TEXT, difficulty TEXT,
    ts TEXT NOT NULL,
    model_id TEXT,
    prompt_sent TEXT,          -- exactly what left the process (pseudonymised for cloud B)
    output_raw TEXT,
    draft TEXT,                -- placeholders restored
    agent_note TEXT,
    parse_ok INTEGER,
    latency_s REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT,
    PRIMARY KEY (variant, complaint_id)
);
CREATE TABLE IF NOT EXISTS judgements (
    variant TEXT NOT NULL,
    complaint_id TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    scores_json TEXT NOT NULL,   -- {criterion: 1-5}
    hallucination INTEGER,
    PRIMARY KEY (variant, complaint_id, judge_model)
);
CREATE TABLE IF NOT EXISTS human_scores (
    variant TEXT NOT NULL,
    complaint_id TEXT NOT NULL,
    rater TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    hallucination INTEGER,
    edit_minutes REAL,
    PRIMARY KEY (variant, complaint_id, rater)
);
CREATE TABLE IF NOT EXISTS redaction_eval (
    ts TEXT NOT NULL,
    dataset TEXT NOT NULL,        -- comma-separated input files
    has_ner INTEGER NOT NULL,
    pii_type TEXT NOT NULL,
    n INTEGER NOT NULL,
    leaked INTEGER NOT NULL,
    missed_json TEXT NOT NULL,    -- ["complaint_id: value", ...]
    PRIMARY KEY (ts, pii_type)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con
