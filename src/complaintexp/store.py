"""SQLite store for runs, judgements, human scores, redaction evals and throughput."""

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
    tier INTEGER,
    route TEXT,                -- local | pseudonymise | abstract | raw | own_tenant | hybrid
    model_key TEXT,
    size_class TEXT,           -- cloud | xl | large | medium | small
    reasoning TEXT,            -- default | on | off
    hardware TEXT,
    routed_to TEXT,            -- hybrid: the variant that answered
    ts TEXT NOT NULL,
    model_id TEXT,
    prompt_sent TEXT,          -- exactly what left the process (pseudonymised/abstracted for cloud routes)
    preprocess_output TEXT,    -- abstract route: the summary; hybrid: the router's answer
    preprocess_latency_s REAL, -- pseudonymisation + abstraction or routing, included in latency_s
    output_raw TEXT,
    output_json TEXT,          -- parsed answer, placeholders restored
    draft TEXT,                -- readable answer (the reply for draft tiers), placeholders restored
    agent_note TEXT,
    parse_ok INTEGER,
    metrics_json TEXT,         -- automatic scores (tiers 1, 2, 5)
    tool_calls INTEGER,
    valid_tool_calls INTEGER,
    trace_json TEXT,           -- tier 5: the conversation as sent
    pii_annotated INTEGER,     -- cloud routes: annotated PII values in the complaint
    pii_leaked INTEGER,        -- ... of which still visible in what was sent
    latency_s REAL,            -- end to end: preprocessing + every model call
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
    extra_json TEXT,             -- tier 6: {"consistent": 0/1, "missed_facts": [...], "n_key_facts": n}
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
CREATE TABLE IF NOT EXISTS summaries (
    key TEXT PRIMARY KEY,         -- sha256 of abstractor + prompt + pseudonymised complaint
    abstractor TEXT NOT NULL,
    complaint_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    latency_s REAL
);
CREATE TABLE IF NOT EXISTS throughput (
    ts TEXT NOT NULL,
    variant TEXT NOT NULL,
    concurrency INTEGER NOT NULL,
    n INTEGER NOT NULL,
    n_ok INTEGER NOT NULL,
    wall_s REAL NOT NULL,
    replies_per_hour REAL,
    p50_s REAL,
    p95_s REAL,
    PRIMARY KEY (ts, variant, concurrency)
);
"""

# Columns added after the first release; older databases get them on connect.
_ADDED = {
    "runs": ["tier INTEGER", "route TEXT", "model_key TEXT", "size_class TEXT", "reasoning TEXT",
             "hardware TEXT", "routed_to TEXT", "preprocess_output TEXT", "preprocess_latency_s REAL",
             "output_json TEXT", "metrics_json TEXT", "tool_calls INTEGER", "valid_tool_calls INTEGER",
             "trace_json TEXT", "pii_annotated INTEGER", "pii_leaked INTEGER"],
    "judgements": ["extra_json TEXT"],
}


def _migrate(con: sqlite3.Connection) -> None:
    for table, cols in _ADDED.items():
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for col in cols:
            if col.split()[0] not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
    con.commit()


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _migrate(con)
    return con
