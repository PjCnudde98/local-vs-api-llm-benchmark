"""SQLite store (store.py): schema and migration of older result databases."""

import sqlite3

from complaintexp.report import build, frontier
from complaintexp.store import connect


def test_v02_database_gets_new_columns_and_keeps_its_rows(tmp_path):
    p = tmp_path / "old.sqlite"
    old = sqlite3.connect(p)
    old.executescript(
        "CREATE TABLE runs (variant TEXT NOT NULL, complaint_id TEXT NOT NULL, source TEXT NOT NULL, "
        "language TEXT, type TEXT, difficulty TEXT, ts TEXT NOT NULL, model_id TEXT, prompt_sent TEXT, "
        "output_raw TEXT, draft TEXT, agent_note TEXT, parse_ok INTEGER, latency_s REAL, prompt_tokens INTEGER, "
        "completion_tokens INTEGER, error TEXT, PRIMARY KEY (variant, complaint_id));"
        "CREATE TABLE judgements (variant TEXT NOT NULL, complaint_id TEXT NOT NULL, judge_model TEXT NOT NULL, "
        "scores_json TEXT NOT NULL, hallucination INTEGER, PRIMARY KEY (variant, complaint_id, judge_model));"
        "INSERT INTO runs VALUES ('C','s1','synthetic','nl','delivery','angry','2026-11-06','m','p','o','d','n',1,"
        "20.0,100,50,NULL);"
    )
    old.commit()
    old.close()
    con = connect(p)
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    assert {"tier", "route", "metrics_json", "pii_leaked"} <= cols
    assert "extra_json" in {r[1] for r in con.execute("PRAGMA table_info(judgements)")}
    assert con.execute("SELECT latency_s, tier FROM runs").fetchone()[:] == (20.0, None)
    d = build(con, hours_csv=None, variants={})
    assert d["operations"][0]["variant"] == "C" and d["frontier"]["cells"] == []  # no tier: not on the ladder
    con.close()


def test_connect_twice_is_idempotent(tmp_path):
    connect(tmp_path / "a.sqlite").close()
    con = connect(tmp_path / "a.sqlite")
    assert frontier(con, {"tiers": {}}, {"scale": [1, 5]})["cells"] == []
    con.close()
