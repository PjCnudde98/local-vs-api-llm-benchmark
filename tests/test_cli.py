"""End-to-end through `cexp` with the FAKE variant: no network, temp DB."""

import json

import pytest

from complaintexp.cli import main

from .conftest import SAMPLE


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "cli.sqlite")


def test_validate_sample_is_clean(capsys):
    assert main(["validate", "--data", str(SAMPLE)]) == 0
    assert "0 annotation problems" in capsys.readouterr().out


def test_validate_flags_annotation_not_in_text(tmp_path, capsys):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "x", "source": "synthetic", "language": "en", "text": "hello",
                             "pii": [{"type": "PERSON", "text": "Jan"}]}) + "\n", encoding="utf-8")
    assert main(["validate", "--data", str(p)]) == 1
    assert "not found in text" in capsys.readouterr().out


def test_unknown_variant(db, capsys):
    assert main(["--db", db, "run", "--variants", "NOPE", "--no-ner"]) == 2


def test_full_dry_run_pipeline(db, tmp_path, capsys):
    assert main(["--db", db, "run", "--variants", "FAKE", "--no-ner", "--warmup"]) == 0
    assert main(["--db", db, "redaction-eval", "--no-ner"]) == 0
    out = tmp_path / "blind.csv"
    assert main(["--db", db, "export-blind", "--n", "3", "--out", str(out)]) == 0
    assert out.exists() and out.with_suffix(".key.csv").exists()
    html, js, md = tmp_path / "r.html", tmp_path / "r.json", tmp_path / "r.md"
    assert main(["--db", db, "report", "-q", "--html", str(html), "--json", str(js), "--out", str(md)]) == 0
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["meta"]["variants"] == ["FAKE"] and data["redaction"] is not None
    assert "report-data" in html.read_text(encoding="utf-8") and "## Gates" in md.read_text(encoding="utf-8")


def test_rerun_skips_finished_work(db, capsys):
    main(["--db", db, "run", "--variants", "FAKE", "--no-ner"])
    main(["--db", db, "run", "--variants", "FAKE", "--no-ner"])
    assert "'skipped_done': 6" in capsys.readouterr().out


def test_bad_score_sheet_returns_error(db, tmp_path, capsys):
    main(["--db", db, "run", "--variants", "FAKE", "--no-ner"])
    out = tmp_path / "blind.csv"
    main(["--db", db, "export-blind", "--n", "1", "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    row = ["B0001", "en", "c", "d"] + ["9"] * (len(head) - 6) + ["0", "1"]
    out.write_text(lines[0] + "\n" + ",".join(row) + "\n", encoding="utf-8")
    assert main(["--db", db, "import-scores", str(out), "--key", str(out.with_suffix(".key.csv")),
                 "--rater", "t"]) == 1
    assert "outside" in capsys.readouterr().err
