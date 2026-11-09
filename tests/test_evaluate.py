"""Redaction recall and the blind human-scoring round trip (evaluate.py)."""

import csv

import pytest

from complaintexp.evaluate import (ScoreSheetError, _leaks, export_blind, import_scores, redaction_recall,
                                   save_redaction_eval)
from complaintexp.schemas import Complaint

from .conftest import insert_run


# -- redaction recall ---------------------------------------------------------
def test_redaction_recall_regex_only_misses_names(sample, ps):
    res = redaction_recall(sample, ps)
    for t in ("EMAIL", "ORDER", "IBAN", "PHONE", "NATIONAL_ID"):
        assert res[t]["recall"] == 1.0, t
    assert res["PERSON"]["recall"] == 0.0  # regex-only cannot see names: NER needed


def test_recall_counts_and_misses(ps):
    c = Complaint(id="c1", source="synthetic", language="en", text="I am Jan Peeters, a@b.com",
                  pii=[{"type": "PERSON", "text": "Jan Peeters"}, {"type": "EMAIL", "text": "a@b.com"}])
    res = redaction_recall([c], ps)
    assert res["PERSON"] == {"n": 1, "leaked": 1, "recall": 0.0, "missed": ["c1: Jan Peeters"]}
    assert res["EMAIL"]["leaked"] == 0


def test_complaints_without_annotations_are_ignored(ps):
    assert redaction_recall([Complaint(id="x", source="synthetic", language="en", text="a@b.com")], ps) == {}


@pytest.mark.parametrize(
    "value, redacted, leaks",
    [
        ("Jan Peeters", "Dear <PERSON_1>", False),
        ("Jan Peeters", "Dear jan peeters", True),             # case-insensitive
        ("Kerkstraat 12, 9000 Gent", "<LOCATION_1>, 9000 Gent", True),  # partial: postcode+city left
        ("Kerkstraat 12, 9000 Gent", "Kerkstraat <X>", True),  # street name left
        ("Jan Peeters", "Jan <PERSON_1>", False),              # short tokens (<4 chars) are not counted
    ],
)
def test_leak_detection(value, redacted, leaks):
    assert _leaks(value, redacted) is leaks


@pytest.mark.ner
def test_ner_recall_on_sample_person_names(sample, ner_ps):
    assert redaction_recall(sample, ner_ps)["PERSON"]["recall"] == 1.0


def test_redaction_eval_is_saved_for_the_report(con, sample, ps):
    ts = save_redaction_eval(con, redaction_recall(sample, ps), "sample.jsonl", ps.has_ner)
    rows = con.execute("SELECT * FROM redaction_eval WHERE ts=?", (ts,)).fetchall()
    assert {r["pii_type"] for r in rows} >= {"EMAIL", "PERSON"} and all(r["has_ner"] == 0 for r in rows)


# -- blind export / import ----------------------------------------------------
@pytest.fixture
def two_variants(con, sample):
    for c in sample:
        insert_run(con, "C", c.id)
        insert_run(con, "D", c.id)
    con.execute("UPDATE runs SET draft = '**Dear** ' || variant || ' customer' ")
    con.commit()
    return con


def _read(p):
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_blind_sheet_hides_variants_and_key_maps_back(two_variants, sample, tmp_path):
    out, key = tmp_path / "s.csv", tmp_path / "s.key.csv"
    n = export_blind(two_variants, sample, out, key, n=3)
    sheet, k = _read(out), _read(key)
    assert n == 6 and len(sheet) == 6 and len(k) == 6
    assert "variant" not in sheet[0] and all("**" not in r["draft"] for r in sheet)
    assert {r["blind_id"] for r in sheet} == {r["blind_id"] for r in k}
    assert {r["variant"] for r in k} == {"C", "D"}


def test_blind_export_is_deterministic(two_variants, sample, tmp_path):
    export_blind(two_variants, sample, tmp_path / "a.csv", tmp_path / "a.key.csv", n=4)
    export_blind(two_variants, sample, tmp_path / "b.csv", tmp_path / "b.key.csv", n=4)
    assert (tmp_path / "a.key.csv").read_text() == (tmp_path / "b.key.csv").read_text()


def test_blind_export_variant_filter(two_variants, sample, tmp_path):
    export_blind(two_variants, sample, tmp_path / "s.csv", tmp_path / "k.csv", n=10, variants=["D"])
    assert {r["variant"] for r in _read(tmp_path / "k.csv")} == {"D"}


def _fill(sheet_path, values):
    rows = _read(sheet_path)
    crit = [c for c in rows[0] if c not in ("blind_id", "language", "complaint", "draft", "hallucination(0/1)", "edit_minutes")]
    for r, v in zip(rows, values):
        if v is None:
            continue
        for c in crit:
            r[c] = v.get("score", 4)
        r["hallucination(0/1)"] = v.get("h", "0")
        r["edit_minutes"] = v.get("edit", "1.5")
    with sheet_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_scored_sheet_round_trip(two_variants, sample, tmp_path):
    out, key = tmp_path / "s.csv", tmp_path / "s.key.csv"
    export_blind(two_variants, sample, out, key, n=2)
    _fill(out, [{}, {}, None, {"score": 2, "h": "1", "edit": ""}])
    assert import_scores(two_variants, out, key, "anna") == 3  # unscored row skipped
    rows = two_variants.execute("SELECT * FROM human_scores ORDER BY hallucination").fetchall()
    assert len(rows) == 3 and rows[-1]["hallucination"] == 1 and rows[-1]["edit_minutes"] is None


def test_reimport_replaces_same_rater(two_variants, sample, tmp_path):
    out, key = tmp_path / "s.csv", tmp_path / "s.key.csv"
    export_blind(two_variants, sample, out, key, n=1)
    _fill(out, [{}, {}])
    import_scores(two_variants, out, key, "anna")
    import_scores(two_variants, out, key, "anna")
    import_scores(two_variants, out, key, "ben")
    assert two_variants.execute("SELECT COUNT(*) FROM human_scores").fetchone()[0] == 4


@pytest.mark.parametrize(
    "bad, message",
    [({"score": 7}, "outside"), ({"score": 0}, "outside"), ({"h": "yes"}, "hallucination"),
     ({"edit": "-1"}, "edit_minutes"), ({"edit": "two"}, "edit_minutes"), ({"score": "x"}, "non-numeric")],
)
def test_invalid_sheet_is_rejected_whole(two_variants, sample, tmp_path, bad, message):
    out, key = tmp_path / "s.csv", tmp_path / "s.key.csv"
    export_blind(two_variants, sample, out, key, n=1)
    _fill(out, [{}, bad])
    with pytest.raises(ScoreSheetError, match=message):
        import_scores(two_variants, out, key, "anna")
    assert two_variants.execute("SELECT COUNT(*) FROM human_scores").fetchone()[0] == 0


def test_unknown_blind_id_is_rejected(two_variants, sample, tmp_path):
    out, key = tmp_path / "s.csv", tmp_path / "s.key.csv"
    export_blind(two_variants, sample, out, key, n=1)
    _fill(out, [{}])
    out.write_text(out.read_text(encoding="utf-8").replace("B0001", "B9999"), encoding="utf-8")
    with pytest.raises(ScoreSheetError, match="not in key"):
        import_scores(two_variants, out, key, "anna")


@pytest.mark.ner
def test_ner_recall_on_sample_is_complete_for_every_type(sample, ner_ps):
    res = redaction_recall(sample, ner_ps)
    assert {t: r["recall"] for t, r in res.items() if r["recall"] < 1} == {}
