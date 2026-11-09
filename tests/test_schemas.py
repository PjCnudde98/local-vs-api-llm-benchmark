"""Dataset loading and validation (schemas.load_dataset)."""

import json

import pytest

from complaintexp.schemas import Complaint, load_dataset

from .conftest import SAMPLE


def _write(tmp_path, *rows, name="d.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8")
    return p


def _row(**kw):
    return {"id": "x1", "source": "synthetic", "language": "en", "text": "hello", **kw}


def test_sample_dataset_loads_and_annotations_match_text():
    cs = load_dataset(SAMPLE)
    assert len(cs) >= 6
    for c in cs:
        for p in c.pii:
            assert p["text"] in c.text, (c.id, p)


def test_sample_covers_all_languages():
    assert {c.language for c in load_dataset(SAMPLE)} == {"nl", "fr", "en"}


def test_unknown_fields_are_ignored(tmp_path):
    (c,) = load_dataset(_write(tmp_path, _row(status="needs_review", extra=1)))
    assert c.id == "x1"


def test_blank_lines_are_skipped(tmp_path):
    p = _write(tmp_path, _row(id="a"), "", "   ", _row(id="b"))
    assert [c.id for c in load_dataset(p)] == ["a", "b"]


def test_multiple_files_are_concatenated(tmp_path):
    a = _write(tmp_path, _row(id="a"), name="a.jsonl")
    b = _write(tmp_path, _row(id="b"), name="b.jsonl")
    assert [c.id for c in load_dataset(a, b)] == ["a", "b"]


def test_duplicate_ids_across_files_are_rejected(tmp_path):
    a = _write(tmp_path, _row(), name="a.jsonl")
    b = _write(tmp_path, _row(), name="b.jsonl")
    with pytest.raises(ValueError, match="duplicate complaint id"):
        load_dataset(a, b)


@pytest.mark.parametrize(
    "row, message",
    [
        (_row(source="scraped"), "bad source"),
        (_row(language="de"), "bad language"),
        (_row(text=""), "missing required"),
        ({"id": "x", "source": "synthetic", "language": "en"}, "missing required"),
        (_row(pii=[{"type": "PERSON"}]), "pii must be"),
        (_row(pii="Jan"), "pii must be"),
    ],
)
def test_invalid_records_are_rejected_with_location(tmp_path, row, message):
    with pytest.raises(ValueError, match=message) as e:
        load_dataset(_write(tmp_path, row))
    assert "d.jsonl:1" in str(e.value)


def test_broken_json_reports_file_and_line(tmp_path):
    with pytest.raises(ValueError, match=r"invalid JSON .*d\.jsonl:2"):
        load_dataset(_write(tmp_path, _row(), "{not json"))


def test_from_dict_defaults():
    c = Complaint.from_dict({"id": "x", "source": "real", "language": "nl", "text": "t"})
    assert c.pii == [] and c.facts == {} and c.gold_reply is None
