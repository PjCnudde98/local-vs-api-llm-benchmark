"""Interactive HTML view (html_report.py + report_template.html)."""

import json
import re

from complaintexp.html_report import render_html
from complaintexp.report import build
from complaintexp.schemas import Complaint

from .conftest import insert_run, insert_score
from .test_report import COSTS


def _data(con, **kw):
    insert_run(con, "C", "s1")
    insert_run(con, "D", "s1", latency=3)
    insert_run(con, "C", "r1", source="real")
    insert_score(con, "judgements", "C", "s1", 4)
    con.execute("UPDATE runs SET draft='</script><script>alert(1)</script> & <b>' WHERE complaint_id='s1'")
    con.commit()
    cs = [Complaint(id="s1", source="synthetic", language="en", text="Synthetic <text>", tier=3),
          Complaint(id="r1", source="real", language="nl", text="REAL CUSTOMER TEXT", tier=3)]
    return build(con, costs=COSTS, complaints=cs, hours_csv=None, **kw)


def _embedded(html):
    m = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', html, re.DOTALL)
    return json.loads(m.group(1))


def test_data_is_embedded_and_round_trips(con):
    d = _data(con)
    got = _embedded(render_html(d))
    assert got["meta"]["n_runs"] == 3 and got["records"][0]["draft"] == d["records"][0]["draft"]


def test_hostile_text_cannot_close_the_script_tag(con):
    html = render_html(_data(con))
    assert html.count("</script>") == 2  # the JSON block and the app script, nothing injected


def test_page_is_self_contained(con):
    html = render_html(_data(con))
    assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
    assert "@import" not in html


def test_real_text_is_not_in_the_page_by_default(con):
    assert "REAL CUSTOMER TEXT" not in render_html(_data(con))


def test_real_text_in_page_when_asked(con):
    assert "REAL CUSTOMER TEXT" in render_html(_data(con, include_real_text=True))


def test_page_supports_dark_mode_and_has_title(con):
    html = render_html(_data(con), title="My <run>")
    assert "<title>My &lt;run&gt;</title>" in html
    assert "prefers-color-scheme: dark" in html and 'data-theme="dark"' in html


def test_page_carries_model_names(con):
    d = _data(con, variants={})
    assert d["names"] == {"C": "m-C", "D": "m-D"}
    assert _embedded(render_html(d))["names"] == d["names"]


def test_page_carries_the_capability_frontier(con):
    d = _data(con)
    got = _embedded(render_html(d))
    assert {c["variant"] for c in got["frontier"]["cells"]} == {"C", "D"}
    assert 'id="c-frontier"' in render_html(d)
