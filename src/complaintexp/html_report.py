"""Self-contained interactive HTML view of the report data.

The page embeds the `report.build()` dict as JSON and does all filtering and
charting in the browser with plain JS and inline SVG: no CDN, no network, so
it opens offline and nothing leaves the machine.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")


def _json_for_script(data: dict) -> str:
    """JSON that is safe inside <script type="application/json">."""
    s = json.dumps(data, ensure_ascii=False, default=str)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_html(data: dict, title: str = "Complaint LLM experiment") -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    return page.replace("__TITLE__", escape(title)).replace("__DATA__", _json_for_script(data))
