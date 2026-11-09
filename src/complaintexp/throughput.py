"""Replies per hour at a given concurrency (plan, cost model: "Throughput").

`cexp throughput` sends the same draft-tier complaints in parallel and records
wall-clock time and per-request latency. Answers are not stored as runs. The
report takes, per variant, the highest replies/hour whose p95 latency still
passes the latency gate: that is the capacity of one server.
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import Variant
from .report import _pct
from .runner import Context, RealDataBlocked, check_allowed, process
from .schemas import Complaint


def measure(
    v: Variant,
    complaints: list[Complaint],
    con: sqlite3.Connection,
    concurrency: list[int],
    n: int = 40,
    variants: dict[str, Variant] | None = None,
    clients: dict | None = None,
    pseudonymiser=None,
    log=print,
) -> list[dict]:
    """One row per concurrency level. Uses tier 3 and 4 complaints, cycled up to `n` requests."""
    pool = [c for c in complaints if c.tier in (3, 4)]
    allowed = []
    for c in pool:
        try:
            check_allowed(v, c)
            allowed.append(c)
        except RealDataBlocked:
            continue
    if not allowed:
        raise ValueError("no tier 3/4 complaints this variant may receive")
    batch = [allowed[i % len(allowed)] for i in range(n)]
    ctx = Context({**(variants or {}), v.key: v}, pseudonymiser, clients)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for conc in concurrency:
        def one(c):
            t0 = time.perf_counter()
            try:
                process(v, c, ctx)
                return time.perf_counter() - t0
            except Exception as e:
                log(f"throughput {v.key} @ {conc}: {type(e).__name__}: {e}")
                return None

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            lat = list(ex.map(one, batch))
        wall = time.perf_counter() - t0
        ok = [x for x in lat if x is not None]
        row = dict(ts=ts, variant=v.key, concurrency=conc, n=n, n_ok=len(ok), wall_s=wall,
                   replies_per_hour=len(ok) / wall * 3600 if wall > 0 else None,
                   p50_s=_pct(ok, 0.5), p95_s=_pct(ok, 0.95))
        con.execute(f"INSERT OR REPLACE INTO throughput ({','.join(row)}) VALUES ({','.join('?' * len(row))})",
                    list(row.values()))
        con.commit()
        out.append(row)
    return out
