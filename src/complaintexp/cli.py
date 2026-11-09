"""`cexp` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ROOT
from .config import load_variants
from .schemas import load_dataset
from .store import DEFAULT_DB, connect

DEFAULT_DATASETS = [ROOT / "data" / "synthetic" / "sample.jsonl"]
_OPEN: list = []


def _db(args):
    """Open the results DB; `main` closes every connection opened this way."""
    con = connect(Path(args.db))
    _OPEN.append(con)
    return con


def _datasets(args) -> list:
    return load_dataset(*(Path(p) for p in (args.data or DEFAULT_DATASETS)))


def cmd_validate(args) -> int:
    cs = _datasets(args)
    bad = 0
    for c in cs:
        for p in c.pii:
            if p["text"] not in c.text:
                print(f"{c.id}: PII annotation not found in text: {p['text']!r}")
                bad += 1
    by_lang: dict[str, int] = {}
    for c in cs:
        by_lang[c.language] = by_lang.get(c.language, 0) + 1
    print(f"{len(cs)} complaints, by language: {by_lang}, {bad} annotation problems")
    return 1 if bad else 0


def cmd_run(args) -> int:
    from .pseudonymise import Pseudonymiser
    from .runner import run_variant

    variants = load_variants()
    complaints = _datasets(args)
    con = _db(args)
    ps = Pseudonymiser(use_ner=not args.no_ner)
    rc = 0
    for key in args.variants:
        if key not in variants:
            print(f"unknown variant {key}; have {sorted(variants)}")
            return 2
        stats = run_variant(variants[key], complaints, con, ps, limit=args.limit, warmup=args.warmup)
        print(f"{key}: {stats}")
        rc |= stats["errors"] > 0
    return int(bool(rc))


def cmd_redaction_eval(args) -> int:
    from .evaluate import redaction_recall, save_redaction_eval
    from .pseudonymise import Pseudonymiser

    ps = Pseudonymiser(use_ner=not args.no_ner)
    print(f"NER enabled: {ps.has_ner}")
    res = redaction_recall(_datasets(args), ps)
    for t, r in res.items():
        print(f"{t:12s} n={r['n']:3d} recall={r['recall']:.1%}")
        for m in r["missed"][:5]:
            print(f"    missed: {m}")
    if not args.no_save:
        files = ",".join(str(p) for p in (args.data or DEFAULT_DATASETS))
        ts = save_redaction_eval(_db(args), res, files, ps.has_ner)
        print(f"saved as redaction eval {ts} (latest one is used by `cexp report`)")
    return 0


def cmd_judge(args) -> int:
    from .judge import judge_variant

    con = _db(args)
    cs = _datasets(args)
    for key in args.variants:
        print(f"{key}: judged {judge_variant(key, cs, con)}")
    return 0


def cmd_export_blind(args) -> int:
    from .evaluate import export_blind

    con = _db(args)
    out = Path(args.out)
    n = export_blind(con, _datasets(args), out, out.with_suffix(".key.csv"), n=args.n, variants=args.variants)
    print(f"wrote {n} drafts to {out} and key to {out.with_suffix('.key.csv')} (keep the key from raters)")
    return 0


def cmd_import_scores(args) -> int:
    from .evaluate import ScoreSheetError, import_scores

    con = _db(args)
    try:
        n = import_scores(con, Path(args.scored), Path(args.key), args.rater)
    except ScoreSheetError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"imported {n} rows")
    return 0


def cmd_report(args) -> int:
    from .html_report import render_html
    from .report import build, render_markdown

    data = build(_db(args), complaints=_datasets(args),
                 include_real_text=args.include_real_text, include_text=not args.no_text)
    text = render_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    if args.html:
        out = Path(args.html)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_html(data), encoding="utf-8")
        print(f"wrote interactive report to {out}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.quiet:
        print(text)
    return 0


def cmd_generate(args) -> int:
    from .generate import generate

    v = load_variants()[args.variant]
    out = Path(args.out)
    n = generate(ROOT / "data" / "synthetic" / "scenarios.yaml", v, args.n, out, seed=args.seed, start_index=args.start)
    print(f"appended {n} synthetic complaints to {out} (status=needs_review: have an agent check realism)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cexp", description=__doc__)
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_arg(sp):
        sp.add_argument("--data", nargs="+", help="JSONL files (default: data/synthetic/sample.jsonl)")

    sp = sub.add_parser("validate", help="check dataset files and PII annotations")
    data_arg(sp)
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("run", help="run complaints through variants (resumable)")
    data_arg(sp)
    sp.add_argument("--variants", nargs="+", required=True, help="e.g. A B C D, or FAKE")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--no-ner", action="store_true", help="skip Presidio (regex redaction only)")
    sp.add_argument("--warmup", action="store_true",
                    help="send one untimed dummy request first (keeps model load out of latency)")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("redaction-eval", help="redaction recall per PII type (no model calls)")
    data_arg(sp)
    sp.add_argument("--no-ner", action="store_true")
    sp.add_argument("--no-save", action="store_true", help="print only, do not store for the report")
    sp.set_defaults(fn=cmd_redaction_eval)

    sp = sub.add_parser("judge", help="LLM-judge stored drafts")
    data_arg(sp)
    sp.add_argument("--variants", nargs="+", required=True)
    sp.set_defaults(fn=cmd_judge)

    sp = sub.add_parser("export-blind", help="shuffled, unlabeled sheet for human scoring")
    data_arg(sp)
    sp.add_argument("--out", default=str(ROOT / "results" / "blind_review.csv"))
    sp.add_argument("--n", type=int, default=60)
    sp.add_argument("--variants", nargs="+")
    sp.set_defaults(fn=cmd_export_blind)

    sp = sub.add_parser("import-scores", help="load a filled-in blind sheet")
    sp.add_argument("scored")
    sp.add_argument("--key", required=True)
    sp.add_argument("--rater", required=True)
    sp.set_defaults(fn=cmd_import_scores)

    sp = sub.add_parser("report", help="gates, operations, cost, quality, privacy and effort summary")
    data_arg(sp)
    sp.add_argument("--out", help="write the Markdown report here")
    sp.add_argument("--html", nargs="?", const=str(ROOT / "results" / "report.html"),
                    help="write the interactive HTML report (default path: results/report.html)")
    sp.add_argument("--json", help="write the raw report data as JSON")
    sp.add_argument("--include-real-text", action="store_true",
                    help="embed real complaint text and drafts in the HTML/JSON (synthetic is always included)")
    sp.add_argument("--no-text", action="store_true", help="leave all complaint text out of the HTML")
    sp.add_argument("-q", "--quiet", action="store_true", help="do not print the Markdown")
    sp.set_defaults(fn=cmd_report)

    sp = sub.add_parser("generate", help="LLM-generate synthetic complaints for agent review")
    sp.add_argument("--variant", default="A")
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--start", type=int, default=1, help="first id number, e.g. 101 to append")
    sp.add_argument("--out", default=str(ROOT / "data" / "synthetic" / "generated.jsonl"))
    sp.set_defaults(fn=cmd_generate)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    finally:
        while _OPEN:
            _OPEN.pop().close()


if __name__ == "__main__":
    sys.exit(main())
