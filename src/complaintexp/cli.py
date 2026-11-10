"""`cexp` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ROOT
from .config import is_judge, load_variants, select
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


def _select(args, variants) -> list[str] | None:
    try:
        return select(variants, args.variants)
    except KeyError as e:
        print(e.args[0])
        return None


def cmd_validate(args) -> int:
    cs = _datasets(args)
    bad = 0
    for c in cs:
        for p in c.pii:
            if p["text"] not in c.text:
                print(f"{c.id}: PII annotation not found in text: {p['text']!r}")
                bad += 1
    count = lambda key: {k: sum(getattr(c, key) == k for c in cs) for k in sorted({getattr(c, key) for c in cs})}  # noqa: E731
    print(f"{len(cs)} complaints, by tier: {count('tier')}, by language: {count('language')}, "
          f"{bad} annotation problems")
    return 1 if bad else 0


def cmd_variants(args) -> int:
    vs = load_variants()
    keys = select(vs, args.variants) if args.variants else [k for k in vs if not is_judge(k)]
    print(f"{'variant':38s} {'size':7s} {'route':13s} {'real data':9s} model")
    for k in keys:
        v = vs[k]
        target = v.model if v.route != "hybrid" else f"router {v.hybrid.get('router')} -> " \
            f"{v.hybrid.get('easy')} | {v.hybrid.get('hard')}"
        print(f"{k:38s} {v.size_class:7s} {v.route:13s} {'yes' if v.allow_real_data and not v.cloud else 'no':9s} "
              f"{target}")
    print(f"\n{len(keys)} variants. Select with globs, e.g. --variants 'local/*' 'pseudonymise/*'")
    return 0


def cmd_run(args) -> int:
    from .pseudonymise import Pseudonymiser
    from .runner import run_variant

    variants = load_variants()
    keys = _select(args, variants)
    if keys is None:
        return 2
    complaints = _datasets(args)
    con = _db(args)
    ps = Pseudonymiser(use_ner=not args.no_ner)
    rc = 0
    for key in keys:
        stats = run_variant(variants[key], complaints, con, ps, limit=args.limit, warmup=args.warmup,
                            variants=variants, tiers=args.tiers)
        print(f"{key}: {stats}")
        rc |= stats["errors"] > 0
    return int(bool(rc))


def cmd_throughput(args) -> int:
    from .pseudonymise import Pseudonymiser
    from .throughput import measure

    variants = load_variants()
    keys = _select(args, variants)
    if keys is None:
        return 2
    con = _db(args)
    ps = Pseudonymiser(use_ner=not args.no_ner)
    for key in keys:
        for r in measure(variants[key], _datasets(args), con, args.concurrency, args.n, variants=variants,
                         pseudonymiser=ps):
            rph = "n/a" if r["replies_per_hour"] is None else f"{r['replies_per_hour']:.0f}"
            p95 = "n/a" if r["p95_s"] is None else f"{r['p95_s']:.1f}"
            print(f"{key} @ {r['concurrency']:>3}: {rph} replies/hour, p95 {p95} s ({r['n_ok']}/{r['n']} ok)")
    return 0


def cmd_rescore(args) -> int:
    """Recompute automatic scores (tiers 1, 2, 5) from stored answers, e.g. after fixing a gold label."""
    from .tiers import Parsed, autoscore

    con = _db(args)
    by_id = {c.id: c for c in _datasets(args)}
    n = 0
    for r in con.execute("SELECT variant, complaint_id, output_json, parse_ok FROM runs "
                         "WHERE error IS NULL AND tier IN (1, 2, 5)").fetchall():
        c = by_id.get(r["complaint_id"])
        if c is None:
            continue
        parsed = Parsed(bool(r["parse_ok"]), json.loads(r["output_json"] or "{}"))
        con.execute("UPDATE runs SET metrics_json=? WHERE variant=? AND complaint_id=?",
                    (json.dumps(autoscore(c.tier, c, parsed)), r["variant"], r["complaint_id"]))
        n += 1
    con.commit()
    print(f"rescored {n} answers")
    return 0


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

    variants = load_variants()
    keys = _select(args, variants)
    if keys is None:
        return 2
    con = _db(args)
    cs = _datasets(args)
    for key in keys:
        print(f"{key}: judged {judge_variant(key, cs, con, variants=variants, judge_key=args.judge)}")
    return 0


def cmd_export_blind(args) -> int:
    from .evaluate import export_blind

    con = _db(args)
    out = Path(args.out)
    keys = select(load_variants(), args.variants) if args.variants else None
    n = export_blind(con, _datasets(args), out, out.with_suffix(".key.csv"), n=args.n, variants=keys,
                     max_drafts=args.drafts)
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
        Path(args.json).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if not args.quiet:
        print(text)
    return 0


def cmd_generate(args) -> int:
    from .generate import generate

    v = load_variants()[args.variant]
    out = Path(args.out)
    tiers = args.tiers or [1, 2, 3, 4, 5, 6]
    total = 0
    for t in tiers:
        n = generate(ROOT / "data" / "synthetic" / "scenarios.yaml", v, args.n, out, seed=args.seed,
                     start_index=args.start, tier=t)
        print(f"tier {t}: appended {n} synthetic complaints to {out}")
        total += n
    print(f"{total} complaints with status=needs_review: have an agent check realism and every gold answer")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cexp", description=__doc__)
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_arg(sp):
        sp.add_argument("--data", nargs="+", help="JSONL files (default: data/synthetic/sample.jsonl)")

    def variants_arg(sp, required=True):
        sp.add_argument("--variants", nargs="+", required=required,
                        help="variant keys or globs, e.g. 'local/*' 'pseudonymise/haiku-5.5' FAKE")

    sp = sub.add_parser("validate", help="check dataset files, gold answers and PII annotations")
    data_arg(sp)
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("variants", help="list the lineup: every model x route x reasoning mode")
    variants_arg(sp, required=False)
    sp.set_defaults(fn=cmd_variants)

    sp = sub.add_parser("run", help="run complaints through variants (resumable)")
    data_arg(sp)
    variants_arg(sp)
    sp.add_argument("--tiers", nargs="+", type=int, help="only these tiers (default: all)")
    sp.add_argument("--limit", type=int)
    sp.add_argument("--no-ner", action="store_true", help="skip Presidio (regex redaction only)")
    sp.add_argument("--warmup", action="store_true",
                    help="send one untimed dummy request first (keeps model load out of latency)")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("throughput", help="replies/hour at several concurrency levels (capacity for the cost model)")
    data_arg(sp)
    variants_arg(sp)
    sp.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    sp.add_argument("--n", type=int, default=40, help="requests per concurrency level")
    sp.add_argument("--no-ner", action="store_true")
    sp.set_defaults(fn=cmd_throughput)

    sp = sub.add_parser("rescore", help="recompute automatic tier 1/2/5 scores after editing gold answers")
    data_arg(sp)
    sp.set_defaults(fn=cmd_rescore)

    sp = sub.add_parser("redaction-eval", help="redaction recall per PII type (no model calls)")
    data_arg(sp)
    sp.add_argument("--no-ner", action="store_true")
    sp.add_argument("--no-save", action="store_true", help="print only, do not store for the report")
    sp.set_defaults(fn=cmd_redaction_eval)

    sp = sub.add_parser("judge", help="LLM-judge stored drafts (tiers 3, 4, 6)")
    data_arg(sp)
    variants_arg(sp)
    sp.add_argument("--judge", default="JUDGE", help="judge key from config/variants.yaml (JUDGE, JUDGE_GPT, JUDGE_OPENAI)")
    sp.set_defaults(fn=cmd_judge)

    sp = sub.add_parser("export-blind", help="shuffled, unlabeled sheet for human scoring (tiers 3, 4, 6)")
    data_arg(sp)
    sp.add_argument("--out", default=str(ROOT / "results" / "blind_review.csv"))
    sp.add_argument("--n", type=int, default=60, help="complaints (every variant's draft of each)")
    sp.add_argument("--drafts", type=int, help="cap the sheet at this many drafts, balanced over variants "
                    "(a small human overlap sample to check the judges)")
    variants_arg(sp, required=False)
    sp.set_defaults(fn=cmd_export_blind)

    sp = sub.add_parser("import-scores", help="load a filled-in blind sheet")
    sp.add_argument("scored")
    sp.add_argument("--key", required=True)
    sp.add_argument("--rater", required=True)
    sp.set_defaults(fn=cmd_import_scores)

    sp = sub.add_parser("report", help="frontier, gates, operations, cost, quality, privacy and effort")
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

    sp = sub.add_parser("generate", help="LLM-generate synthetic complaints per tier for agent review")
    sp.add_argument("--variant", default="raw/sonnet-5.5", help="the model that writes them (synthetic only)")
    sp.add_argument("--tiers", nargs="+", type=int, help="tiers to generate (default: all six)")
    sp.add_argument("--n", type=int, default=25, help="complaints per tier")
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
