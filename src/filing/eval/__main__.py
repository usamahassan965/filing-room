"""``python -m filing.eval`` -- the entry point the gate is written against.

Kept out of the main ``filing`` CLI on purpose. Evaluation is a different kind
of command from ingestion: it is slow, it is budgeted, and it is the thing whose
invocation ends up quoted in a README and a definition of done. A separate
module means ``python -m filing.eval run --config baseline`` keeps working even
if the CLI's command tree is rearranged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from filing.config import settings
from filing.eval import dataset, metrics
from filing.eval.runner import CONFIGS, get_config, results_dir, run, title_for


def _cmd_dataset(args: argparse.Namespace) -> int:
    cfg = settings()
    path = dataset.dataset_path(Path(cfg.data_dir), args.version)
    questions = dataset.read(path)
    print(f"{path}")
    print(f"  sha256   {dataset.fingerprint(path)}")
    print(f"  counts   {dataset.counts(questions)}")
    spans = [s for q in questions for s in q.spans]
    print(f"  gold     {len(spans)} spans over {len({s.accn for s in spans})} filings")
    if args.show:
        for q in questions[: args.show]:
            print(f"\n  {q.id}  [{q.slice} -> {q.route}]  {q.question}")
            for s in q.spans[:2]:
                print(f"      {s.accn} {s.char_start}:{s.char_end}  {s.quote[:100]}...")
    return 0


def _cmd_build_naive(args: argparse.Namespace) -> int:
    from filing.eval.naive import build_naive_chunks, build_naive_index

    cfg = settings()
    chunks = build_naive_chunks(cfg, rebuild=args.rebuild)
    print(
        f"chunks: {chunks.chunks:,} over {chunks.filings} filings "
        f"({chunks.chars:,} chars, {chunks.seconds:.1f}s)"
    )
    if chunks.quarantined:
        print(f"  quarantined: {len(chunks.quarantined)}")
    if args.chunks_only:
        return 0

    def tick(done: int, total: int) -> None:
        print(f"  embedded {done:,}/{total:,}", end="\r", file=sys.stderr)

    report = build_naive_index(cfg, rebuild=args.rebuild, limit=args.limit, on_batch=tick)
    print(
        f"\nindex: {report.collection}\n"
        f"  upserted {report.upserted:,}, already present {report.already_indexed:,}, "
        f"{report.embed_http_calls} HTTP calls in {report.seconds / 60:.1f} min"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Re-parse every cited filing and check each span still says what it said.

    The datasheet claims this was done before the set was frozen. A claim about
    verification that cannot itself be run is worth about as much as no
    verification, so it is a command rather than a line in a script's history.
    """
    from filing.eval.authoring import verify_spans

    cfg = settings()
    path = dataset.dataset_path(Path(cfg.data_dir), args.version)
    questions = dataset.read(path)
    problems = verify_spans(cfg, questions)
    spans = [s for q in questions for s in q.spans]
    if problems:
        print(f"{len(problems)} problem(s):")
        for line in problems[:40]:
            print(" ", line)
        return 1
    filings = len({s.accn for s in spans})
    print(f"clean: {len(spans)} spans over {filings} filings re-read from disk")
    return 0


def _cmd_depth(args: argparse.Namespace) -> int:
    """Report how far down the ranking the evidence actually sits."""
    from filing.eval import depth

    cfg = settings()
    slices = depth.measure(cfg, depth=args.depth, version=args.version)
    print(depth.to_markdown(slices, title=f"dense only, top {args.depth}"))
    for s in slices.values():
        print(
            f"{s.name}: gold chunk in top {args.depth} for {s.found}/{s.n}; "
            f"right filing for {s.filing_found}/{s.n}"
        )
    path = depth.write(cfg, slices, root=results_dir(cfg))
    print(f"wrote {path}")
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    """Run one question of the frozen set and write its span tree to docs/.

    A command rather than a note saying which button in Phoenix to press. The
    gate's artefact is a file, and a file nobody can regenerate is a screenshot.
    """
    from filing.agent.trace import capture_question, write_trace
    from filing.eval.runner import build_agent_tools
    from filing.llm.factory import build_backend

    cfg = settings()
    ec = get_config(args.config).resolved(cfg)
    path = dataset.dataset_path(Path(cfg.data_dir), args.version)
    questions = {q.id: q for q in dataset.read(path)}
    try:
        q = questions[args.qid]
    except KeyError:
        print(f"no question {args.qid!r} in {path}", file=sys.stderr)
        return 1

    backend = build_backend(cfg, ec.chat_backend or None) if ec.generate else None
    tools = build_agent_tools(cfg, backend=backend, config=ec)
    state, spans = capture_question(q.question, tools=tools, qid=q.id)

    out = write_trace(Path(args.out), question=q.question, qid=q.id, state=state, spans=spans)
    names = [s["name"] for s in spans]
    print(f"{len(spans)} spans: {' -> '.join(n.removeprefix('agent.') for n in names)}")
    print(
        f"route={state.get('route')} repairs={state.get('repairs')} "
        f"llm_calls={state.get('llm_calls')} refused={state.get('refused')}"
    )
    print(f"wrote {out}")
    return 0 if spans else 1


def _cmd_run(args: argparse.Namespace) -> int:
    cfg = settings()
    ec = get_config(args.config)
    slices = tuple(args.slice) if args.slice else dataset.SLICES

    def tick(i: int, total: int, outcome) -> None:  # noqa: ANN001
        mark = "!" if outcome.error else ("-" if outcome.refused else ".")
        print(f"  {i:>3}/{total} {outcome.qid} {mark}", end="\r", file=sys.stderr)

    report = run(
        cfg,
        config=ec,
        limit=args.limit,
        slices=slices,
        use_cache=not args.no_cache,
        on_question=tick,
    )
    print(file=sys.stderr)
    print(metrics.to_markdown(report.card, title=title_for(report.config, partial=not report.full)))
    print(
        f"{report.answered} answered, {report.from_cache} from cache, "
        f"{report.llm_calls} LLM calls, {report.errors} errors, {report.seconds:.1f}s"
    )
    print(f"fingerprint {report.fingerprint[:16]}")
    if not report.full:
        n = sum(report.counts.values())
        print(f"PARTIAL: {n} of the frozen set. Not the gate's run; not comparable to one.")
    # report.written, not a path rebuilt here -- the run decides where it went,
    # and a message naming a file nobody wrote is the bug it is meant to prevent.
    print(f"wrote {report.written}")
    return 1 if report.errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m filing.eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="answer the frozen set under one configuration")
    p.add_argument("--config", default="baseline", choices=sorted(CONFIGS))
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--slice", action="append", choices=list(dataset.SLICES))
    p.add_argument("--no-cache", action="store_true", help="re-answer even if cached")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("trace", help="export one question's span tree to docs/")
    p.add_argument("--qid", default="num-001", help="question id from the frozen set")
    p.add_argument("--config", default="agent", choices=sorted(CONFIGS))
    p.add_argument("--version", default=dataset.DATASET_VERSION)
    p.add_argument("--out", default="docs/trace_example.json")
    p.set_defaults(func=_cmd_trace)

    p = sub.add_parser("dataset", help="describe the frozen question set")
    p.add_argument("--version", default=dataset.DATASET_VERSION)
    p.add_argument("--show", type=int, default=0, help="print the first N questions")
    p.set_defaults(func=_cmd_dataset)

    p = sub.add_parser("depth", help="how deep the evidence sits in the ranking")
    p.add_argument("--depth", type=int, default=500)
    p.add_argument("--version", default=dataset.DATASET_VERSION)
    p.set_defaults(func=_cmd_depth)

    p = sub.add_parser("verify", help="re-read every gold span from the filing on disk")
    p.add_argument("--version", default=dataset.DATASET_VERSION)
    p.set_defaults(func=_cmd_verify)

    p = sub.add_parser("build-naive", help="chunk and index the corpus for the baseline")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--chunks-only", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=_cmd_build_naive)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
