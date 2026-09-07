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
from dataclasses import replace
from pathlib import Path

from filing.agent.verify import GUARD_MODES
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
    if args.live:
        # The response cache off, on purpose. Every prompt in the frozen set has
        # already been answered by the full run, so a cached capture reports
        # llm_calls=0 and sub-millisecond model spans -- true, because
        # `llm_calls` counts hosted HTTP calls and a free re-run must not claim
        # a day's quota, but read out of the file it says the agent makes no
        # model calls. The committed trace is a live one so its costs and its
        # latencies are the real ones.
        cfg = cfg.model_copy(update={"cache_enabled": False})
    ec = get_config(args.config).resolved(cfg)
    if args.question:
        # A probe rather than a frozen-set question. Useful for the branches the
        # eval set does not reliably reach -- the repair loop above all, since a
        # question that repairs is by definition one the first attempt failed,
        # and the graph is meant to make those rare. The note written into the
        # file says which kind of question produced it, because a trace whose
        # provenance is unclear is worth less than no trace.
        qid, question = args.qid or "probe", args.question
    else:
        path = dataset.dataset_path(Path(cfg.data_dir), args.version)
        questions = {q.id: q for q in dataset.read(path)}
        try:
            found = questions[args.qid]
        except KeyError:
            print(f"no question {args.qid!r} in {path}", file=sys.stderr)
            return 1
        qid, question = found.id, found.question

    backend = build_backend(cfg, ec.chat_backend or None) if ec.generate else None
    tools = build_agent_tools(cfg, backend=backend, config=ec)
    state, spans = capture_question(question, tools=tools, qid=qid)

    source = "a question written to reach one branch" if args.question else "the frozen set v1.0"
    note = (
        f"One question from {source}, through the M5 agent graph, captured "
        "in-process by filing.agent.trace and regenerable with "
        f"`python -m filing.eval trace --qid {qid}"
        f"{' --question ...' if args.question else ''}"
        f"{' --live' if args.live else ''}`."
    )
    out = write_trace(
        Path(args.out), question=question, qid=qid, state=state, spans=spans, note=note
    )
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
    if args.guard:
        # The override renames the config as well as changing it. A results
        # file is named after its config and the fingerprint is computed from
        # it, so a flag that changed behaviour while leaving the name alone
        # would write one experiment over another's file -- the same failure
        # the `.partial` suffix exists to prevent. `off` also switches the
        # verifier off, because a guard mode is only meaningful when there is
        # a verdict for it to act on.
        ec = replace(
            ec,
            name=f"{ec.name}-guard-{args.guard}",
            guard=args.guard,
            verify=args.guard != "off",
        )
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
    p.add_argument(
        "--guard",
        choices=list(GUARD_MODES),
        default=None,
        help="override the config's verification guard; runs under a renamed config",
    )
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("trace", help="export one question's span tree to docs/")
    p.add_argument("--qid", default="num-001", help="question id from the frozen set")
    p.add_argument("--question", default="", help="trace this text instead of a frozen question")
    p.add_argument("--config", default="agent", choices=sorted(CONFIGS))
    p.add_argument("--version", default=dataset.DATASET_VERSION)
    p.add_argument("--out", default="docs/trace_example.json")
    p.add_argument(
        "--live",
        action="store_true",
        help="bypass the response cache so the trace records real calls and real latencies",
    )
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
