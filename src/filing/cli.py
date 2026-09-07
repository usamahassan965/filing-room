"""Command line for the rig.

    filing smoke    the M0 gate -- five checks, exit code 0 or 1
    filing probe    which registered model IDs are still live
    filing cache    inspect or clear the call cache
    filing ingest   fetch the corpus declared in universe.yaml
    filing corpus   the M1 gate -- five checks, exit code 0 or 1
    filing facts    build the structured store from the filings already on disk
    filing numbers  the M2 gate -- five checks, exit code 0 or 1
    filing chunks   parse and chunk the filings already on disk
    filing index    embed the narrative chunks and build the BM25 index
    filing graph    extract the entity graph from those same chunks
    filing text     the M3 gate -- six checks, exit code 0 or 1
    filing failures the M6 failure gallery, counted by a Phoenix filter
    filing serve    the M7 API -- /ask, with the evidence in the response

No gate in this project advances on a claim, so the gate commands assert rather
than print: ``smoke`` fails loudly if the cache is not deduplicating requests or
if the reranker does not put the obviously-relevant passage first, and
``corpus`` fails if a second ingest would download so much as one file.
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from filing.config import MODEL_REGISTRY, PROJECT_ROOT, Backend, model_for, settings
from filing.ingest.corpus import ingest as corpus_ingest
from filing.ingest.corpus import write_corpus_doc
from filing.ingest.edgar import EdgarForbidden
from filing.ingest.manifest import Manifest
from filing.ingest.universe import UniverseError, load_universe
from filing.llm.cache import CallCache
from filing.llm.errors import MissingCredentials, ModelUnavailable
from filing.llm.factory import build_backend
from filing.stores.facts import FactsStore, build_facts, duplicate_keys
from filing.stores.questions import check_derivations, run_questions
from filing.stores.verify import verify_against_filings
from filing.tracing import flush_tracing, setup_tracing, span_count, span_names

app = typer.Typer(add_completion=False, help="Filing Room -- agentic RAG over SEC filings.")
console = Console()

# The rerank check needs a question with an unambiguous answer among the
# passages, or "did it work" becomes a matter of taste.
_RERANK_QUERY = "What did the company say about supply chain risk?"
_RERANK_PASSAGES = [
    "Net revenues for fiscal 2024 increased 8% to $394.3 billion compared with fiscal 2023.",
    "Our results could be harmed if our suppliers cannot obtain components, or if "
    "manufacturing is disrupted at facilities concentrated in a single region.",
    "The board declared a quarterly dividend of $0.25 per share payable in November.",
]
_RERANK_EXPECTED = 1  # the supply-chain passage


def _run_marker() -> str:
    """A token that makes this run's chat and embed payloads unlike any other's.

    Without it the gate grades the cache. The cache key is a hash of the
    payload, so the second `filing smoke` of the day replays the first one's
    answers: chat, embed and dedup all pass, the footer reads `http calls: 0,
    cache hits: 4`, and a revoked key, a retired model ID or an expired free
    tier still reports 5/5. That is the one failure a smoke test exists to
    catch, and it was the one failure it could not see.

    So the first chat and embed of every run carry a fresh marker and therefore
    must miss, which is asserted rather than hoped for; check 4 then repeats
    the *marked* prompt, so dedup is still proved -- on a payload this run put
    there, not one left over from last week. The cost is two live calls per
    smoke run, which is the price of the gate meaning anything.
    """
    return uuid.uuid4().hex[:8]


def _check(ok: bool, label: str, detail: str = "") -> bool:
    mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {mark}  {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))
    return ok


@app.command()
def smoke(
    backend: Annotated[str | None, typer.Option(help="Override LLM_BACKEND for this run.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """M0 gate: chat, embed, rerank, cache dedup, and a >= 3-span trace."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = settings()
    chosen: Backend = backend or cfg.llm_backend  # type: ignore[assignment]

    console.rule(f"[bold]filing smoke[/bold]  backend={chosen}")
    tracing_live = setup_tracing(cfg)
    try:
        client = build_backend(cfg, backend=chosen)  # type: ignore[arg-type]
    except MissingCredentials as exc:
        # The first thing a new clone hits. A traceback here teaches nothing.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("role")
    table.add_column("model id")
    for role in ("chat", "chat_fast", "embed", "rerank"):
        table.add_row(role, model_for(role, chosen).id)
    console.print(table)
    console.print()

    results: list[bool] = []
    try:
        marker = _run_marker()

        # 1 -- chat, and it has to be *this run's* chat: a fresh marker in the
        # payload means a cache hit here is a bug, not a saving.
        prompt = [
            {
                "role": "system",
                "content": f"Answer in one short sentence. Ignore this run marker: {marker}.",
            },
            {"role": "user", "content": "What is a 10-K filing?"},
        ]
        hits_before = client.usage().cache_hits
        answer = client.chat(prompt, role="chat_fast", max_tokens=80)
        live = client.usage().cache_hits == hits_before
        results.append(
            _check(
                bool(answer.strip()) and live,
                "chat",
                (answer[:70].replace("\n", " ") + ("" if live else "  [CACHED -- not a check]")),
            )
        )

        # 2 -- embed (query and passage go through different code paths), same
        # marker, same reason.
        hits_before = client.usage().cache_hits
        vectors = client.embed([f"annual report risk factors {marker}"], input_type="passage")
        live = client.usage().cache_hits == hits_before
        dim = len(vectors[0]) if vectors else 0
        expected_dim = model_for("embed", chosen).dim
        ok_dim = dim > 0 and (expected_dim is None or dim == expected_dim)
        results.append(
            _check(ok_dim and live, "embed", f"dim={dim} expected={expected_dim} live={live}")
        )

        # 3 -- rerank, and it has to be *right*, not merely non-empty. No
        # marker here: reranking is a local forward pass on every backend, so
        # a cache hit cannot hide a dead credential or a retired model ID,
        # and re-running the cross-encoder to learn that would just be slow.
        rankings = client.rerank(_RERANK_QUERY, _RERANK_PASSAGES, top_n=3)
        top = rankings[0].index if rankings else -1
        results.append(
            _check(
                top == _RERANK_EXPECTED,
                "rerank",
                f"order={[r.index for r in rankings]} top={top} expected={_RERANK_EXPECTED}",
            )
        )

        # 4 -- cache: the same prompt must not reach the network twice. The
        # prompt is check 1's, marker and all, so this proves dedup on an entry
        # this process wrote seconds ago rather than on whatever was already
        # warm.
        before = client.http_calls
        hits_before = client.usage().cache_hits
        repeat = client.chat(prompt, role="chat_fast", max_tokens=80)
        delta = client.http_calls - before
        hit = client.usage().cache_hits > hits_before
        results.append(
            _check(
                delta == 0 and hit and repeat == answer,
                "cache dedup",
                f"http_calls delta={delta} (must be 0) hit={hit}",
            )
        )

        # 5 -- tracing. Flush first: a batched span that never left the process
        # is not a trace, and this command exits long before the batch timer
        # would have fired on its own.
        flushed = flush_tracing()
        results.append(
            _check(
                tracing_live and flushed and span_count() >= 3,
                "tracing",
                f"{span_count()} spans flushed={flushed}: "
                f"{', '.join(sorted(set(span_names())))[:60]}",
            )
        )
    except ModelUnavailable as exc:
        # The registry's whole purpose is that this is a one-line fix, so say so
        # instead of unrolling a stack that points at the HTTP client. Providers
        # retire model IDs on their own schedule; that is not a crash.
        console.print(f"[red]{exc}[/red]")
        results.append(False)
    finally:
        usage = client.usage()
        close = getattr(client, "close", None)
        if callable(close):
            close()
        flush_tracing()

    console.print()
    console.print(
        f"[dim]http calls: {usage.http_calls}   cache hits: {usage.cache_hits}   "
        f"tokens in/out: {usage.prompt_tokens}/{usage.completion_tokens}[/dim]"
    )
    if tracing_live:
        console.print(f"[dim]trace: {cfg.phoenix_endpoint} (project {cfg.phoenix_project})[/dim]")
    else:
        console.print("[yellow]tracing not exporting -- run `docker compose up -d`[/yellow]")

    passed = sum(results)
    console.rule(
        f"[bold green]{passed}/{len(results)} passed[/bold green]"
        if all(results)
        else f"[bold red]{passed}/{len(results)} passed[/bold red]"
    )
    raise typer.Exit(0 if all(results) else 1)


@app.command()
def probe(
    backend: Annotated[str | None, typer.Option(help="Which registry to probe.")] = None,
) -> None:
    """Check every model ID in the registry, primaries and alternates alike.

    Run this when something 404s. Providers rename and retire model IDs without
    warning, so the fix is to copy a live ID from this table into MODEL_REGISTRY.
    """
    cfg = settings()
    chosen: Backend = backend or cfg.llm_backend  # type: ignore[assignment]
    # A cached probe would report a dead model as live on the second run.
    client = build_backend(cfg.model_copy(update={"cache_enabled": False}), backend=chosen)  # type: ignore[arg-type]

    table = Table(title=f"{chosen} model registry", header_style="bold")
    table.add_column("role")
    table.add_column("model id")
    table.add_column("in use")
    table.add_column("status")

    for role, spec in MODEL_REGISTRY[chosen].items():
        # Probing a local model downloads it. Probing its alternates would
        # download several hundred MB to answer a question nobody asked.
        candidates = (spec.id,) if spec.local else spec.candidates
        for candidate in candidates:
            table.add_row(
                role,
                candidate,
                "*" if candidate == spec.id else "",
                _probe_one(client, role, candidate),
            )
    console.print(table)
    console.print("[dim]* = the ID currently in MODEL_REGISTRY[/dim]")
    close = getattr(client, "close", None)
    if callable(close):
        close()
    flush_tracing()


def _probe_one(client, role: str, model_id: str) -> str:  # noqa: ANN001
    """One minimal call per ID. Costs a handful of tokens, saves an afternoon."""
    try:
        if role.startswith("chat"):
            client.chat(
                [{"role": "user", "content": "ok"}], role=role, model_id=model_id, max_tokens=4
            )
        elif role == "embed":
            client.embed(["ok"], input_type="query", model_id=model_id)
        else:
            client.rerank("ok", ["ok"], top_n=1, model_id=model_id)
        return "[green]live[/green]"
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return f"[red]{type(exc).__name__}[/red] [dim]{str(exc)[:48]}[/dim]"


@app.command()
def cache(
    clear: Annotated[bool, typer.Option("--clear", help="Delete every cached call.")] = False,
) -> None:
    """Inspect or clear the content-addressed call cache."""
    cfg = settings()
    c = CallCache(cfg.cache_dir, enabled=True)
    if clear:
        n = c.clear()
        console.print(f"cleared {n} entries from {cfg.cache_dir}")
    else:
        console.print(f"{len(c)} entries in {cfg.cache_dir}")
    c.close()


@app.command()
def ingest(
    forms: Annotated[
        str | None, typer.Option(help="Comma-separated form override, e.g. '10-K'.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(help="Only the first N companies (development).")
    ] = None,
    facts_only: Annotated[bool, typer.Option("--facts-only")] = False,
    filings_only: Annotated[bool, typer.Option("--filings-only")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch the corpus declared in universe.yaml. Safe to re-run; resumes."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = settings()
    if not cfg.sec_user_agent:
        # Cheaper to say so now than to let EDGAR answer twenty 403s.
        console.print(
            "[red]SEC_USER_AGENT is not set.[/red] EDGAR rejects anonymous clients.\n"
            'Add to .env:  SEC_USER_AGENT="Your Name your@email.com"'
        )
        raise typer.Exit(1)

    console.rule("[bold]filing ingest[/bold]")
    wanted = {f.strip() for f in forms.split(",")} if forms else None
    try:
        report = _run_ingest(
            cfg,
            forms=wanted,
            limit=limit,
            do_filings=not facts_only,
            do_facts=not filings_only,
        )
    except (UniverseError, EdgarForbidden) as exc:
        # Both are configuration problems with a named fix. A traceback would
        # bury the one sentence that resolves them.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None

    _print_ingest_report(report)
    raise typer.Exit(1 if report.errors else 0)


def _run_ingest(cfg, **kwargs):  # noqa: ANN001, ANN201
    """Shared by `ingest` and the idempotency check in `corpus`."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    with Progress(
        TextColumn("[bold]{task.fields[ticker]:>5}[/bold]"),
        BarColumn(bar_width=28),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.fields[msg]}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as bar:
        universe = load_universe(cfg=cfg)
        total = min(len(universe.companies), kwargs.get("limit") or len(universe.companies))
        task = bar.add_task("ingest", total=total, ticker="", msg="")

        def on_progress(ticker: str, message: str) -> None:
            bar.update(task, advance=1, ticker=ticker, msg=message)

        return corpus_ingest(cfg, universe=universe, progress=on_progress, **kwargs)


def _print_ingest_report(report) -> None:  # noqa: ANN001
    console.print()
    console.print(
        f"[dim]companies: {report.companies}   "
        f"filings: {report.filings_downloaded} new / {report.filings_skipped} cached   "
        f"facts: {report.facts_downloaded} new / {report.facts_skipped} cached[/dim]"
    )
    console.print(
        f"[dim]edgar requests: {report.requests}   "
        f"downloaded: {report.bytes_downloaded / 1e6:,.1f} MB   "
        f"elapsed: {report.elapsed_s:,.0f}s[/dim]"
    )
    if report.pruned:
        console.print(
            f"[yellow]pruned {report.pruned} manifest row(s) for CIKs no longer "
            "in universe.yaml; their files are left on disk[/yellow]"
        )
    for ticker, message in report.errors:
        console.print(f"[red]{ticker}: {message}[/red]")


# The gate's floor, from the M1 definition of done. Not a target -- the corpus
# is far larger than this -- but the number below which the later gates do not
# have enough documents to mean anything.
MIN_FILINGS = 100

# Per-company floors. The window spans five fiscal years, so a healthy company
# lands on 5-6 10-Ks and 15-16 10-Qs; these sit well under that on purpose.
# They exist because a corpus-wide total hides a per-company hole: XOM resolved
# to a post-reorganisation CIK with no history, contributed zero documents, and
# every aggregate check still passed. A silent zero in one company is worth more
# than a total, because it is the peer set of every comparison it appears in.
MIN_ANNUAL_PER_COMPANY = 4
MIN_QUARTERLY_PER_COMPANY = 8


@app.command()
def corpus(
    offline: Annotated[
        bool, typer.Option("--offline", help="Skip the re-ingest check; report from the manifest.")
    ] = False,
    write_docs: Annotated[bool, typer.Option("--write-docs/--no-write-docs")] = True,
) -> None:
    """M1 gate: a complete corpus, in the manifest, that a second run leaves alone."""
    cfg = settings()
    console.rule("[bold]filing corpus[/bold]")
    if not cfg.manifest_path.exists():
        console.print(f"[red]no manifest at {cfg.manifest_path} -- run `filing ingest`[/red]")
        raise typer.Exit(1)

    universe = load_universe(cfg=cfg)
    expected = len(universe.companies)
    results: list[bool] = []

    # 1 -- idempotency. The only honest way to check "a second run downloads
    # zero files" is to do the second run: it re-reads every submissions index
    # from EDGAR and re-decides, so a broken resume shows up here rather than
    # in a comment claiming it works.
    if offline:
        console.print("  [dim]SKIP  idempotent re-run (--offline)[/dim]")
    else:
        report = _run_ingest(cfg, forms=None, limit=None, do_filings=True, do_facts=True)
        results.append(
            _check(
                report.downloaded_nothing and not report.errors,
                "idempotent",
                f"second run fetched {report.filings_downloaded} filings, "
                f"{report.facts_downloaded} facts (both must be 0)",
            )
        )

    with Manifest(cfg.manifest_path, cfg.data_dir) as manifest:
        stats = manifest.stats()
        missing = manifest.missing_files()
        rows = manifest.per_company()

    # 2 -- enough documents for the later gates to mean anything
    results.append(
        _check(
            stats.filings >= MIN_FILINGS,
            "corpus size",
            f"{stats.filings} filings ({_forms(stats.by_form)}), minimum {MIN_FILINGS}",
        )
    )

    # 3 -- every manifest row still has its bytes. The manifest is the source of
    # truth, which is only a useful property if it does not lie.
    results.append(
        _check(not missing, "manifest matches disk", f"{len(missing)} rows with no file")
    )
    for key, path in missing[:5]:
        console.print(f"        [red]missing[/red] {key} -> {path}")

    # 4 -- companyfacts for everyone. A company without XBRL cannot answer a
    # numeric question, so a partial pull here silently caps M2's coverage.
    results.append(
        _check(
            stats.companyfacts == expected,
            "companyfacts",
            f"{stats.companyfacts}/{expected} companies",
        )
    )
    # 4b -- and every company individually. A total of 387 says nothing about
    # whether one company contributed none of them.
    thin = [
        (sector, ticker, annual, quarterly, facts)
        for sector, ticker, _cik, annual, quarterly, *_rest, facts in rows
        if annual < MIN_ANNUAL_PER_COMPANY or quarterly < MIN_QUARTERLY_PER_COMPANY or not facts
    ]
    results.append(
        _check(
            not thin,
            "per-company coverage",
            f"{len(rows) - len(thin)}/{len(rows)} companies with "
            f"{MIN_ANNUAL_PER_COMPANY}+ 10-K, {MIN_QUARTERLY_PER_COMPANY}+ 10-Q, facts",
        )
    )
    for sector, ticker, annual, quarterly, facts in thin:
        console.print(
            f"        [red]thin[/red] {sector}/{ticker}: "
            f"{annual} 10-K, {quarterly} 10-Q, facts={'yes' if facts else 'NO'}"
        )

    # 5 -- the corpus is on record, with a parse-time estimate M3 can budget from
    if write_docs:
        doc = write_corpus_doc(cfg)
        results.append(_check(doc.exists(), "docs/corpus.md", str(doc.relative_to(PROJECT_ROOT))))

    console.print()
    console.print(
        f"[dim]{stats.companies} companies   {stats.filings} filings   "
        f"{(stats.filing_bytes + stats.facts_bytes) / 1e9:,.2f} GB on disk[/dim]"
    )

    passed = sum(results)
    console.rule(
        f"[bold green]{passed}/{len(results)} passed[/bold green]"
        if all(results)
        else f"[bold red]{passed}/{len(results)} passed[/bold red]"
    )
    raise typer.Exit(0 if all(results) else 1)


# The verification rate a passing store has to clear. Set from measurement
# rather than from the plan's assumed 48/50: the observed rate is 49/50, and the
# single miss is a fact that is genuinely not in its filing's primary document.
# Pfizer's $15.0B TCJA transition-tax liability lives only in that 10-Q's XBRL
# attachment -- the words "transition tax" appear nowhere in the document. So
# the floor sits one below what was measured, which leaves room for the sample
# to shift and none for a scale error to slip through.
MIN_VERIFIED = 48
VERIFY_SAMPLE = 50


@app.command()
def facts() -> None:
    """Build data/facts.duckdb from the companyfacts payloads already on disk.

    No network. The store is a pure function of the JSON the manifest points at,
    which is why it is safe to rebuild at any time -- and why the gate below can
    afford to rebuild before it checks anything.
    """
    cfg = settings()
    if not cfg.manifest_path.exists():
        console.print(f"[red]no manifest at {cfg.manifest_path} -- run `filing ingest`[/red]")
        raise typer.Exit(1)

    console.rule("[bold]filing facts[/bold]")
    _print_facts_report(build_facts(cfg))


def _print_facts_report(report) -> None:  # noqa: ANN001
    table = Table(box=None, pad_edge=False)
    table.add_column("", style="dim")
    table.add_column("", justify="right")
    table.add_row("companies", f"{report.companies:,}")
    table.add_row("facts", f"{report.facts:,}")
    table.add_row("concepts", f"{report.concepts:,}")
    table.add_row("restated period-keys", f"{report.restatements:,}")
    table.add_row("skipped (no date)", f"{report.skipped_no_date:,}")
    table.add_row("skipped (no value)", f"{report.skipped_no_value:,}")
    if report.metrics is not None:
        table.add_row(
            "metrics resolved", f"{report.metrics.resolved:,}/{report.metrics.expected:,}"
        )
        table.add_row("annual rows", f"{report.metrics.annual_rows:,}")
    table.add_row("elapsed", f"{report.elapsed_s:,.1f}s")
    console.print(table)
    console.print(
        "[dim]"
        + "  ".join(f"{span}={n:,}" for span, n in sorted(report.by_span.items()))
        + "[/dim]"
    )


@app.command()
def numbers(
    sample: Annotated[
        int, typer.Option("--sample", help="Facts to check against source filings.")
    ] = VERIFY_SAMPLE,
    rebuild: Annotated[
        bool, typer.Option("--rebuild/--no-rebuild", help="Rebuild the store before checking.")
    ] = True,
) -> None:
    """M2 gate: a store whose numbers survive being checked against the filings.

    Every check here asserts something about the data, not about the code. The
    store is rebuilt first by default, so a pass means the source as it stands
    produces a passing store rather than that some earlier build did.
    """
    cfg = settings()
    if not cfg.manifest_path.exists():
        console.print(f"[red]no manifest at {cfg.manifest_path} -- run `filing ingest`[/red]")
        raise typer.Exit(1)

    console.rule("[bold]filing numbers[/bold]")
    report = None
    if rebuild or not cfg.facts_path.exists():
        report = build_facts(cfg)
        _print_facts_report(report)
        console.print()

    results: list[bool] = []
    with FactsStore(cfg.facts_path) as store:
        con = store.con

        # 1 -- the identity tuple is unique. Asserted with GROUP BY rather than
        # declared as a constraint: SQL treats NULLs as distinct and
        # period_start is NULL for every instant, so a declared UNIQUE would
        # silently exempt 38% of the table while reading as though it covered it.
        dupes = duplicate_keys(con)
        results.append(_check(dupes == 0, "identity tuple unique", f"{dupes:,} duplicate keys"))

        # 2 -- the numbers are the filings' numbers. The only check in this
        # project that consults a source outside the pipeline that produced the
        # data under test: companyfacts JSON in, filing HTML out.
        verified = verify_against_filings(con, cfg.data_dir, n=sample)
        floor = MIN_VERIFIED * verified.checked / VERIFY_SAMPLE
        results.append(
            _check(
                verified.found >= floor,
                "facts found in source filings",
                f"{verified.found}/{verified.checked} across {verified.docs_read} documents",
            )
        )
        for miss in verified.misses:
            console.print(
                f"        [yellow]not found[/yellow] {miss.ticker} {miss.form} "
                f"{miss.period_end}  {miss.tag}  {miss.val:,.0f}"
            )

        # 3 -- ten numeric questions, answered by SQL with no model involved.
        answers = run_questions(con)
        results.append(
            _check(
                not answers.failures,
                "numeric questions answered",
                f"{answers.passed}/{answers.total} correct",
            )
        )
        for failed in answers.failures:
            console.print(f"        [red]wrong[/red] {failed.question.id}: {failed.detail}")

        # 4 -- the derived columns agree with the same arithmetic done by hand.
        derived = check_derivations(con)
        results.append(
            _check(
                derived.passed,
                "derivations match manual calculation",
                f"{derived.values_checked:,} values over {derived.rows_checked} fiscal years "
                f"for {', '.join(derived.companies)}",
            )
        )
        for bad in derived.mismatches[:10]:
            console.print(
                f"        [red]differs[/red] {bad.ticker} {bad.period_end} {bad.column}: "
                f"stored {bad.stored} vs {bad.recomputed}"
            )

        # 5 -- every metric that should be universal is, for every company.
        # Non-universal metrics are reported but never failed on: Walmart has no
        # R&D line because Walmart does no R&D, and that is an answer, not a gap.
        if report is not None and report.metrics is not None:
            unmet = [
                (metric, n)
                for metric, n, universal, derivable in report.metrics.coverage
                if universal and not derivable and n < report.companies
            ]
            results.append(
                _check(
                    not unmet,
                    "universal metrics cover every company",
                    ", ".join(f"{m} {n}/{report.companies}" for m, n in unmet) or "all resolved",
                )
            )

    passed = sum(results)
    console.rule(
        f"[bold green]{passed}/{len(results)} passed[/bold green]"
        if all(results)
        else f"[bold red]{passed}/{len(results)} passed[/bold red]"
    )
    raise typer.Exit(0 if all(results) else 1)


# --------------------------------------------------------------------------
# M3 -- text
# --------------------------------------------------------------------------

OFFSET_SAMPLE = 200
MIN_RECALL_AT_50 = 0.85


@app.command()
def chunks(
    rebuild: Annotated[
        bool, typer.Option("--rebuild/--no-rebuild", help="Re-parse every filing.")
    ] = False,
) -> None:
    """Parse the corpus into sections and cut it into chunks.

    Cached against the source hash and the chunker version, so a second run
    with neither changed re-reads parquet and parses nothing.
    """
    from filing.stores.chunks import build_chunks

    cfg = settings()
    if not cfg.manifest_path.exists():
        console.print(f"[red]no manifest at {cfg.manifest_path} -- run `filing ingest`[/red]")
        raise typer.Exit(1)

    console.rule("[bold]filing chunks[/bold]")
    report = build_chunks(cfg, rebuild=rebuild)
    _print_chunk_report(report)


def _print_chunk_report(report) -> None:  # noqa: ANN001
    console.print(
        f"  {report.filings:,} filings  "
        f"[dim]{report.parsed:,} parsed, {report.reused:,} reused from cache[/dim]"
    )
    console.print(
        f"  {report.split:,} split into items  {report.degraded:,} whole-document  "
        f"{len(report.quarantined):,} quarantined"
    )
    for accn, reason in report.quarantined:
        console.print(f"        [yellow]quarantined[/yellow] {accn}: {reason}")
    console.print(
        f"  {report.chunks:,} chunks  {report.chars:,} characters  "
        f"{report.stub_sections:,} sections too short to chunk"
    )
    table = Table(box=None, pad_edge=False)
    table.add_column("item", style="cyan")
    table.add_column("chunks", justify="right")
    for item, n in sorted(report.by_item.items(), key=lambda kv: -kv[1])[:12]:
        table.add_row(item, f"{n:,}")
    console.print(table)
    console.print(f"  [dim]{report.seconds / 60:.1f} minutes[/dim]")


@app.command()
def index(
    rebuild: Annotated[
        bool, typer.Option("--rebuild/--no-rebuild", help="Drop the collection first.")
    ] = False,
    all_items: Annotated[
        bool, typer.Option("--all-items", help="Index the financial statements too.")
    ] = False,
    limit: Annotated[
        int, typer.Option("--limit", help="Stop after this many chunks (a dry run).")
    ] = 0,
) -> None:
    """Embed the narrative chunks into Qdrant and build the BM25 index."""
    from filing.stores.index import build_index

    cfg = settings()
    console.rule("[bold]filing index[/bold]")
    report = build_index(cfg, rebuild=rebuild, all_items=all_items, limit=limit or None)
    console.print(f"  collection [cyan]{report.collection}[/cyan]  {report.model} {report.dim}d")
    console.print(
        f"  {report.selected:,} of {report.candidates:,} chunks selected  "
        f"[dim]{'every item' if all_items else 'narrative items only'}[/dim]"
    )
    console.print(
        f"  {report.upserted:,} upserted  {report.already_indexed:,} already present  "
        f"{report.sparse_documents:,} in the BM25 index"
    )
    console.print(
        f"  {report.embed_http_calls:,} embedding HTTP calls  "
        f"{report.cache_hits:,} cache hits  [dim]{report.seconds / 60:.1f} minutes[/dim]"
    )


@app.command()
def graph(
    all_items: Annotated[
        bool, typer.Option("--all-items", help="Extract from the financial statements too.")
    ] = False,
) -> None:
    """Extract the entity graph from the same chunks the index covers."""
    from filing.stores.graph import build_graph

    console.rule("[bold]filing graph[/bold]")
    report = build_graph(settings(), all_items=all_items)
    console.print(
        f"  {report.sentences:,} sentences over {report.chunks:,} chunks  "
        f"[dim]{report.organizations:,} organisations discovered[/dim]"
    )
    console.print(f"  {report.edges:,} edges over {report.nodes:,} nodes")
    table = Table(box=None, pad_edge=False)
    table.add_column("relation", style="cyan")
    table.add_column("edges", justify="right")
    for kind, n in (report.by_kind or {}).items():
        table.add_row(kind, f"{n:,}")
    console.print(table)
    console.print("  most connected: " + ", ".join(f"{n} ({d})" for n, d in report.top_nodes[:8]))
    console.print(f"  [dim]{report.seconds / 60:.1f} minutes[/dim]")


@app.command()
def failures(
    kind: Annotated[
        str | None, typer.Option("--kind", help="Show example traces for one tag.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Examples to show.")] = 5,
) -> None:
    """The M6 failure gallery, counted by Phoenix rather than read by hand.

    Every number below is the length of a server-side filter's result, and the
    expression that produced it is printed beside it -- paste one into the
    Phoenix UI's filter box and you get the same spans. That is the whole
    claim: the gallery is a filter.

    The consistency line is not decoration. Phoenix answers a misspelt
    attribute path with an empty result rather than an error, so a broken
    query and a clean run look identical; cross-checking the per-tag filters
    against a tag-agnostic one is what tells them apart.
    """
    from filing.agent.verify import TAXONOMY
    from filing.gallery import QUERIES, Gallery, collect, examples, trace_url

    cfg = settings()
    g: Gallery = collect(cfg=cfg)
    console.rule(f"[bold]failure gallery[/bold] -- {g.project} @ {g.endpoint}")

    if not g.counts and g.errors:
        console.print("[red]Phoenix did not answer.[/red] Is the collector running?")
        console.print(f"  [dim]{next(iter(g.errors.values()))}[/dim]")
        raise typer.Exit(1)

    table = Table("filter", "spans", "expression", box=None, pad_edge=False)
    for label in ("verified", "flagged", "blocked", "tagged"):
        n = g.counts.get(label)
        table.add_row(label, "err" if n is None else f"{n:,}", f"[dim]{QUERIES[label]}[/dim]")
    table.add_row("", "", "")
    for tag in TAXONOMY:
        n = g.counts.get(tag)
        table.add_row(tag, "err" if n is None else f"{n:,}", f"[dim]{QUERIES[tag]}[/dim]")
    console.print(table)

    for tag, why in TAXONOMY.items():
        console.print(f"  [dim]{tag:<22} {why}[/dim]")

    console.print()
    if g.consistent:
        console.print("  [green]consistent[/green] -- per-tag filters cover every tagged span")
    else:
        console.print(
            "  [yellow]inconsistent[/yellow] -- a tag filter matched fewer spans than exist; "
            "check the attribute path before trusting a zero"
        )

    if kind:
        if kind not in TAXONOMY:
            console.print(f"[red]unknown tag[/red] {kind!r}; known: {', '.join(TAXONOMY)}")
            raise typer.Exit(1)
        console.print()
        console.rule(f"[bold]{kind}[/bold]")
        rows = examples(kind, limit=limit, cfg=cfg)
        if not rows:
            console.print("  [dim]no spans carry this tag[/dim]")
        for r in rows:
            # Rich reads square brackets as markup, so the tag list gets its
            # own delimiter rather than the obvious one silently vanishing.
            tags = " + ".join(r["kinds"]) or "none"
            console.print(f"  {r['start_time']}  {tags}  mode={r['mode']}")
            if r["reason"]:
                console.print(f"    [dim]{r['reason'][:160]}[/dim]")
            console.print(f"    [dim]{trace_url(r['trace_id'], cfg=cfg)}[/dim]")


@app.command()
def serve(
    config: Annotated[
        str, typer.Option("--config", help="Which eval config the server answers under.")
    ] = "",
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    warm: Annotated[
        bool, typer.Option("--warm/--lazy", help="Open the stores before accepting requests.")
    ] = True,
) -> None:
    """Serve /ask, the answer with its evidence attached.

    Warm by default: the first request would otherwise pay for opening Qdrant,
    the BM25 index and the cross-encoder, and a demo whose first question takes
    forty seconds is a demo nobody watches to the end. ``--lazy`` is there for
    the case where the point is to see the 503 a missing corpus produces.

    Bound to localhost, because the endpoint has no authentication and spends a
    model budget on every request.
    """
    from filing.api import DEFAULT_CONFIG
    from filing.api import serve as run_server

    name = config or DEFAULT_CONFIG
    console.print(f"[bold]filing[/bold] serving [cyan]{name}[/cyan] on http://{host}:{port}")
    console.print(f"  docs   http://{host}:{port}/docs")
    console.print("  ui     streamlit run src/filing/ui.py")
    run_server(config=name, host=host, port=port, warm=warm)


@app.command()
def text(
    sample: Annotated[
        int, typer.Option("--sample", help="Chunks to resolve back to source text.")
    ] = OFFSET_SAMPLE,
) -> None:
    """M3 gate: six checks over the chunks, the indexes and the graph.

    Nothing here is rebuilt first. Unlike the M2 gate, whose store takes a
    minute, the artefacts under test cost four hours of rate-limited embedding
    calls -- so this asserts against what is on disk, and check 3 is what makes
    that safe: it proves a rebuild would be free.
    """
    import random

    from filing.stores.chunks import CHUNKER, ChunkStore
    from filing.stores.evalset import SMOKE_QUESTIONS, evaluate
    from filing.stores.graph import GraphStore
    from filing.stores.index import build_index, select
    from filing.stores.parse import flatten
    from filing.stores.retrieve import Retriever

    cfg = settings()
    store = ChunkStore(cfg.chunks_dir)
    if not store.exists:
        console.print(f"[red]no chunks at {cfg.chunks_dir} -- run `filing chunks`[/red]")
        raise typer.Exit(1)

    console.rule("[bold]filing text[/bold]")
    results: list[bool] = []
    outcomes = store.outcomes()
    all_chunks = store.chunks()
    with Manifest(cfg.manifest_path, cfg.data_dir) as m:
        rows = m.con.execute("SELECT accn, path FROM filings").fetchall()
        # Resolved here, once: a manifest path is relative to the data
        # directory, and the manifest is the only thing that knows that.
        paths = {accn: m.resolve(rel) for accn, rel in rows}

    # 1 -- every filing in the manifest has a written outcome, and every
    # quarantine says why. A filing that is simply absent from the chunk store
    # is the failure this catches: it would show up nowhere else.
    missing = [a for a in paths if a not in outcomes]
    silent = [o.accn for o in outcomes.values() if o.outcome == "quarantined" and not o.reason]
    results.append(
        _check(
            not missing and not silent,
            "every filing parsed or quarantined with a reason",
            f"{len(outcomes):,}/{len(paths):,} accounted for, "
            f"{sum(1 for o in outcomes.values() if o.outcome == 'quarantined')} quarantined",
        )
    )
    for accn in missing[:5]:
        console.print(f"        [red]no outcome[/red] {accn}")

    # 2 -- the offsets are real. Re-flatten the filing and compare the slice
    # against the stored text, character for character. This is the check that
    # makes a citation checkable rather than decorative.
    rng = random.Random(20240301)
    picked = rng.sample(all_chunks, min(sample, len(all_chunks)))
    by_accn: dict[str, list] = {}
    for c in picked:
        by_accn.setdefault(c.accn, []).append(c)
    bad: list[str] = []
    for accn, group in by_accn.items():
        source, _blocks = flatten(paths[accn].read_text(encoding="utf-8"))
        for c in group:
            if source[c.char_start : c.char_end] != c.text:
                bad.append(f"{c.ticker} {accn} {c.char_start}:{c.char_end}")
    results.append(
        _check(
            not bad,
            "chunk offsets resolve to identical source text",
            f"{len(picked)} chunks across {len(by_accn)} filings, chunker {CHUNKER}",
        )
    )
    for b in bad[:5]:
        console.print(f"        [red]differs[/red] {b}")

    # 3 -- re-indexing an unchanged corpus is free. Two mechanisms have to
    # agree for this: Qdrant reports the ids it already holds, and the backend
    # caches every vector by text. Zero HTTP calls means neither was needed.
    again = build_index(cfg)
    results.append(
        _check(
            again.embed_http_calls == 0 and again.upserted == 0,
            "re-indexing an unchanged corpus costs zero embedding calls",
            f"{again.upserted:,} upserted, {again.already_indexed:,} already present, "
            f"{again.embed_http_calls:,} HTTP calls",
        )
    )

    # 4 and 5 -- the smoke set. One retrieval per question, scored twice: the
    # RRF order and the cross-encoder's reordering of the same fifty.
    retriever = Retriever(cfg)
    retriever.require()
    report = evaluate(retriever)
    results.append(
        _check(
            report.recall_at_50 >= MIN_RECALL_AT_50 and not report.missing_gold,
            "recall@50 on the smoke set",
            f"{report.recall_at_50:.0%} over {report.n} questions (floor {MIN_RECALL_AT_50:.0%})",
        )
    )
    for miss in report.misses:
        console.print(f"        [yellow]no gold in top 50[/yellow] {miss.question.id}")
    for gone in report.missing_gold:
        console.print(f"        [red]no gold anywhere in the index[/red] {gone}")

    results.append(
        _check(
            report.rerank_delta > 0,
            "reranking improves precision@5",
            f"{report.precision_at_5_fused:.3f} fused -> "
            f"{report.precision_at_5_reranked:.3f} reranked "
            f"({report.rerank_delta:+.3f})",
        )
    )

    # 6 -- every graph edge quotes text that is really there. The graph's whole
    # claim is auditability, and an edge whose sentence does not appear at its
    # own offsets is an assertion with a fabricated citation attached.
    edges = GraphStore(cfg.graph_dir).edges()
    if edges:
        picked_edges = random.Random(20240302).sample(edges, min(sample, len(edges)))
        by_accn.clear()
        for e in picked_edges:
            by_accn.setdefault(e.accn, []).append(e)
        wrong = []
        for accn, group in by_accn.items():
            source, _blocks = flatten(paths[accn].read_text(encoding="utf-8"))
            for e in group:
                if " ".join(source[e.char_start : e.char_end].split()) != e.sentence:
                    wrong.append(f"{e.source} -> {e.target} {accn} {e.char_start}")
        results.append(
            _check(
                not wrong,
                "graph edges quote their source sentence",
                f"{len(picked_edges)} of {len(edges):,} edges checked",
            )
        )
        for w in wrong[:5]:
            console.print(f"        [red]not at those offsets[/red] {w}")
    else:
        results.append(_check(False, "graph edges quote their source sentence", "no graph built"))

    console.print()
    console.print(
        f"  [dim]{len(all_chunks):,} chunks, {len(select(all_chunks)):,} indexed, "
        f"{len(edges):,} edges, {len(SMOKE_QUESTIONS)} questions[/dim]"
    )
    passed = sum(results)
    console.rule(
        f"[bold green]{passed}/{len(results)} passed[/bold green]"
        if all(results)
        else f"[bold red]{passed}/{len(results)} passed[/bold red]"
    )
    raise typer.Exit(0 if all(results) else 1)


def _forms(by_form: dict[str, int]) -> str:
    return ", ".join(f"{n} {form}" for form, n in sorted(by_form.items())) or "none"


def main() -> None:  # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    app()
