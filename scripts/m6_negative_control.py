"""Does the verifier catch anything? Four mutations of every real answer.

    ./.conda/python.exe scripts/m6_negative_control.py

The M6 run flags nothing, which is the good outcome and the unconvincing one:
a check that never fires is indistinguishable from a check that cannot. So
every answer the agent actually produced is perturbed in the four ways the
verifier claims to detect, and the catch rate is measured.

This is not a gate and is not in the test suite -- it replays all 150 questions
through the real graph, which is free only because the chat cache is warm. It
is the evidence behind the M6 write-up's catch-rate line, and it earns its
place in the repo by having found things the unit tests did not:

  * four false positives -- a multi-marker citation read as three figures, the
    date a period ends read as a pair of figures, a minus-signed loss compared
    against a positive, and a refusal checked against fact evidence it never
    claimed. Between them they produced every flag in the first M6 run.
  * one false negative -- a purely qualitative answer with every citation
    stripped passed, because the uncited-claim rule was figure-scoped. That is
    exactly the "claim without a resolvable locator" the gate forbids, and it
    only showed up because something deliberately broke 116 real answers.

Results are written to results/m6-negative-control.json so the numbers in the
README have a file behind them.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

from filing.agent.graph import build_graph, run_question
from filing.agent.verify import _CITATION, _read, strip_citations, verify_answer
from filing.config import PROJECT_ROOT, settings
from filing.eval import dataset
from filing.eval.runner import build_agent_tools, get_config
from filing.llm.factory import build_backend

OUT = PROJECT_ROOT / "results" / "m6-negative-control.json"

#: The verifier reads figures with its own parser; this one only has to find
#: the digits of a figure the verifier already located, so it can be crude.
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def bend(answer: str) -> str | None:
    """Change the first stated figure to one no store carries."""
    stated = _read(strip_citations(answer))
    if not stated:
        return None
    m = NUM.search(stated[0].text.strip("$() "))
    if not m:
        return None
    raw = m.group(0)
    try:
        val = float(raw.replace(",", ""))
    except ValueError:
        return None
    if val == 0:
        return None
    bent = f"{val * 1.37:,.0f}" if "." not in raw else f"{val * 1.37:,.2f}"
    return answer.replace(raw, bent, 1)


def dangle(answer: str) -> str | None:
    """Point a citation past the end of the evidence."""
    return answer + " Also relevant [99]." if _CITATION.search(answer) else None


def uncite(answer: str) -> str | None:
    """Remove the markers, leaving the claims asserted with nothing to check."""
    out = _CITATION.sub("", answer)
    return out if out != answer else None


def swap(answer: str) -> str | None:
    """Replace the first figure with a plausible one from nowhere."""
    stated = _read(strip_citations(answer))
    if not stated:
        return None
    m = NUM.search(stated[0].text.strip("$() "))
    return answer.replace(m.group(0), "48,317,905", 1) if m else None


MUTATIONS = {
    "bent figure": bend,
    "dangling citation": dangle,
    "stripped citations": uncite,
    "swapped figure": swap,
}


def main() -> int:
    cfg = settings()
    ec = get_config("agent-flagged").resolved(cfg)
    tools = build_agent_tools(cfg, backend=build_backend(cfg, ec.chat_backend or None), config=ec)
    app = build_graph(tools)
    qs = dataset.read(dataset.dataset_path(pathlib.Path(cfg.data_dir), ec.dataset_version))

    caught = dict.fromkeys(MUTATIONS, 0)
    applied = dict.fromkeys(MUTATIONS, 0)
    clean_pass = clean_total = 0
    missed: list[str] = []

    for i, q in enumerate(qs, 1):
        state = run_question(q.question, tools=tools, qid=q.id, graph=app)
        answer = str(state.get("answer") or "")
        evidence = list(state.get("evidence") or [])
        # A refusal has no claim to bend, and is measured by the abstention
        # gate rather than this one.
        if not answer or state.get("refused"):
            continue
        kw = dict(
            question=q.question,
            route=str(state.get("route") or ""),
            grade=state.get("grade"),
            repairs=int(state.get("repairs") or 0),
            repair_log=list(state.get("repair_log") or []),
            sql=tools.sql,
            refused=bool(state.get("refused")),
        )
        clean_total += 1
        clean_pass += int(verify_answer(answer, evidence, **kw).ok)
        for name, fn in MUTATIONS.items():
            bad = fn(answer)
            if bad is None or bad == answer:
                continue
            applied[name] += 1
            if not verify_answer(bad, evidence, **kw).ok:
                caught[name] += 1
            elif len(missed) < 8:
                missed.append(f"{q.id} [{name}] {bad[:110]}")
        print(f"  {i}/{len(qs)} {q.id}", end="\r", file=sys.stderr)

    tot_a, tot_c = sum(applied.values()), sum(caught.values())
    print(f"\n\nclean answers passing: {clean_pass}/{clean_total}")
    print(f"{'mutation':22} {'applied':>8} {'caught':>8}  rate")
    for name in MUTATIONS:
        a, c = applied[name], caught[name]
        print(f"{name:22} {a:>8} {c:>8}  {c / a:.1%}" if a else f"{name:22} {a:>8} {c:>8}     --")
    print(f"{'TOTAL':22} {tot_a:>8} {tot_c:>8}  {tot_c / tot_a:.1%}")
    if missed:
        print("\nnot caught (first few):")
        for line in missed:
            print("  " + line)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "config": "agent-flagged",
                "dataset_version": ec.dataset_version,
                "clean_answers": clean_total,
                "clean_passing": clean_pass,
                "mutations": {
                    name: {"applied": applied[name], "caught": caught[name]} for name in MUTATIONS
                },
                "total_applied": tot_a,
                "total_caught": tot_c,
                "not_caught": missed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT.relative_to(PROJECT_ROOT)}")
    # A false positive on a real answer is a worse failure than a missed
    # mutation, so it, and only it, sets the exit code.
    return 0 if clean_pass == clean_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
