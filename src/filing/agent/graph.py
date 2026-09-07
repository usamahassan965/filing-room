"""The graph, assembled. Eleven nodes, two conditional edges, one bounded loop.

Written with LangGraph for one reason that survives contact with an interviewer:
the topology is a value. ``build_graph().get_graph().draw_mermaid()`` prints the
diagram, so "plan -> route -> retrieve -> rerank -> grade -> repair ->
synthesise -> verify" is something a reader can check rather than something they
have to reconstruct from a call stack. The repair budget is the clearest case: it is not
a ``while`` loop somebody has to be trusted to have bounded correctly, it is a
conditional edge that has no branch to take once the counter reaches two.

The shape::

              plan ──> route ──┬──> retrieve_sql ───┐
                               ├──> retrieve_text ──┤
                               ├──> retrieve_graph ─┼──> rerank ──> grade
                               └──> refuse ─────────┘                 │
                                                                      │
                       repair <────── (not ok, budget left) ──────────┤
                         │                                            │
                         └──> route (again)          synthesise <─────┘
                                                          │
                                                       verify
                                                          │
                                                        END

Repair re-enters at ``route`` rather than at a retrieval node, because a repair
is allowed to change the route -- the SQL branch giving up and handing the
question to the text retriever is the single most useful thing the loop does,
and it is a routing decision.
"""

from __future__ import annotations

import time
from typing import Any

from filing.agent.nodes import Nodes, Tools, route_branch, should_repair
from filing.agent.state import REPAIR_BUDGET, AgentState

__all__ = ["REPAIR_BUDGET", "build_graph", "run_question", "mermaid"]


def build_graph(tools: Tools) -> Any:
    """Compile the agent. One node per named step, so one span per named step."""
    from langgraph.graph import END, StateGraph

    nodes = Nodes(tools)
    g: Any = StateGraph(AgentState)
    g.add_node("plan", nodes.plan)
    g.add_node("route", nodes.route)
    g.add_node("retrieve_sql", nodes.retrieve_sql)
    g.add_node("retrieve_text", nodes.retrieve_text)
    g.add_node("retrieve_graph", nodes.retrieve_graph)
    g.add_node("refuse", nodes.refuse)
    g.add_node("rerank", nodes.rerank)
    g.add_node("grade", nodes.grade)
    g.add_node("repair", nodes.repair)
    g.add_node("synthesise", nodes.synthesise)
    g.add_node("verify", nodes.verify)

    g.set_entry_point("plan")
    g.add_edge("plan", "route")
    g.add_conditional_edges(
        "route",
        route_branch,
        {
            "retrieve_sql": "retrieve_sql",
            "retrieve_text": "retrieve_text",
            "retrieve_graph": "retrieve_graph",
            "refuse": "refuse",
        },
    )
    for node in ("retrieve_sql", "retrieve_text", "retrieve_graph", "refuse"):
        g.add_edge(node, "rerank")
    g.add_edge("rerank", "grade")
    # The budget lives here and nowhere else. `should_repair` returns
    # "synthesise" once the counter reaches REPAIR_BUDGET, so a third repair is
    # not a thing the graph can express.
    g.add_conditional_edges(
        "grade", should_repair, {"repair": "repair", "synthesise": "synthesise"}
    )
    g.add_edge("repair", "route")
    # Verification is a node rather than a wrapper around the runner, so it
    # gets a span like every other step and shows up in the exported trace. An
    # unconditional edge: the check runs on every answer, including refusals,
    # and the `verify` flag decides whether it does anything rather than
    # whether the graph has the shape it is documented to have.
    g.add_edge("synthesise", "verify")
    g.add_edge("verify", END)
    return g.compile()


def mermaid(tools: Tools | None = None) -> str:
    """The diagram, from the compiled graph rather than from a docstring.

    A hand-drawn architecture diagram is a claim about the code; this one is the
    code. It goes in the README for M5 for that reason.
    """
    return build_graph(tools or Tools()).get_graph().draw_mermaid()


def run_question(
    question: str,
    *,
    tools: Tools,
    qid: str = "",
    graph: Any = None,
    recursion_limit: int = 40,
) -> AgentState:
    """One question through the graph, with the wall clock and errors captured.

    A single bad question must not end a 150-question run -- that is M4's rule
    for the baseline and it applies here with more force, because an agent has
    four more ways to fail. The exception is recorded in the state and the run
    continues; :func:`filing.eval.metrics.score` counts an errored outcome as
    wrong, which is what it is.
    """
    started = time.monotonic()
    app = graph if graph is not None else build_graph(tools)
    state: AgentState = {
        "qid": qid,
        "question": question,
        "repairs": 0,
        "llm_calls": 0,
        "repair_log": [],
        "evidence": [],
        "candidates": [],
    }
    try:
        # One span around the whole question, so the node spans have a parent to
        # hang from. Without it every node is a root and the "trace" is ten
        # unrelated spans that happen to share a process -- which is how the
        # exported trace ends up a list where a reader expects a tree.
        with tools.tracer.start_as_current_span("agent.question") as span:
            span.set_attribute("qid", qid)
            span.set_attribute("question", question[:200])
            final: AgentState = app.invoke(state, {"recursion_limit": recursion_limit})
            span.set_attribute("route", str(final.get("route", "")))
            span.set_attribute("repairs", int(final.get("repairs", 0)))
            span.set_attribute("llm_calls", int(final.get("llm_calls", 0)))
            span.set_attribute("refused", bool(final.get("refused", False)))
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        final = dict(state)  # type: ignore[assignment]
        final["error"] = f"{type(exc).__name__}: {exc}"
        final["answer"] = ""
    final["seconds"] = time.monotonic() - started  # type: ignore[typeddict-unknown-key]
    return final
