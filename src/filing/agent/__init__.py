"""M5: the agent, and the parts of it that are not a language model.

The graph is plan -> route -> retrieve -> rerank -> grade -> repair ->
synthesise, and the design rule running through it is that **a hosted model is
used where judgement is genuinely needed and nowhere else**. Planning and
routing are one call, not two. Retrieval and reranking are local. Grading is
deterministic wherever the evidence is checkable -- a SQL row either came back
or it did not -- because M4 kept language models out of the metric path and an
agent that grades itself with a model puts them straight back in.

Two consequences fall out of that. The budget is roughly two hosted calls per
question rather than five, which is the difference between a 150-question run
costing 350 calls and costing 750 on a free tier that meters calls. And the
numbers the agent reports about its own evidence are reproducible without a
network.
"""
