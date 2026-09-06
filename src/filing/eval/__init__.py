"""Measurement: the naive system, the frozen question set, and the metrics.

M4 exists so that every later claim about this project is a number with a
comparison attached. Three things live here and nothing else does.

* ``naive`` -- a deliberately unsophisticated RAG system: the whole filing cut
  every 2,048 characters, embedded with no context, retrieved dense-only at
  top-5, answered in one call. It is the thing the real system has to beat, and
  it is built honestly rather than crippled, because a straw baseline makes
  every later improvement meaningless.
* ``dataset`` -- 150 questions, frozen and versioned, each carrying gold the
  corpus can verify rather than gold a model asserted.
* ``metrics`` -- deterministic scoring only. Nothing in this package calls a
  language model to decide whether an answer was right.
"""
