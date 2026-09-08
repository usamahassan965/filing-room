"""The deploy path: an embedded store, a second surface, and one command.

Three things have to hold for the free deployment to be a deployment rather
than a demo of one:

1. ``qdrant_path`` really removes the server. Not "falls back to" -- a
   ``VectorIndex`` built with it must never open a socket, because the host it
   is built for has nothing to open one to.
2. ``pack_embedded`` moves the points rather than the files. The formats
   differ, and a copy that appeared to work would fail at query time.
3. The Gradio surface renders the same four outcomes the Streamlit one does,
   from the same payload, without holding an opinion of its own.

Nothing here needs a corpus, a model or a network. The Qdrant used below is a
real embedded one over a tmp_path, which is the whole point: it is the same
client the Space runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import qdrant_client

from filing.config import Settings
from filing.stores import index as index_mod


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    return Settings(qdrant_path=tmp_path / "store", embed_backend="local")


# --------------------------------------------------------------------------
# the server is gone, not merely unused
# --------------------------------------------------------------------------


def test_qdrant_path_opens_a_directory_and_never_a_url(cfg: Settings, monkeypatch) -> None:  # noqa: ANN001
    """The property the Space depends on, asserted rather than assumed.

    A ``VectorIndex`` that quietly fell back to ``localhost:6333`` would pass
    every test on a laptop with Docker running and fail on the only machine
    that matters.
    """
    seen: dict[str, object] = {}

    class Spy:
        def __init__(self, *args, **kw) -> None:  # noqa: ANN002, ANN003
            seen.update(kw)
            seen["args"] = args

    monkeypatch.setattr(qdrant_client, "QdrantClient", Spy)
    index_mod.VectorIndex(cfg)
    assert "path" in seen
    assert "url" not in seen and "api_key" not in seen


def test_without_qdrant_path_the_url_is_used(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    seen: dict[str, object] = {}

    class Spy:
        def __init__(self, *args, **kw) -> None:  # noqa: ANN002, ANN003
            seen.update(kw)

    monkeypatch.setattr(qdrant_client, "QdrantClient", Spy)
    index_mod.VectorIndex(Settings(qdrant_url="http://example:6333", embed_backend="local"))
    assert seen["url"] == "http://example:6333"
    assert "path" not in seen


def test_an_empty_api_key_is_sent_as_none(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """``""`` and ``None`` are not the same to qdrant-client.

    It sends the header whenever the value is not None, and a local Qdrant
    rejects an empty bearer token rather than ignoring it -- so the default
    would break the default deployment.
    """
    seen: dict[str, object] = {}

    class Spy:
        def __init__(self, *args, **kw) -> None:  # noqa: ANN002, ANN003
            seen.update(kw)

    monkeypatch.setattr(qdrant_client, "QdrantClient", Spy)
    index_mod.VectorIndex(Settings(embed_backend="local"))
    assert seen["api_key"] is None


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def _shape(cfg: Settings) -> tuple[str, int]:
    """The collection name and dimension the config implies, without holding it.

    An embedded store takes an exclusive lock on its directory, so a
    ``VectorIndex`` left open here would make the seeding below fail with a
    message about concurrent access.
    """
    idx = index_mod.VectorIndex(cfg)
    try:
        return idx.name, idx.dim
    finally:
        idx.client.close()


def _seed(path: Path, name: str, dim: int, n: int) -> None:
    """A real embedded collection, standing in for the served one."""
    from qdrant_client import QdrantClient, models

    c = QdrantClient(path=str(path))
    try:
        c.create_collection(
            name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )
        c.upsert(
            name,
            points=[
                models.PointStruct(
                    id=i,
                    vector=[float((i + j) % 7) / 7 for j in range(dim)],
                    payload={"ticker": "NVDA" if i % 2 else "AMD", "chunk_id": f"c{i}"},
                )
                for i in range(n)
            ],
            wait=True,
        )
    finally:
        c.close()


def test_pack_moves_every_point_with_its_payload(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """Read out and written back, not copied.

    The source here is itself an embedded store rather than a server, which
    changes nothing about what is being tested: ``pack_embedded`` drives the
    source through ``scroll`` and the destination through ``upsert``, and
    neither call knows which kind it is talking to.
    """
    from qdrant_client import QdrantClient

    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    cfg = Settings(qdrant_path=src_dir, embed_backend="local")
    name, dim = _shape(cfg)
    _seed(src_dir, name, dim, 40)

    # `require` asserts the collection is present and populated; the seeded one
    # is, and nothing else in this test needs the real corpus.
    report = index_mod.pack_embedded(cfg, dst_dir, batch=7)
    assert report.points == 40
    assert report.dim == dim
    assert report.collection == name
    assert report.seconds >= 0

    out = QdrantClient(path=str(dst_dir))
    try:
        assert out.count(name).count == 40
        got = out.retrieve(name, ids=[3], with_payload=True, with_vectors=True)
        assert got[0].payload == {"ticker": "NVDA", "chunk_id": "c3"}
        assert len(got[0].vector) == dim  # type: ignore[arg-type]
    finally:
        out.close()


def test_pack_replaces_a_destination_rather_than_doubling_it(tmp_path: Path) -> None:
    """Twice is the same as once. A pack that appended would silently duplicate
    every point on the second run, and the count is the only thing that would
    say so."""
    from qdrant_client import QdrantClient

    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    cfg = Settings(qdrant_path=src_dir, embed_backend="local")
    name, dim = _shape(cfg)
    _seed(src_dir, name, dim, 12)

    index_mod.pack_embedded(cfg, dst_dir)
    second = index_mod.pack_embedded(cfg, dst_dir)
    assert second.points == 12

    out = QdrantClient(path=str(dst_dir))
    try:
        assert out.count(name).count == 12
    finally:
        out.close()


def test_a_packed_store_still_filters(tmp_path: Path) -> None:
    """The embedded client has no payload indexes and warns about it.

    It evaluates the filter over the points instead, which is the only property
    ``VectorIndex.search`` depends on -- and the routes that scope a search to
    one ticker would return the whole corpus if it did not hold.
    """
    src_dir, dst_dir = tmp_path / "src", tmp_path / "dst"
    cfg = Settings(qdrant_path=src_dir, embed_backend="local")
    name, dim = _shape(cfg)
    _seed(src_dir, name, dim, 20)
    index_mod.pack_embedded(cfg, dst_dir)

    packed = index_mod.VectorIndex(Settings(qdrant_path=dst_dir, embed_backend="local"))
    hits = packed.search([0.5] * dim, limit=20, where={"ticker": "NVDA"})
    assert hits
    assert len(hits) == 10  # half the points, and not the other half


# --------------------------------------------------------------------------
# the Gradio surface
# --------------------------------------------------------------------------

gr = pytest.importorskip("gradio", reason="the `space` extra is not installed")


class FakeEngine:
    """Yields the events the real engine yields, and nothing else."""

    def __init__(self, payload: dict, *, boom: bool = False) -> None:
        self.payload, self.boom = payload, boom

    def stream(self, question: str, *, qid: str = ""):  # noqa: ANN201, ARG002
        yield {"event": "stage", "data": {"node": "plan", "label": "planning the route"}}
        yield {"event": "stage", "data": {"node": "retrieve", "label": "retrieving"}}
        if self.boom:
            raise RuntimeError("the graph fell over")
        yield {"event": "answer", "data": self.payload}


ANSWERED = {
    "outcome": "answered",
    "answer": "NVDA reported Revenues of 26,974,000,000 USD [1].",
    "seconds": 2.4,
    "llm_calls": 2,
    "config": "agent-guarded",
    "verification": {"enabled": True, "figures_checked": 1, "unsupported": 0, "markers": [1]},
    "evidence": [],
    "trace": {},
}


def test_every_yield_is_one_value_per_output() -> None:
    """Gradio raises at runtime, not at build time, on a width mismatch --
    which means a wrong tuple ships and fails on the first question."""
    from filing import gradio_app

    yields = list(gradio_app.run(FakeEngine(ANSWERED), "what were revenues?"))
    widths = {len(y) for y in yields}
    assert len(widths) == 1
    page = gradio_app.build(FakeEngine(ANSWERED))
    assert isinstance(page, gr.Blocks)


def test_the_stages_accumulate_and_the_answer_lands_last() -> None:
    from filing import gradio_app

    yields = list(gradio_app.run(FakeEngine(ANSWERED), "what were revenues?"))
    assert "planning the route" in yields[-2][0]
    assert "retrieving" in yields[-2][0]
    final = yields[-1]
    assert "26,974,000,000" in final[2]
    assert final[-1] == ANSWERED  # the raw payload, unedited


def test_an_empty_question_asks_rather_than_running() -> None:
    from filing import gradio_app

    engine = FakeEngine(ANSWERED)
    yields = list(gradio_app.run(engine, "   "))
    assert len(yields) == 1
    assert "Ask something" in yields[0][2]


def test_a_graph_that_falls_over_becomes_a_rendered_error() -> None:
    """Not a traceback into the browser and not a spinner that never resolves.

    By the time a run fails the page has already drawn a plan and a route, and
    leaving that on screen under a spinner is the state M7 says must not happen.
    """
    from filing import gradio_app

    final = list(gradio_app.run(FakeEngine(ANSWERED, boom=True), "q"))[-1]
    assert "Error" in final[1]
    assert "the graph fell over" in final[2]


def test_a_stream_that_ends_without_a_payload_is_an_error_too() -> None:
    from filing import gradio_app

    class Silent:
        def stream(self, question: str, *, qid: str = ""):  # noqa: ANN201, ARG002
            yield {"event": "stage", "data": {"node": "plan", "label": "planning"}}

    final = list(gradio_app.run(Silent(), "q"))[-1]
    assert "Error" in final[1]
    assert "without producing a payload" in final[2]


# --------------------------------------------------------------------------
# the push is not rejected on file size
# --------------------------------------------------------------------------


def test_every_large_file_in_the_payload_has_an_lfs_rule() -> None:
    """The LFS patterns are checked against the payload, not against memory.

    A Hugging Face repository refuses a plain-git file over 10 MB, and the
    rejection names the size limit rather than LFS -- so the fix looks like
    "carry less" when it is actually "write one line". ``.gitattributes`` is
    written by ``filing space`` from a fixed list of extensions, which means the
    list is only correct until the pipeline writes a format nobody added to it.
    It already was not: ``data/chunks/chunks.parquet`` is 44 MB and ``*.parquet``
    was missing, so the very first push would have been rejected on a file the
    deploy cannot run without.

    So the assertion walks what would actually be uploaded. It skips where the
    corpus is absent, which is CI -- and that is the honest arrangement rather
    than a gap, because the failure it guards is a push, and a push happens on a
    machine that has the corpus.
    """
    import fnmatch

    from filing.cli import SPACE_DATA, SPACE_GITATTRIBUTES

    root = Path(__file__).resolve().parents[1]
    present = [root / rel for rel in SPACE_DATA if (root / rel).exists()]
    if not present:
        pytest.skip("no built corpus here; this guards the machine that pushes")

    patterns = [
        line.split()[0]
        for line in SPACE_GITATTRIBUTES.splitlines()
        if line and not line.startswith("#")
    ]
    limit = 10 * 1024 * 1024

    unmatched = []
    for path in present:
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for f in files:
            if f.stat().st_size <= limit:
                continue
            if not any(fnmatch.fnmatch(f.name, pat) for pat in patterns):
                unmatched.append(f"{f.relative_to(root)} ({f.stat().st_size // 1024**2} MB)")

    assert not unmatched, "over 10 MB and not tracked by LFS: " + ", ".join(unmatched)
