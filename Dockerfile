# syntax=docker/dockerfile:1
# One image, two entrypoints: the API and the page that reads it.
#
# Three decisions worth the comment.
#
# 1. CPU torch, installed from PyTorch's own index *before* anything else. The
#    default sentence-transformers dependency resolves to a CUDA build and
#    drags ~2.5 GB of kernels onto a machine that will never have a GPU.
#
#    The measured result, once this file had actually been built:
#
#        1.32 GB  pip install -e ".[ui]"
#        1.03 GB  torch + torchvision (cpu)
#         227 MB  the two models, baked in -- see (2)
#         107 MB  apt: git, curl
#        ------
#        3.76 GB  filing-api:latest
#
#    An earlier version of this comment claimed the CPU index was "the
#    difference between a 1.4 GB image and a 4 GB one". The 4 GB was about
#    right; the 1.4 GB was invented before anything was built, and is off by
#    2.4 GB. The saving is real -- a CUDA torch layer alone runs past 2.5 GB --
#    but it is a saving off the top, not a small image.
#
#    The remaining fat is in the dependency layer: docling pulls rapidocr,
#    which pulls opencv-python at 74 MB, for a code path the *served* container
#    never runs. Parsing happens once, at `filing ingest` time, on a laptop.
#    Moving docling to an extra alongside [ui] is the largest single win left
#    and is deliberately not bundled into the torchvision fix below.
#
# 2. The two local models are baked in at build time. A container whose first
#    request goes to huggingface.co is a container that fails on a plane, in a
#    locked-down network, or on the third attempt when the rate limiter notices
#    you. `filing serve` is warm by default for the same reason -- a demo whose
#    first question takes forty seconds is a demo nobody watches to the end.
#
# 3. The corpus is a mount, not a layer. data/ is 1.2 GB of SEC filings and
#    their vectors; baking it in would make every code change a 1.2 GB rebuild,
#    and it is reproducible from `filing ingest` besides.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# git: docling asks for it at import time on some paths. curl: the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# CPU torch first -- see (1). Pinned to the index, not to a version, because
# the CPU index only ever serves CPU wheels.
#
# torchvision has to come from that index too, and the first version of this
# line left it out. transformers imports `torchvision.io` eagerly, so pip
# resolved it from PyPI on the next instruction -- at the *correct* paired
# version, 0.29.0 against torch 2.14.0, which is what made the mistake hard to
# see. Version pairing is not the thing that matters. The PyPI wheel is linked
# against the CUDA build's ABI, so its compiled `torchvision::nms` fails to
# register against a CPU torch and `import sentence_transformers` dies with
# "operator torchvision::nms does not exist" -- twenty minutes into the build,
# on the line that downloads the models.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Dependencies before source, so editing a module does not reinstall docling.
# The pip cache is a BuildKit cache mount rather than a layer: this step
# pulls ~250 MB from PyPI, and the container gets about a fifth of the
# host's throughput to it -- 70 kB/s against 380. A rebuild that has to
# redo this step therefore costs twenty minutes of re-downloading wheels
# that have not changed. A cache mount is not committed to the image, so
# it buys that back without the size PIP_NO_CACHE_DIR was set to avoid,
# which is why that variable is gone: it would have won over the mount.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/filing && touch src/filing/__init__.py \
 && pip install -e ".[ui]"

# See (2). Downloaded under HF_HOME, which the runtime reads from the same env.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder;\
SentenceTransformer('BAAI/bge-small-en-v1.5');\
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY src ./src
COPY universe.yaml ./
COPY data/eval ./data/eval

# Inside the network the API is reachable by service name; the port is only
# published to the host's loopback by compose. `serve` binds 127.0.0.1 by
# default and that is right for a laptop -- here it has to answer the ui
# container, which is a different address.
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "filing.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
