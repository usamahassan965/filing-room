# One image, two entrypoints: the API and the page that reads it.
#
# Three decisions worth the comment.
#
# 1. CPU torch, installed from PyTorch's own index *before* anything else. The
#    default sentence-transformers dependency resolves to a CUDA build and
#    drags ~2.5 GB of kernels onto a machine that will never have a GPU. This
#    line is the difference between a 1.4 GB image and a 4 GB one.
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
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# git: docling asks for it at import time on some paths. curl: the healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl \
 && rm -rf /var/lib/apt/lists/*

# CPU torch first -- see (1). Pinned to the index, not to a version, because
# the CPU index only ever serves CPU wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Dependencies before source, so editing a module does not reinstall docling.
COPY pyproject.toml README.md ./
RUN mkdir -p src/filing && touch src/filing/__init__.py \
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
