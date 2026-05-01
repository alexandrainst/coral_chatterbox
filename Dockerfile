# Base image carries the same torch + CUDA versions the project pins
# (torch==2.7.1, CUDA 12.8). Using the official PyTorch runtime image avoids
# fighting wheel compatibility for the rest of the stack.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/conda

# System deps for librosa / soundfile and git-based installs (resemble-perth).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv: fast resolver/installer. Pinned by digest-equivalent tag.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install into the base image's existing conda env (which already has the
# pinned torch build) so we don't rebuild a venv from scratch.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN uv pip install --python /opt/conda/bin/python -e ".[serve]"

# Pre-fetch NLTK punkt_tab so the first request doesn't try to download it.
# (utils/splitter.py also calls nltk.download at import time, but baking the
# data in lets the container run with no outbound network.)
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

EXPOSE 8000

# voices/ is the conventional mount point. Override with -e VOICES_DIR=...
VOLUME ["/voices"]
ENV VOICES_DIR=/voices

WORKDIR /app/src
CMD ["python", "-m", "server"]
