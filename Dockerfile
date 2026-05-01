# Base image carries the same torch + CUDA versions the project pins
# (torch==2.7.1, CUDA 12.8). Using the official PyTorch runtime image avoids
# fighting wheel compatibility for the rest of the stack.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for librosa / soundfile and git-based installs (resemble-perth).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the project + serve extra. Copy pyproject first to leverage layer
# caching when only sources change.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[serve]"

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
