# START Hack Munich 2026 - Feyer: Faster Physics.
# The upstream challenge requires Python 3.12 + JAX with Linux CPU-affinity pinning (--cpus 1), which native Windows
# doesn't support. This image gives a matching Linux environment through Docker Desktop's WSL2 backend, so we don't
# have to hand-install a WSL distro just for this one repo. Mirrors the style of the repo's own docker/Dockerfile
# (matching host UID/GID, pip cache mount, non-root user) but skips everything that setup doesn't need (TeX, profiles).
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=0
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
ARG USERNAME=hack
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME} \
    && mkdir -p /work && chown ${UID}:${GID} /work
WORKDIR /work
# Install deps from the pinned upstream requirements.txt (copied in at build time) before the full repo is bind-mounted,
# so editing source files does not invalidate this layer.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt
# differometor itself is installed editable at container start (entrypoint below), once the full repo is bind-mounted
# over /work - the package source isn't in the build context yet at this layer.
USER ${USERNAME}
CMD ["sleep", "infinity"]
