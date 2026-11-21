#!/usr/bin/env bash
# container.bash - the ONE entrypoint for this challenge's dev container (Linux, WSL2, macOS with Docker Desktop).
# Differometor needs Python 3.12 + JAX with Linux CPU-affinity pinning (benchmark.py --cpus 1), which native Windows
# does not support - this container gives everyone the same Linux environment regardless of host OS.
#
#   ./container.bash                 build (if needed) + open a shell in the container
#   ./container.bash -b | --build    force rebuild the image (after editing requirements.txt or the Dockerfile)
#   ./container.bash exec CMD...     run one command in the container and exit, e.g.:
#                                       ./container.bash exec python start_here.py
#                                       ./container.bash exec python benchmark.py --cpus 1 --output benchmark-baseline.json
#   ./container.bash -s | --stop     stop the container (keeps the image/volumes)
#   ./container.bash -d | --delete   remove the container
#
# macOS (incl. Apple Silicon): works as is through Docker Desktop. If the image fails to build on an M-series Mac
# because a pinned wheel has no linux/arm64 build, add `platform: linux/amd64` under the `hack` service in
# docker/docker-compose.yaml to build/run under Rosetta emulation (slower, but correct).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
COMPOSE=(docker compose -f docker/docker-compose.yaml)
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

Mode=shell
case "${1:-}" in
  -b|--build) "${COMPOSE[@]}" build; shift; Mode=shell ;;
  -s|--stop) "${COMPOSE[@]}" stop; exit 0 ;;
  -d|--delete) "${COMPOSE[@]}" down; exit 0 ;;
  exec) shift; Mode=exec ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
esac

# Build once if the image doesn't exist yet
if ! docker image inspect feyer-hackathon:dev >/dev/null 2>&1; then
  echo "-> building feyer-hackathon:dev (first build is large, pulls jax/jaxlib etc.)"
  "${COMPOSE[@]}" build
fi

if [ "$Mode" = exec ]; then
  "${COMPOSE[@]}" run --rm hack "$@"
else
  "${COMPOSE[@]}" run --rm hack bash
fi
