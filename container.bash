#!/usr/bin/env bash
# container.bash - the ONE entrypoint for this challenge's dev container (Linux, WSL2, macOS with Docker Desktop).
# Differometor needs Python 3.12 + JAX with Linux CPU-affinity pinning (benchmark.py --cpus 1), which native Windows
# does not support - this container gives everyone the same Linux environment regardless of host OS.
#
#   ./container.bash                 env up + VS Code attached to the container (needs the 'code' CLI)
#   ./container.bash -u | --up       env up, no shell, no VS Code
#   ./container.bash -t | --terminal shell only (start the container first if needed)
#   ./container.bash exec CMD...     run one command in the running container and exit, e.g.:
#                                       ./container.bash exec python start_here.py
#                                       ./container.bash exec python benchmark.py --cpus 1 --output benchmark-baseline.json
#   ./container.bash -b | --build    force rebuild the image (after editing requirements.txt or the Dockerfile)
#   ./container.bash -s | --stop     stop the container (keeps the image/volumes)
#   ./container.bash -d | --delete   remove the container
#
# A running, up-to-date container is reused as is (no `docker compose up` on repeat calls).
# macOS (incl. Apple Silicon): works as is through Docker Desktop. If the image fails to build on an M-series Mac
# because a pinned wheel has no linux/arm64 build, add `platform: linux/amd64` under the `hack` service in
# docker/docker-compose.yaml to build/run under Rosetta emulation (slower, but correct).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RED='\033[0;31m'; CYAN='\033[0;36m'; GREEN='\033[0;32m'; YEL='\033[0;33m'; NC='\033[0m'
info() { echo -e "${CYAN}-> $*${NC}"; }; ok() { echo -e "${GREEN}OK $*${NC}"; }; warn() { echo -e "${YEL}!! $*${NC}"; }
die() { echo -e "${RED}XX $*${NC}" >&2; exit 1; }

MODE=attach; FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -u|--up) MODE=up ;;
    -t|--terminal) MODE=shell ;;
    exec) shift; MODE=exec; break ;;
    -b|--build) FORCE=1 ;;
    -s|--stop) MODE=stop ;;
    -d|--delete) MODE=delete ;;
    *) die "unknown option $1 (see -h)" ;;
  esac; shift
done
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not found"

HOST_UID="$(id -u)"; HOST_GID="$(id -g)"
# root, and Git Bash/Cygwin on Windows (Docker Desktop does not map IDs; id -u is a Windows SID there): use 1000
[[ "$HOST_UID" == 0 || "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]] && { HOST_UID=1000; HOST_GID=1000; }
export HOST_UID HOST_GID
export MSYS_NO_PATHCONV=1   # Git Bash/Cygwin would otherwise rewrite /work-style container paths before Docker sees them
C=(docker compose -f docker/docker-compose.yaml)
CNAME=feyer_hackathon

case "$MODE" in
  stop) "${C[@]}" stop; exit 0 ;;
  delete) "${C[@]}" down; exit 0 ;;
esac

if [[ $FORCE == 1 || -z "$(docker image inspect -f '{{.Id}}' feyer-hackathon:dev 2>/dev/null)" ]]; then
  info "building image feyer-hackathon:dev (first build is large, pulls jax/jaxlib etc.)"
  "${C[@]}" build || die "image build failed"
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" != "true" ]]; then
  info "starting container"
  "${C[@]}" up -d >/dev/null || die "docker compose up failed"
fi

case "$MODE" in
  up) ok "container is up; shell: ./container.bash -t" ;;
  attach)
    if command -v code >/dev/null 2>&1; then
      cid="$(docker inspect -f '{{.Id}}' "$CNAME")"
      hex="$(printf '%s' "$cid" | od -An -tx1 | tr -d ' \n')"
      info "opening VS Code attached..."
      code --folder-uri "vscode-remote://attached-container+${hex}/work"
    else
      warn "'code' CLI not found (install the VS Code 'code' shell command) - opening a shell instead"
      exec docker exec -it "$CNAME" bash
    fi
    ;;
  shell) exec docker exec -it "$CNAME" bash ;;
  exec) exec docker exec -it "$CNAME" "$@" ;;
esac
