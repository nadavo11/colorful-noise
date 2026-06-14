#!/bin/bash
# Enter the sandbox. Usage:
#   ./enter.sh                  -> bash shell
#   ./enter.sh claude           -> Claude Code with --dangerously-skip-permissions
#   ./enter.sh <cmd...>         -> run any command
#   ./enter.sh --no-storage [...] -> skip the default host bind mounts, then run as above.
# By default host /storage/malnick (rw) and ~/.kube (rw, so runai/kubectl can reach the
# cluster and refresh tokens) are bind-mounted in. /storage/malnick needs the host sshfs
# mounted with -o allow_other (see ~/storage-automount).
set -e
cd "$(dirname "$0")"

# Mount the repo at its host path inside the container so file references
# (e.g. /home/you/proj/foo.py:10) match the host and stay clickable.
export HOST_REPO="$(cd .. && pwd)"

docker volume create hf-cache >/dev/null

vol=(-v /storage/malnick:/storage/malnick -v "$HOME/.kube:/home/dev/.kube")
if [ "$1" = "--no-storage" ]; then
    shift
    vol=()
fi

if [ "$1" = "claude" ]; then
    shift
    exec docker compose run --rm "${vol[@]}" dev claude --dangerously-skip-permissions "$@"
fi
exec docker compose run --rm "${vol[@]}" dev "${@:-bash}"
