#!/bin/bash
# Enter the sandbox. Usage:
#   ./enter.sh           -> bash shell
#   ./enter.sh claude    -> Claude Code with --dangerously-skip-permissions
#   ./enter.sh <cmd...>  -> run any command
set -e
cd "$(dirname "$0")"

# Mount the repo at its host path inside the container so file references
# (e.g. /home/you/proj/foo.py:10) match the host and stay clickable.
export HOST_REPO="$(cd .. && pwd)"

docker volume create hf-cache >/dev/null

if [ "$1" = "claude" ]; then
    shift
    exec docker compose run --rm dev claude --dangerously-skip-permissions "$@"
fi
exec docker compose run --rm dev "${@:-bash}"
