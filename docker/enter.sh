#!/bin/bash
# Enter the sandbox. Usage:
#   ./enter.sh           -> bash shell
#   ./enter.sh claude    -> Claude Code with --dangerously-skip-permissions
#   ./enter.sh <cmd...>  -> run any command
set -e
cd "$(dirname "$0")"

docker volume create hf-cache >/dev/null

if [ "$1" = "claude" ]; then
    shift
    exec docker compose run --rm dev claude --dangerously-skip-permissions "$@"
fi
exec docker compose run --rm dev "${@:-bash}"
