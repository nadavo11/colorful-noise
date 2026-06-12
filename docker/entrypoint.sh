#!/bin/bash
# One-time git bootstrap; settings persist in the home named volume.
set -e

if [ ! -f "$HOME/.gitconfig" ]; then
    git config --global user.name "${GIT_USER_NAME:-Shimon Malnick}"
    git config --global user.email "${GIT_USER_EMAIL:-shimonmalnick@gmail.com}"
    git config --global credential.helper store
    git config --global init.defaultBranch main
fi
git config --global --replace-all safe.directory "${HOST_REPO:-/workspace}"

# Prefer the native Claude Code install (~/.local/bin, persisted in the home
# volume) over the root-owned npm install baked into the image.
export PATH="$HOME/.local/bin:$PATH"

exec "$@"
