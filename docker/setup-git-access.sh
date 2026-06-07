#!/bin/bash
# One-time GitHub access setup, run from the HOST (uses the host's gh login):
#   ./docker/setup-git-access.sh [--public]
# Creates the GitHub repo (private by default), generates an SSH key inside the
# container's home volume, and registers it as a deploy key scoped to only this
# repo. Covers git push/pull inside the sandbox; not gh API calls (PRs etc.).
set -e
cd "$(dirname "$0")"

repo=$(basename "$(cd .. && pwd)")
title="$repo-sandbox"

gh auth status >/dev/null
owner=$(gh api user -q .login)
gh repo view "$owner/$repo" >/dev/null 2>&1 || gh repo create "$owner/$repo" "${1:---private}"

# key + pinned github host key + git identity, all in the home volume (idempotent)
docker volume create hf-cache >/dev/null
name=$(git config user.name || true)
email=$(git config user.email || true)
docker compose run --rm -T dev bash -c "
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_ed25519 -C '$title'
    grep -q github.com ~/.ssh/known_hosts 2>/dev/null || ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null
    [ -n '$name' ] && git config --global user.name '$name'
    [ -n '$email' ] && git config --global user.email '$email'
    true
"

if ! gh repo deploy-key list -R "$owner/$repo" | grep -q "$title"; then
    pub=$(mktemp)
    docker compose run --rm -T dev cat /home/dev/.ssh/id_ed25519.pub > "$pub"
    gh repo deploy-key add "$pub" -R "$owner/$repo" --allow-write --title "$title"
    rm "$pub"
fi

git -C .. remote get-url origin >/dev/null 2>&1 || \
    git -C .. remote add origin "git@github.com:$owner/$repo.git"

echo "Done. Push from inside the sandbox (key lives in the home volume):"
echo "  ./docker/enter.sh, then: git push -u origin main"
