#!/usr/bin/env bash
# Commit and push one phase of work.
#
# Usage:
#   ./scripts/push.sh 1 "acquire and verify the raw corpus"
#
# Produces a commit message of the form "phase 1: acquire and verify the raw corpus",
# then pushes the current branch to origin. Refuses to run if there is nothing staged,
# and shows what it is about to commit so nothing goes out unseen.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: ./scripts/push.sh <phase-number> <short message>" >&2
  exit 64
fi

PHASE="$1"; shift
MESSAGE="$*"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> repository: $REPO_ROOT"
git add -A

NOTHING_STAGED=0
if git diff --cached --quiet; then
  NOTHING_STAGED=1
  echo "==> nothing new to commit, working tree matches HEAD"
  echo "==> checking for commits not yet on origin"
fi

if [ "$NOTHING_STAGED" -eq 1 ]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if git rev-parse --verify --quiet "origin/${BRANCH}" >/dev/null; then
    AHEAD="$(git rev-list --count "origin/${BRANCH}..HEAD")"
  else
    AHEAD="$(git rev-list --count HEAD)"
    echo "    origin/${BRANCH} does not exist yet, this will be the first push"
  fi
  if [ "$AHEAD" -eq 0 ]; then
    echo "already up to date with origin. Nothing to do."
    exit 0
  fi
  echo "    ${AHEAD} commit(s) to push:"
  git log --oneline "@{u}..HEAD" 2>/dev/null || git log --oneline -n "$AHEAD"
  echo
  echo "==> pushing ${BRANCH} to origin"
  git push -u origin "${BRANCH}"
  echo "==> done."
  exit 0
fi

echo "==> files in this commit:"
git diff --cached --name-status

echo
echo "==> guard: confirming no email data is being committed"
if git diff --cached --name-only | grep -E '^data/' ; then
  echo "REFUSING: files under data/ are staged. That directory is meant to be git ignored." >&2
  echo "Fix .gitignore or unstage them, then rerun." >&2
  exit 1
fi
echo "clean, no data/ paths staged"

echo
git commit -q -m "phase ${PHASE}: ${MESSAGE}"
echo "==> committed: $(git log -1 --oneline)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "==> pushing ${BRANCH} to origin"
git push -u origin "${BRANCH}"
echo "==> done. phase ${PHASE} is live."
