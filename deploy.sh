#!/usr/bin/env bash
# Publishes course/ to github.com/wynch/ml-course as a snapshot: copies it into
# the git-ignored local clone deploy/ and makes one commit there. Pushes only
# with --push; Vincent runs or allows every push.
#
#   course/deploy.sh "<message>"            commit, print the push command
#   course/deploy.sh "<message>" --push     commit and push
#   course/deploy.sh "<message>" --reset    once: replace the public history
#                                           with one fresh commit (2026-10-09)
#
# Never publish with `git subtree split`: the old split commits hold private data.
set -euo pipefail

REPO="git@github.com:wynch/ml-course.git"
OLD_HEAD="ef8ceeb"   # public head before the history reset; pins the force push
AUTHOR_NAME="Vincent Barthelemy"
AUTHOR_EMAIL="wynch@users.noreply.github.com"
TODAY="$(date +%F)"

usage() {
  echo "usage: course/deploy.sh <message> [--push] [--reset]" >&2
  exit 2
}

message="" push=0 reset=0
for arg in "$@"; do
  case "$arg" in
    --push) push=1 ;;
    --reset) reset=1 ;;
    -*) usage ;;
    *) [ -z "$message" ] || usage; message="$arg" ;;
  esac
done
[ -n "$message" ] || usage

cd "$(dirname "$0")/.."   # the ml project root; every path below is relative to it
[ -f course/index.html ] && [ -f AGENTS.md ] || { echo "deploy: not the ml project" >&2; exit 1; }

if [ ! -d deploy/.git ]; then
  echo "deploy: cloning $REPO into deploy/"
  git clone --quiet "$REPO" deploy
fi
git -C deploy config user.name "$AUTHOR_NAME"
git -C deploy config user.email "$AUTHOR_EMAIL"
git -C deploy config commit.gpgsign false
git -C deploy fetch --quiet origin
git -C deploy checkout --quiet main

if [ -n "$(git status --porcelain -- course)" ]; then
  echo "deploy: warning: course/ has uncommitted changes; they will be published" >&2
fi

copy() {
  # Ignored files (venvs, model weights, build output) stay out; deploy/ carries
  # the same .gitignore files, so git there would skip them anyway.
  rsync -a --delete --exclude .git --exclude-from=course/.gitignore course/ deploy/
  git -C deploy add -A
}

report() {
  local files bytes
  files="$(git -C deploy ls-files | wc -l | tr -d ' ')"
  bytes="$(git -C deploy ls-tree -r -l HEAD | awk '{s += $4} END {print s}')"
  echo "deploy: commit $(git -C deploy rev-parse --short HEAD), $files files, $((bytes / 1024)) KB"
}

if [ "$reset" = 1 ]; then
  if ! git -C deploy merge-base --is-ancestor "$OLD_HEAD" main 2>/dev/null; then
    echo "deploy: refusing --reset: main in deploy/ does not descend from $OLD_HEAD; the reset already happened" >&2
    exit 1
  fi
  remote="$(git -C deploy rev-parse origin/main)"
  if [ "$remote" != "$(git -C deploy rev-parse "$OLD_HEAD^{commit}")" ]; then
    echo "deploy: refusing --reset: origin/main is ${remote:0:7}, not $OLD_HEAD; the remote moved" >&2
    exit 1
  fi
  git -C deploy checkout --quiet --orphan fresh
  copy
  git -C deploy commit --quiet -m "Publish $TODAY: fresh history ($message)"
  git -C deploy branch -M main
  report
  cmd="git -C deploy push --force-with-lease=main:$OLD_HEAD origin main"
else
  if ! git -C deploy merge-base --is-ancestor origin/main main; then
    echo "deploy: refusing: main in deploy/ does not contain origin/main." >&2
    echo "deploy: if the reset is not pushed yet, push it first:" >&2
    echo "  git -C deploy push --force-with-lease=main:$OLD_HEAD origin main" >&2
    exit 1
  fi
  copy
  if git -C deploy diff --cached --quiet; then
    if [ "$(git -C deploy rev-parse main)" = "$(git -C deploy rev-parse origin/main)" ]; then
      echo "deploy: nothing to publish"
      exit 0
    fi
    echo "deploy: nothing new to commit; main is ahead of origin/main"
  else
    git -C deploy commit --quiet -m "Publish $TODAY: $message"
  fi
  report
  cmd="git -C deploy push origin main"
fi

if [ "$push" = 1 ]; then
  $cmd
  git -C deploy branch --quiet --set-upstream-to=origin/main main
  echo "deploy: pushed; the Pages workflow redeploys https://wynch.github.io/ml-course/"
else
  echo "deploy: not pushed. To publish, run from the ml folder:"
  echo "  $cmd"
fi
