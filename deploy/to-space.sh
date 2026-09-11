#!/usr/bin/env bash
# Deploy the current commit to a Hugging Face Space.
#
#   ./deploy/to-space.sh https://huggingface.co/spaces/<user>/<space>
#
# The Space gets its own README, because Hugging Face reads the Space's
# configuration from that file's front matter and the project README has no
# reason to carry it.
#
# The push is a force-push to an orphan history. A Space is a deployment
# target, not a shared branch -- there is nothing there to preserve.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <space-git-url>" >&2
    exit 1
fi

SPACE_URL="$1"
SOURCE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

# Export the committed tree, so an uncommitted experiment cannot be deployed
# by accident.
git -C "$SOURCE_DIR" archive HEAD | tar -x -C "$STAGING"
cp "$SOURCE_DIR/deploy/space-README.md" "$STAGING/README.md"

COMMIT="$(git -C "$SOURCE_DIR" rev-parse --short HEAD)"

cd "$STAGING"
git init -q -b main
git add -A
git -c user.email="deploy@local" -c user.name="deploy" \
    commit -q -m "Deploy $COMMIT"
git push -q --force "$SPACE_URL" main

echo "Deployed $COMMIT to $SPACE_URL"
echo "Set GEMINI_API_KEY in the Space's Settings > Variables and secrets."
