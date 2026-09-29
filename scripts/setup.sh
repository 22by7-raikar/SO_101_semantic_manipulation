#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-core}"
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv 0.11.14 first; see docs/ENVIRONMENT.md.' >&2
  exit 1
fi
sync_args=(sync --locked)
doctor_args=(so101-doctor)
case "$profile" in
  core) ;;
  vision) sync_args+=(--extra vision); doctor_args+=(--vision) ;;
  robot)
    if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
      echo 'The robot profile requires Ubuntu x86_64. Use core or vision on Apple Silicon.' >&2
      exit 1
    fi
    sync_args+=(--extra vision --extra hardware)
    doctor_args+=(--gpu --hardware)
    ;;
  *) echo 'Usage: bash scripts/setup.sh [core|vision|robot]' >&2; exit 2 ;;
esac
uv "${sync_args[@]}"
uv run --no-sync "${doctor_args[@]}"
printf '\nReady. Run: uv run --no-sync so101-demo\n'
