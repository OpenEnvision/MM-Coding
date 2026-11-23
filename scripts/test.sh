#!/usr/bin/env bash
# Offline test suite (no VM, network or model call), in the OSWorld-V2 venv.
#   scripts/test.sh [pytest args]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
cd "$RV_OSWORLD"
PYTHONPATH="$RV_REPO${PYTHONPATH:+:$PYTHONPATH}" exec uv run --no-sync --with pytest pytest -q -p no:cacheprovider "$RV_REPO/tests" "$@"
