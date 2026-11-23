#!/usr/bin/env bash
# WeaveBench with one harness, judged by WeaveBench's OpenClaw agent-as-judge running the user's
# judge model through its local retry proxy.
#   scripts/run_weavebench.sh --harness mm_coding --model MODEL [--tasks all|@file.json|ID,ID] \
#       --num_envs N --result_dir DIR [--network-guard] [--judge_model JUDGE] [--no_judge] [--rejudge]
#
# Judge (once): set WEAVEBENCH_JUDGE_MODEL (or weavebench.judge_model in configs/protocol.json),
# WEAVEBENCH_JUDGE_BASE_URL and WEAVEBENCH_JUDGE_API_KEY, then
#   scripts/render_judge_profile.py && scripts/judge_proxy.sh
# The runner checks the profile and the proxy before any VM starts.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
rv_load_credentials

export AJ_OPENCLAW_BIN="${AJ_OPENCLAW_BIN:-$(command -v openclaw || true)}"
export WB_TASKS_ROOT="${WB_TASKS_ROOT:-$RV_DATA/assets/weavebench/tasks}"

[[ -f "${RV_WEAVEBENCH_IMAGE:-$RV_DATA/images/weavebench/Ubuntu_rv.qcow2}" ]] || { echo "WeaveBench VM image missing (docker/IMAGES.md)" >&2; exit 2; }
[[ -d "$WB_TASKS_ROOT" ]] || { echo "WeaveBench tasks missing: $WB_TASKS_ROOT (docker/IMAGES.md)" >&2; exit 2; }
[[ -d "${RV_ASSET_DIR:-$RV_DATA/assets/sandbox}/wheelhouse" ]] || { echo "sandbox assets missing: python3 scripts/fetch_sandbox_assets.py" >&2; exit 2; }
if [[ " $* " != *" --no_judge "* ]]; then
  [[ -n "$AJ_OPENCLAW_BIN" ]] || { echo "openclaw not on PATH (judge; npm install -g openclaw)" >&2; exit 2; }
fi
rv_require_host_network

cd "$RV_OSWORLD"
exec uv run --no-sync python -m mmcoding.benchmarks.weavebench "$@"
