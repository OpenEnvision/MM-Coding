#!/usr/bin/env bash
# Agents' Last Exam (v1.1, Linux CPU tasks) with one harness, in the ALE checkout's venv.
#   scripts/run_ale.sh --harness mm_coding --model MODEL [--tasks all|@file.json|domain/task,...] \
#       --num_envs N --result_dir DIR [--network-guard] [--regrade]
# Grading uses ALE's own grading-model config ($RV_ALE/secret/eval_time/openai.env, gitignored
# upstream; see the ALE README). Its values are never printed.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
rv_load_credentials
# The reference-archive password is public (ALE docs, secret/.env.example).
ALE_REFERENCE_ARCHIVE_PASSWORD="${ALE_REFERENCE_ARCHIVE_PASSWORD:-$(sed -n 's/^ALE_REFERENCE_ARCHIVE_PASSWORD=//p' "$RV_ALE/secret/.env.example")}"
export ALE_REFERENCE_ARCHIVE_PASSWORD

[[ -f "$RV_ALE/secret/eval_time/openai.env" ]] || { echo "ALE grading config missing: $RV_ALE/secret/eval_time/openai.env" >&2; exit 2; }
[[ -f "${RV_ALE_QEMU_ROOT:-$RV_DATA/images/ale/qemu}/images/ale-ubuntu22-v1.1.qcow2" ]] \
  || echo "note: ALE disk not cached yet; the QEMU provider downloads it on first use (about 93 GB; docker/fetch_ale_disk.py)" >&2
[[ -d "${RV_ASSET_DIR:-$RV_DATA/assets/sandbox}/wheelhouse" ]] || { echo "sandbox assets missing: python3 scripts/fetch_sandbox_assets.py" >&2; exit 2; }

cd "$RV_ALE"  # ALE task modules resolve paths relative to the checkout
exec uv run --no-sync python -m mmcoding.benchmarks.ale "$@"
