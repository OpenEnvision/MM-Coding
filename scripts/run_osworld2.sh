#!/usr/bin/env bash
# OSWorld 2.0 (OSWorld-V2 release osworld-v2.1) with one harness. Every harness of a comparison goes
# through this launcher, so benchmark release, evaluator and user simulator are identical.
#   scripts/run_osworld2.sh --harness mm_coding --model MODEL [--tasks all|@file.json|001,002] \
#       --num_envs N --result_dir DIR [--network-guard] [--resume]
#
# Evaluator and user simulator (upstream OSWORLD_EVAL_MODEL_* / OSWORLD_USER_SIM_* variables):
#   OSWORLD_EVAL_MODEL_NAME         evaluator model (or osworld2.evaluator_model in configs/protocol.json)
#   OSWORLD_USER_SIM_MODEL          user simulator model (or osworld2.user_simulator_model)
#   OSWORLD_EVAL_MODEL_PROVIDER     upstream client dialect: openai_compatible (default) or anthropic
#   OSWORLD_EVAL_MODEL_BASE_URL     endpoint (required for openai_compatible)
#   OSWORLD_EVAL_MODEL_API_KEY_ENV  name of the variable holding the key (default OSWORLD_EVAL_API_KEY)
# The simulator inherits provider, endpoint and key variable unless its own OSWORLD_USER_SIM_* are set.
# Both use the upstream default thinking and output caps.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
rv_load_credentials
B=osworld2

# ---- pinned benchmark release
export OSWORLD_BENCHMARK_RELEASE="${OSWORLD_BENCHMARK_RELEASE:-$(rv_protocol $B benchmark_release)}"
export OSWORLD_FILE_BASE_URL="${OSWORLD_FILE_BASE_URL:-$RV_DATA/assets/osworld_v2/osworld_v2_assets_v2.1}"
export OSWORLD_CLIENT_PASSWORD="$(rv_protocol $B client_password)"

# ---- evaluator and user simulator, identical for every harness
export OSWORLD_EVAL_MODEL_PROVIDER="${OSWORLD_EVAL_MODEL_PROVIDER:-$(rv_protocol $B evaluator_provider)}"
export OSWORLD_EVAL_MODEL_NAME="${OSWORLD_EVAL_MODEL_NAME:-$(rv_protocol $B evaluator_model)}"
export OSWORLD_EVAL_MODEL_API_KEY_ENV="${OSWORLD_EVAL_MODEL_API_KEY_ENV:-$(rv_protocol $B evaluator_api_key_env)}"
export OSWORLD_USER_SIM_PROVIDER="${OSWORLD_USER_SIM_PROVIDER:-$OSWORLD_EVAL_MODEL_PROVIDER}"
export OSWORLD_USER_SIM_MODEL="${OSWORLD_USER_SIM_MODEL:-$(rv_protocol $B user_simulator_model)}"
export OSWORLD_USER_SIM_API_KEY_ENV="${OSWORLD_USER_SIM_API_KEY_ENV:-$OSWORLD_EVAL_MODEL_API_KEY_ENV}"
[[ -n "$OSWORLD_EVAL_MODEL_NAME" ]] || { echo "Set the evaluator model: export OSWORLD_EVAL_MODEL_NAME or osworld2.evaluator_model in configs/protocol.json" >&2; exit 2; }
[[ -n "$OSWORLD_USER_SIM_MODEL" ]] || { echo "Set the user simulator model: export OSWORLD_USER_SIM_MODEL or osworld2.user_simulator_model in configs/protocol.json" >&2; exit 2; }
if [[ "$OSWORLD_USER_SIM_PROVIDER" == "$OSWORLD_EVAL_MODEL_PROVIDER" && -z "${OSWORLD_USER_SIM_BASE_URL:-}" \
      && -n "${OSWORLD_EVAL_MODEL_BASE_URL:-}" ]]; then
  OSWORLD_USER_SIM_BASE_URL="$OSWORLD_EVAL_MODEL_BASE_URL"
fi

rv_osworld_model_route() {  # rv_osworld_model_route OSWORLD_EVAL_MODEL|OSWORLD_USER_SIM
  local provider_var="${1}_PROVIDER" base_var="${1}_BASE_URL"
  local key_env_var="${1}_API_KEY_ENV" literal_var="${1}_API_KEY" key_env
  case "${!provider_var}" in
    openai_compatible) [[ -n "${!base_var:-}" ]] || { echo "Set $base_var for the openai_compatible dialect" >&2; return 2; } ;;
    anthropic) ;;
    *) echo "$provider_var must be openai_compatible or anthropic" >&2; return 2 ;;
  esac
  # An upstream literal key overrides every *_API_KEY_ENV, including the simulator's; keys are
  # therefore passed only by variable name.
  [[ -z "${!literal_var:-}" ]] || { echo "Unset $literal_var; name the key variable in $key_env_var instead" >&2; return 2; }
  key_env="${!key_env_var}"
  [[ "$key_env" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || { echo "$key_env_var must name an environment variable" >&2; return 2; }
  [[ -n "${!key_env:-}" ]] || { echo "Credential referenced by $key_env_var ($key_env) is not set" >&2; return 2; }
  [[ -z "${!base_var:-}" ]] || export "$base_var"
  export "$key_env"
}
rv_osworld_model_route OSWORLD_EVAL_MODEL
rv_osworld_model_route OSWORLD_USER_SIM
# The upstream client applies the evaluator effort to simulator calls too; the protocol keeps the
# upstream default thinking for both.
for effort_var in OSWORLD_EVAL_MODEL_REASONING_EFFORT OSWORLD_USER_SIM_REASONING_EFFORT; do
  [[ -z "${!effort_var:-}" ]] || { echo "Unset $effort_var: the protocol uses the upstream default thinking" >&2; exit 2; }
done

# ---- guards
tag="$(git -C "$RV_OSWORLD" describe --tags --exact-match 2>/dev/null || true)"
[[ "$tag" == "$OSWORLD_BENCHMARK_RELEASE" ]] || { echo "OSWorld-V2 checkout is at '${tag:-<no tag>}', expected $OSWORLD_BENCHMARK_RELEASE" >&2; exit 2; }
[[ -d "$OSWORLD_FILE_BASE_URL" ]] || { echo "OSWorld 2.0 asset snapshot missing: $OSWORLD_FILE_BASE_URL (docker/IMAGES.md)" >&2; exit 2; }
[[ -f "${RV_OSWORLD_V2_IMAGE:-$RV_DATA/images/osworld_v2/osworld-v2-ubuntu-x86.qcow2}" ]] || { echo "OSWorld 2.0 VM image missing (docker/IMAGES.md)" >&2; exit 2; }
[[ -d "${RV_ASSET_DIR:-$RV_DATA/assets/sandbox}/wheelhouse" ]] || { echo "sandbox assets missing: python3 scripts/fetch_sandbox_assets.py" >&2; exit 2; }
rv_require_host_network

cd "$RV_OSWORLD"  # the official runner resolves task files relative to its checkout
exec uv run --no-sync python -m mmcoding.benchmarks.osworld2 "$@"
