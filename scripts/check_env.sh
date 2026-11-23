#!/usr/bin/env bash
# Check that this machine is ready for a run: upstream commits, VM images, container images,
# sandbox assets, venvs, host network, credentials, judge profile.
#   scripts/check_env.sh [--hash]     --hash also re-computes the qcow2 sha256 values (minutes)
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
export PYTHONPATH="$RV_REPO${PYTHONPATH:+:$PYTHONPATH}"
OSWORLD_DOCKER=happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9
ALE_RUNNER=agentslastexam/ale-qemu:0.2.0
fail=0
ok() { printf '  ok   %s\n' "$*"; }
bad() { printf '  FAIL %s\n' "$*"; fail=1; }

echo "upstream checkouts (configs/upstream.json)"
while read -r name commit; do
  dir="$RV_THIRD_PARTY/$name"
  head=$(git -C "$dir" rev-parse --short=7 HEAD 2>/dev/null || echo missing)
  dirty=$(git -C "$dir" status --short --untracked-files=no 2>/dev/null | wc -l)
  [[ $head == "$commit" && $dirty == 0 ]] && ok "$name $head clean" || bad "$name at $head ($dirty changes), expected $commit"
done < <(python3 -c 'import json,sys
for k,v in json.load(open(sys.argv[1])).items():
    if not k.startswith("_"): print(k, v["commit"])' "$RV_REPO/configs/upstream.json")

echo "VM images ($RV_DATA/images, docker/IMAGES.md)"
for f in osworld_v2/osworld-v2-ubuntu-x86.qcow2 weavebench/Ubuntu_rv.qcow2 ale/qemu/images/ale-ubuntu22-v1.1.qcow2; do
  p="$RV_DATA/images/$f"
  if [[ ! -f $p ]]; then bad "$f missing"; continue; fi
  if [[ ${1:-} == --hash && -f $p.sha256 ]]; then
    (cd "$RV_DATA/images" && sha256sum -c --quiet "$f.sha256") && ok "$f sha256" || bad "$f sha256 mismatch"
  else ok "$f"; fi
done

echo "container images"
for img in "$OSWORLD_DOCKER" "$ALE_RUNNER"; do
  docker image inspect "$img" >/dev/null 2>&1 && ok "$img" || bad "$img not pulled"
done

echo "benchmark data"
[[ -d ${RV_ASSET_DIR:-$RV_DATA/assets/sandbox}/wheelhouse ]] && python3 "$RV_REPO/scripts/fetch_sandbox_assets.py" --check >/dev/null \
  && ok "sandbox assets" || bad "sandbox assets (python3 scripts/fetch_sandbox_assets.py)"
[[ -d ${OSWORLD_FILE_BASE_URL:-$RV_DATA/assets/osworld_v2/osworld_v2_assets_v2.1} ]] && ok "OSWorld 2.0 assets" || bad "OSWorld 2.0 assets (docker/IMAGES.md)"
compgen -G "$RV_OSWORLD/evaluation_examples/task_class/task_*.py" >/dev/null && ok "OSWorld 2.0 task classes" \
  || bad "OSWorld 2.0 task classes (scripts/tools/download_osworld_v2_tasks.py in $RV_OSWORLD)"
[[ -d ${WB_TASKS_ROOT:-$RV_DATA/assets/weavebench/tasks} ]] && ok "WeaveBench tasks" || bad "WeaveBench tasks (docker/IMAGES.md)"

echo "python environments"
"$RV_OSWORLD/.venv/bin/python" -c "import mmcoding, weavebench, desktop_env, hybrid_agents.deepseek, paramiko, yaml" 2>/dev/null \
  && ok "OSWorld-V2 venv" || bad "OSWorld-V2 venv (scripts/setup.sh)"
"$RV_ALE/.venv/bin/python" -c "import mmcoding, ale_run, paramiko, yaml" 2>/dev/null && ok "ALE venv" || bad "ALE venv (scripts/setup.sh)"

echo "host"
rv_require_host_network && ok "iptable_nat" || fail=1
key="${RV_SSH_KEY:-$HOME/.ssh/mm_coding_vm_ed25519}"
[[ -f $key && -f $key.pub ]] && ok "guest SSH key" || bad "guest SSH key $key (ssh-keygen -t ed25519 -N '' -f $key)"
rv_load_credentials 2>/dev/null && [[ -n ${RV_LLM_BASE_URL:-} && -n ${RV_LLM_API_KEY:-} ]] && ok "policy model credentials" \
  || bad "RV_LLM_BASE_URL / RV_LLM_API_KEY ($RV_MODEL_CREDENTIALS or environment)"
[[ -f $RV_ALE/secret/eval_time/openai.env ]] && ok "ALE grading config" || bad "ALE grading config ($RV_ALE/secret/eval_time/openai.env)"

echo "WeaveBench judge"
if judge="$(cd "$RV_REPO" && python3 - <<'PY' 2>&1
from mmcoding.benchmarks import weavebench_judge as wj
name, cfg = wj.judge()
problem = wj.problem(name, cfg)
print(problem or f"{cfg['provider']}/{name}")
raise SystemExit(bool(problem))
PY
)"; then ok "judge $judge"; else bad "$judge"; fi
command -v openclaw >/dev/null && ok "openclaw" || bad "openclaw not on PATH (npm install -g openclaw)"
exit $fail
