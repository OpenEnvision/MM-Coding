#!/usr/bin/env bash
# Prepare the upstream checkouts and the two Python environments of the project.
#
#   1. clone OSWorld-V2, WeaveBench and agents-last-exam into $RV_THIRD_PARTY at the commits pinned
#      in configs/upstream.json (existing checkouts are only checked, never changed);
#   2. OSWorld-V2 venv : OSWorld 2.0 and WeaveBench runners, WeaveBench judge, tests
#      ALE venv        : ALE runner (ALE's own lock)
#      Each checkout keeps its own uv.lock; we add only editable installs (the upstream package,
#      WeaveBench into the OSWorld-V2 venv, this repository into both, --no-deps) and paramiko
#      (guest SSH transport);
#   3. fetch the sandbox assets (wheels and debs of configs/sandbox/assets.sha256) into $RV_DATA.
#
# VM images, benchmark task data and container images are separate downloads: docker/IMAGES.md.
#   scripts/setup.sh [--no-assets]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
PARAMIKO="paramiko==4.0.0"
command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 2; }

echo "== upstream checkouts ($RV_THIRD_PARTY)"
mkdir -p "$RV_THIRD_PARTY"
while IFS=$'\t' read -r name url ref commit; do
  dir="$RV_THIRD_PARTY/$name"
  if [[ ! -d $dir/.git ]]; then
    git clone --quiet "$url" "$dir"
    git -C "$dir" fetch --quiet --tags origin "$ref"
    git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
  fi
  head=$(git -C "$dir" rev-parse --short=7 HEAD)
  [[ $head == "$commit" ]] || { echo "$dir is at $head, expected $commit" >&2; exit 2; }
  echo "$name $head"
done < <(python3 -c 'import json,sys
for k,v in json.load(open(sys.argv[1])).items():
    if not k.startswith("_"): print(k, v["url"], v["ref"], v["commit"], sep="\t")' "$RV_REPO/configs/upstream.json")

echo "== OSWorld-V2 venv ($RV_OSWORLD)"
cd "$RV_OSWORLD"
[[ -d .venv ]] || uv sync --frozen
uv pip install --python .venv/bin/python --no-deps -e . -e "$RV_WEAVEBENCH" -e "$RV_REPO"
uv pip install --python .venv/bin/python "$PARAMIKO"

echo "== ALE venv ($RV_ALE)"
cd "$RV_ALE"
[[ -d .venv ]] || uv sync --frozen
uv pip install --python .venv/bin/python --no-deps -e . -e "$RV_REPO"
uv pip install --python .venv/bin/python "$PARAMIKO"

echo "== check"
"$RV_OSWORLD/.venv/bin/python" -c "import mmcoding, weavebench, desktop_env, hybrid_agents.deepseek, paramiko, yaml; print('OSWorld-V2 venv ok')"
"$RV_ALE/.venv/bin/python" -c "import mmcoding, ale_run, paramiko, yaml; print('ALE venv ok')"

if [[ ${1:-} != --no-assets ]]; then
  echo "== sandbox assets"
  python3 "$RV_REPO/scripts/fetch_sandbox_assets.py"
fi
