#!/usr/bin/env bash
# Run the optional network guard proxy in the foreground (needed only with --network-guard).
# Prefer running it as the systemd unit named by "service" in the guard config, e.g.
#   systemd-run --unit rv-guard.service --property=Restart=on-failure scripts/guard_proxy.sh
# Requires mitmproxy on the host; see mmcoding/host/guard_addon.py.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
config="${RV_GUARD_CONFIG:-}"
if [[ -z "$config" ]]; then
  config="$RV_REPO/configs/guard.json"
  [[ -f "$config" ]] || config="$RV_REPO/configs/guard.example.json"
fi
export RV_GUARD_CONFIG="$config"
port="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["port"])' "$config")"
exec mitmdump --mode transparent --listen-port "$port" --set confdir="${MITMPROXY_CONFDIR:-$HOME/.mitmproxy}" \
  --set block_global=false -s "$RV_REPO/mmcoding/host/guard_addon.py"
