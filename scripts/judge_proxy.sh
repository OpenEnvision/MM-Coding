#!/usr/bin/env bash
# (Re)start the WeaveBench judge's retry proxy, detached, on 127.0.0.1:<weavebench.judge.proxy_port>.
# The judge profile's provider baseUrl points here; the proxy forwards to $WEAVEBENCH_JUDGE_BASE_URL
# (the variable named by weavebench.judge.base_url_env).
#   scripts/judge_proxy.sh          start / restart
#   scripts/judge_proxy.sh stop     stop
set -eu
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
. "$here/env.sh"
rv_load_credentials
action="${1:-start}"
PORT="${PORT:-$(rv_judge proxy_port)}"
url_env="$(rv_judge base_url_env)"
UPSTREAM="${!url_env:?$url_env not set}"
LOG="$RV_DATA/logs/weavebench_judge_proxy.log"
old=$(ss -ltnp "sport = :$PORT" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)
[ -n "$old" ] && kill "$old" && sleep 1
[ "$action" = stop ] && { echo "stopped judge proxy (:$PORT)"; exit 0; }
mkdir -p "$RV_DATA/logs"
cd "$RV_OSWORLD"
setsid nohup .venv/bin/python -m mmcoding.benchmarks.weavebench_judge_proxy --upstream "$UPSTREAM" --port "$PORT" \
  >> "$LOG" 2>&1 < /dev/null &
sleep 2
curl -sf "127.0.0.1:$PORT/_stats" >/dev/null && echo "judge proxy on :$PORT (log: $LOG)"
