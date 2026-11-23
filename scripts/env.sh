#!/usr/bin/env bash
# Shared environment of every script. Source it; it prints nothing and never echoes secrets.
# Locations mirror mmcoding/paths.py; override any of them before sourcing.
RV_REPO="${RV_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RV_REPO
export RV_THIRD_PARTY="${RV_THIRD_PARTY:-$RV_REPO/third_party}"
export RV_OSWORLD="${RV_OSWORLD:-$RV_THIRD_PARTY/OSWorld-V2}"
export RV_WEAVEBENCH="${RV_WEAVEBENCH:-$RV_THIRD_PARTY/WeaveBench}"
export RV_ALE="${RV_ALE:-$RV_THIRD_PARTY/agents-last-exam}"
export RV_DATA="${RV_DATA:-$RV_REPO/data}"
export RV_MODEL_CREDENTIALS="${RV_MODEL_CREDENTIALS:-$HOME/.config/mm-coding/credentials.env}"
export PATH="$HOME/.local/bin:$PATH"

# Export the variables of the credentials file (RV_LLM_BASE_URL, RV_LLM_API_KEY, judge and
# evaluator keys, ...). Without the file, credentials already in the environment are used.
rv_load_credentials() {
  if [[ -r "$RV_MODEL_CREDENTIALS" ]]; then
    set -a
    # shellcheck disable=SC1090
    . "$RV_MODEL_CREDENTIALS"
    set +a
  elif [[ -z "${RV_LLM_API_KEY:-}" ]]; then
    echo "no credentials: create $RV_MODEL_CREDENTIALS or export RV_LLM_BASE_URL / RV_LLM_API_KEY" >&2
    return 2
  fi
}

# rv_protocol BENCH KEY -> configs/protocol.json value (empty when null)
rv_protocol() {
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]]; print("" if v is None else v)' \
    "$RV_REPO/configs/protocol.json" "$1" "$2"
}

# rv_judge KEY -> configs/protocol.json weavebench.judge.<KEY>
rv_judge() {
  (cd "$RV_REPO" && python3 -m mmcoding.benchmarks.weavebench_judge field "$1")
}

# The guests reach the network through NAT on the host (Docker provider and the optional network
# guard need iptable_nat). BBR is recommended for long-distance model endpoints: with cubic,
# requests carrying images can stall for minutes on lossy paths.
rv_require_host_network() {
  grep -q '^iptable_nat ' /proc/modules || { echo "kernel module iptable_nat not loaded (modprobe ip_tables iptable_nat)" >&2; return 2; }
  [[ "$(cat /proc/sys/net/ipv4/tcp_congestion_control 2>/dev/null)" == bbr ]] \
    || echo "note: host TCP congestion control is not bbr (sysctl net.ipv4.tcp_congestion_control=bbr recommended)" >&2
  return 0
}
