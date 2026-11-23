#!/usr/bin/env python3
"""Render the WeaveBench judge's OpenClaw profile from docker/weavebench_judge/openclaw.template.json.

The rendered ``openclaw.json`` holds the judge key, so it lives only under
``$RV_DATA/judge/weavebench/template/profiles/<judge>/`` (never in the repository). The model talks
to the local retry proxy (scripts/judge_proxy.sh), which forwards to the judge endpoint. An
existing profile keeps its gateway token (used on loopback only); a new one gets a random token.
The template workspace is copied to ``$RV_DATA/judge/weavebench/template/workspace`` if missing.

usage: scripts/render_judge_profile.py [JUDGE]
  JUDGE defaults to $WEAVEBENCH_JUDGE_MODEL, else weavebench.judge_model in configs/protocol.json.
  The key is read from $WEAVEBENCH_JUDGE_API_KEY (weavebench.judge.api_key_env); after
  `. scripts/env.sh && rv_load_credentials` it may come from the credentials file.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmcoding.benchmarks import weavebench_judge  # noqa: E402

if __name__ == "__main__":
    weavebench_judge.main(["render", *sys.argv[1:]])
