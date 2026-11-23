# WeaveBench judge template

The OpenClaw profile template and starter workspace of WeaveBench's trajectory-aware
agent-as-judge, copied from upstream WeaveBench (`README.upstream.md` is the upstream note).

`openclaw.template.json` contains placeholders only. `scripts/render_judge_profile.py` fills them
from `configs/protocol.json` (`weavebench.judge`) and the environment, and writes the profile to
`$RV_DATA/judge/weavebench/template/profiles/<judge>/openclaw.json` (mode 0600; it holds the key):

| placeholder | value |
|---|---|
| `PLACEHOLDER_JUDGE_MODEL` | the judge model (`--judge_model`, `$WEAVEBENCH_JUDGE_MODEL` or `weavebench.judge_model`) |
| `PLACEHOLDER_PROVIDER` | `weavebench.judge.provider` (an OpenClaw provider id) |
| `PLACEHOLDER_API` | `weavebench.judge.api` or `$WEAVEBENCH_JUDGE_API` (OpenClaw request adapter, e.g. `openai-responses`, `openai-completions`) |
| `PLACEHOLDER_API_KEY` | the value of `$WEAVEBENCH_JUDGE_API_KEY` |
| `PLACEHOLDER_PROXY_URL` | `http://127.0.0.1:<weavebench.judge.proxy_port>/v1` (`scripts/judge_proxy.sh`, which forwards to `$WEAVEBENCH_JUDGE_BASE_URL`) |
| `PLACEHOLDER_WORKSPACE_DIR` | `$RV_DATA/judge/weavebench/template/workspace` (a copy of `workspace/`) |
| `PLACEHOLDER_GATEWAY_TOKEN` | a random token for OpenClaw's loopback gateway |

`contextWindow` and `maxTokens` of the model entry come from `weavebench.judge`.
