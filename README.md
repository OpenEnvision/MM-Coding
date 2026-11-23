# Rethinking Vision in Computer-Use Agents

Code for *Rethinking Vision in Computer-Use Agents: Decoupling Perception from Action*.
[Project page](https://openenvision.org/Rethinking-CUA-Vision/)

MM-Coding treats computer use as a multimodal coding task. The policy acts through programs, and
it calls vision as a tool when it needs visual information. This repository contains the five
harnesses compared in the paper and the runners that evaluate them on OSWorld 2.0, WeaveBench and
Agents' Last Exam (ALE).

| Harness | `--harness` | Acts through | Perceives through |
|---|---|---|---|
| GUI | `gui` | one mouse or keyboard primitive per call | a screenshot after every action |
| OS-Helix | `os_helix` | one primitive or one Bash/Python block per step | a screenshot after every action, ten kept in context |
| **MM-Coding** | `mm_coding` | programs and file operations | screenshots and image files on request |
| Vision-Tool | `vision_tool` | programs and file operations | text answers of a separate, task-blind vision model |
| CLI | `cli` | programs and file operations | nothing beyond the text output of programs |

All harnesses share the policy model, the desktop, the task instruction, the budget of 500 model
calls and the official evaluator. They differ only in their tools and in the paragraph of the
system prompt that describes these tools. Details: [docs/harnesses.md](docs/harnesses.md).

## How it works

```
host                                         task VM (Ubuntu desktop, 1920x1080)
mmcoding.benchmarks.<bench>                  rv-run sandbox (own network namespace, filtered bus)
  └─ official benchmark pipeline   ── SSH ──▶  └─ DeepSeek Harness agent loop (worker_rv.py)
     (setup, evaluation)                           ├─ profiles/common + profiles/<harness>.patch.yml
  mmcoding.host.deploy                             ├─ shell / file tools   (all but GUI, OS-Helix)
     uploads guest/, profiles/, prompts            ├─ desktop broker       (GUI, OS-Helix, MM-Coding)
                                                   ├─ ask_vision           (Vision-Tool)
                                                   └─ run_code + shim      (OS-Helix)
```

The agent loop is the [DeepSeek Harness SDK](https://pypi.org/project/deepseek-harness-sdk/),
driven by the official OSWorld-V2 integration on every benchmark. A harness is a set of DSH patches
(`mmcoding/profiles/`), a capability paragraph (`mmcoding/prompts/capability/`) and the guest tools
it mounts (`mmcoding/guest/`).

```
mmcoding/
  harnesses.py        the five harnesses: tools, desktop broker profile, image budget
  prompts/            capability paragraphs and the shared prompt pieces
  profiles/           DSH patches: common + one per harness + ask_user (OSWorld 2.0)
  guest/              runs inside the VM, standard library only
    rv_run.py           sandbox, network namespace, D-Bus filter, worker unit (as /usr/local/sbin/rv-run)
    worker_rv.py        DSH worker: composes the patches, enforces the call budget
    rv_desktop.py       desktop broker: screenshot and mouse/keyboard tools
    vision_mcp.py       ask_vision (Vision-Tool)
    helix_code_mcp.py   run_code (OS-Helix)
    helix_shim.py       OS-Helix context policy: ten screenshots, one action per step
    report_mcp.py       report_infeasible (every harness)
    ask_user_mcp.py     simulated user (OSWorld 2.0)
  host/
    deploy.py           host side: settings, prompt, deployment, Docker provider
    audit.py            checks that no run used a channel its harness does not have
    sites.py            local OSWorld 2.0 websites
    guard.py            optional network guard (host side)
    guard_addon.py      optional network guard (mitmproxy addon)
  benchmarks/         one runner per benchmark: osworld2, weavebench, ale
configs/              run protocol, models, upstream pins, task lists, sandbox packages
scripts/              setup, environment check, launchers, judge proxy
docker/               VM images and the WeaveBench judge template
tests/                offline tests
```

## Requirements

* Linux host with KVM, Docker, `uv`, Python 3.11+, and root access for the VM networking.
  OSWorld 2.0 and WeaveBench boot QEMU inside Docker; ALE uses its own QEMU provider. Each VM gets
  8 GB of RAM and 4 cores.
* An OpenAI-compatible endpoint that serves the policy model, and for Vision-Tool the vision
  model (the paper uses Qwen3.8-27B with thinking disabled).
* For evaluation: an evaluator and user-simulator model for OSWorld 2.0, a judge model for
  WeaveBench, and ALE's judge configuration for the ALE tasks graded by a model.

## Setup

```bash
git clone https://github.com/OpenEnvision/MM-Coding.git && cd MM-Coding
scripts/setup.sh          # upstream checkouts at pinned commits, two venvs, sandbox assets
# VM images and benchmark data: docker/IMAGES.md
mkdir -p ~/.config/mm-coding && cat > ~/.config/mm-coding/credentials.env <<'EOF'
RV_LLM_BASE_URL=https://your-endpoint/v1
RV_LLM_API_KEY=...
EOF
cp configs/models.example.yaml configs/models.yaml   # model ids of your endpoint
scripts/check_env.sh      # checkouts, images, venvs, host network, credentials
scripts/test.sh           # offline tests
```

`configs/models.yaml` lists the models of the endpoint with their context window, output limit,
reasoning levels and compatibility switches. The example holds the four policy models of the
paper. `configs/protocol.json` holds the run protocol of each benchmark: 500 model calls, maximum
reasoning effort, 64K output tokens per call, and the vision model settings.

## Running

```bash
scripts/run_osworld2.sh   --harness mm_coding --model kimi-k3 --num_envs 8 --result_dir runs/osworld2/kimi-k3/mm_coding
scripts/run_weavebench.sh --harness gui       --model kimi-k3 --num_envs 6 --result_dir runs/weavebench/kimi-k3/gui
scripts/run_ale.sh        --harness vision_tool --model kimi-k3 --num_envs 4 --result_dir runs/ale/kimi-k3/vision_tool
```

`--harness` accepts the ids above or the paper names (`MM-Coding`). `--tasks` takes `all` (the
complete task set, default), `@file.json` or comma-separated task ids. `--vision_model` and the
other `--vision_*` flags configure Vision-Tool; `RV_VISION_BASE_URL` and `RV_VISION_API_KEY` put
the vision model on its own endpoint. `--network-guard` turns on the network guard used in the
paper (see below). Benchmark-specific setup, evaluator and judge configuration:
[docs/benchmarks.md](docs/benchmarks.md).

Each task writes one directory under `--result_dir` with the official score, the full agent
transcript (`dsh/`), the budget record and the sandbox evidence. After OSWorld 2.0 runs,
`mmcoding.host.audit` also writes `rv_audit.json` per task, which checks that the run used no
channel its harness does not have.

## Network guard

With `--network-guard`, the outbound traffic of each task VM passes through a transparent
mitmproxy on the host while the agent runs. It blocks public sources of benchmark tasks, reference
answers and evaluators, including mirrors, page-fetch services and archives, and logs every
request. Setup: copy `configs/guard.example.json` to `configs/guard.json`, then start
`scripts/guard_proxy.sh` (as a systemd unit). See `mmcoding/host/guard_addon.py` for the rules.

## Citation

```bibtex
@article{xiao2026rethinking,
  title   = {Rethinking Vision in Computer-Use Agents: Decoupling Perception from Action},
  author  = {Xiao, Chuan and Jiao, Zhengbo and Wang, Baojie and Tian, Juanxi and Fan, Haizhao and
             Hua, De and Xu, Xander and Guo, Zhijiang},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Benchmarks and the agent runtime keep their own licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md). OS-Helix is our implementation of the harness of Yang et al.
(2026), not the authors' code.
