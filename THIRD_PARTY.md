# Third-party software and references

This repository contains only our own code, with one exception listed below. Benchmarks and the
agent runtime are fetched at the pinned versions in `configs/upstream.json` and `pyproject.toml`, and
are used under their own licenses.

| Component | Use | License |
|---|---|---|
| [OSWorld-V2](https://github.com/xlang-ai/OSWorld-V2) (`osworld-v2.1`) | OSWorld 2.0 tasks, evaluators and Docker provider; its DeepSeek Harness integration (`hybrid_agents/deepseek`) runs every harness on every benchmark | Apache-2.0 |
| [WeaveBench](https://github.com/weavebench/WeaveBench) | WeaveBench tasks, VM image and agent-as-judge. `docker/weavebench_judge/` is the judge template published with WeaveBench | MIT |
| [Agents' Last Exam](https://github.com/rdi-berkeley/agents-last-exam) (v1.1) | ALE tasks, QEMU provider and graders | Apache-2.0 (code), CC BY 4.0 (data) |
| [DeepSeek Harness SDK](https://pypi.org/project/deepseek-harness-sdk/) `0.1.5rc1` | agent loop inside the VM, composed per harness by `mmcoding/profiles/` | MIT |
| [mitmproxy](https://mitmproxy.org/) | optional network guard (`mmcoding/host/guard_addon.py`) | MIT |

**OS-Helix.** The OS-Helix harness is our implementation of the hybrid loop described in
B. Yang et al., *OS-Helix: Unifying GUI Interaction and Code Execution in Native Computer-Using
Agents*, arXiv:2610.22234, 2026. It follows the paper's action space and context policy and is not
the authors' code.
