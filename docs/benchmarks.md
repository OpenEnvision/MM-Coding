# Benchmarks

Every runner uses the complete task set of its benchmark (`configs/tasks/<bench>.json`), the
benchmark's own VM image and initial state, and its official evaluator. The harness is the only
thing that changes between runs. Shared runner flags: `--harness`, `--model`, `--tasks`,
`--result_dir`, `--num_envs`, `--call_budget`, `--reasoning_effort`, `--max_tokens`,
`--vision_model`, `--vision_reasoning_effort`, `--vision_coords`, `--vision_thinking`,
`--network-guard`. Defaults come from `configs/protocol.json`.

## OSWorld 2.0

108 long-horizon workflows across desktop applications (OSWorld-V2 release `osworld-v2.1`), run
on the OSWorld-V2 Docker provider with the official `run_case` pipeline.

* **Tasks.** The task classes are gated upstream. Request access and download them into the
  OSWorld-V2 checkout with its `scripts/tools/download_osworld_v2_tasks.py`. Task assets go to
  `$RV_DATA/assets/osworld_v2/osworld_v2_assets_v2.1` (or set `OSWORLD_FILE_BASE_URL`).
* **Evaluator and simulated user.** Set `OSWORLD_EVAL_MODEL_NAME`, `OSWORLD_EVAL_MODEL_BASE_URL`
  and the key in `OSWORLD_EVAL_API_KEY` (or fill `evaluator_model` and `user_simulator_model` in
  `configs/protocol.json`). The user simulator uses the same route unless `OSWORLD_USER_SIM_*` is
  set. All variables are listed at the top of `scripts/run_osworld2.sh`.
* **Websites.** Tasks on the benchmark's websites need the upstream OSWorld-web deployment on the
  host; `mmcoding/host/sites.py` connects each task VM to it (`RV_WEBSITE_CONFIG`).
* **Evaluator-only files.** A task's setup may place files in the VM that only its evaluator should
  read. List them per task in `$RV_DATA/private/osworld_v2_task_private.json`
  (`{"hidden_paths": {"<task>": ["/path", ...]}}`, or `RV_TASK_PRIVATE`) to hide them from the
  agent sandbox.
* `--resume` runs only tasks without a result directory. After the run, `mmcoding.host.audit`
  writes `rv_audit.json` for every task.

```bash
scripts/run_osworld2.sh --harness mm_coding --model kimi-k3 --num_envs 8 --result_dir runs/osworld2/kimi-k3/mm_coding
```

## WeaveBench

114 tasks in eight categories, each requiring both the GUI and the command line, graded by
WeaveBench's trajectory-aware OpenClaw agent-as-judge.

* **Image.** Download the published VM and add `openssh-server` and your SSH key with
  `docker/bake_weavebench_image.py` (docker/IMAGES.md). Task data goes to
  `$RV_DATA/assets/weavebench/tasks` (or `WB_TASKS_ROOT`).
* **Judge.** Set `WEAVEBENCH_JUDGE_MODEL`, `WEAVEBENCH_JUDGE_BASE_URL` and
  `WEAVEBENCH_JUDGE_API_KEY`, then run `scripts/render_judge_profile.py` and keep
  `scripts/judge_proxy.sh` running (a local retry proxy in front of the judge endpoint).
* `--no_judge` skips grading, `--rejudge` grades finished runs again with the current judge.
  Scores are stored per judge model under `<task>/judges/<model>/score.json`.

```bash
scripts/run_weavebench.sh --harness mm_coding --model kimi-k3 --num_envs 6 --result_dir runs/weavebench/kimi-k3/mm_coding
```

## Agents' Last Exam

The 85 ALE v1.1 tasks that run on the Linux CPU desktop, graded by each task's deterministic
`evaluate()` on the delivered files, run on ALE's QEMU provider in ALE's own venv.

* **Image.** `docker/fetch_ale_disk.py` downloads and verifies the disk through ALE's provider.
* **Grading model.** Some ALE graders call a model: configure it in the ALE checkout at
  `secret/eval_time/openai.env` as ALE documents.
* `--regrade` grades finished runs again without running the agent.

```bash
scripts/run_ale.sh --harness mm_coding --model kimi-k3 --num_envs 4 --result_dir runs/ale/kimi-k3/mm_coding
```
