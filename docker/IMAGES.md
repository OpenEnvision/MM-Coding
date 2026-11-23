# VM images, container images and benchmark data

Everything a run boots or reads besides this repository and the upstream checkouts. Large files
live under `$RV_DATA` (default `data/` in the repository; see `mmcoding/paths.py`). All VM disks are
read-only bases: every task boots a copy-on-write overlay, so the files never change.
`scripts/check_env.sh` checks that everything below is present; `scripts/check_env.sh --hash`
re-checks a disk against a `<file>.sha256` next to it.

## VM disks (qcow2)

| Benchmark | Path under `$RV_DATA/images` | Size | sha256 (qcow2) | Source |
|---|---|---|---|---|
| OSWorld 2.0 | `osworld_v2/osworld-v2-ubuntu-x86.qcow2` | 14.9 GB | `28b617987f3edf14edd835069cdde6a6e4708be8e4ae4eb957200d5da335186a` | Hugging Face dataset `xlangai/v2-image` @ `6e16459a2feb5a8f1ed65babcfe7a2a6205d049d`, file `osworld-v2-ubuntu-x86.qcow2.zip` (the zip's sha256 is pinned in OSWorld-V2 `benchmark_releases/osworld-v2.1.json`); unzip it |
| WeaveBench (published) | `weavebench/Ubuntu.qcow2` | 28.5 GB | `3c1caf41cb75b8482a30d1a251545393aee8175375ab405e0d6870f8a07fa3f8` | WeaveBench `weavebench-download-vm --dest <dir>` (lands in `<dir>/vm/Ubuntu.qcow2`) |
| WeaveBench (used) | `weavebench/Ubuntu_rv.qcow2` | 28.4 GB | depends on your SSH key | the published image + `openssh-server` + your SSH key for `user`, nothing else: `docker/bake_weavebench_image.py` (writes `Ubuntu_rv.qcow2.sha256`) |
| ALE v1.1 | `ale/qemu/images/ale-ubuntu22-v1.1.qcow2` | 92.6 GB | manifest sha256 `6557e22ca4dc5a3faba79b36e8c57d859951d17ace9e218f2355db7b77a42bc0` (ALE's own record, `*.ale-source.json`) | `hf://agents-last-exam/ale-images-qcow2/ale-ubuntu22-v1.1.qcow2` @ `40ba46b348a684b7d1b34d18e308aed54737f41a`, downloaded and checked by ALE's QEMU provider (`docker/fetch_ale_disk.py`, or on first use) |

Check a download with `sha256sum <file>` against the table (the WeaveBench published image and the
OSWorld 2.0 disk); ALE's provider checks its disk itself. Guest logins are the public benchmark
defaults: OSWorld 2.0 `user` / `osworld-public-evaluation`; WeaveBench and ALE `user` / `password`
(ALE: passwordless sudo). `configs/protocol.json` holds them per benchmark.

## Container images (Docker)

| Used by | Image |
|---|---|
| OSWorld 2.0, WeaveBench (QEMU-in-Docker runner of the OSWorld Docker provider) | `happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9` (pinned by digest in OSWorld-V2's Docker provider) |
| ALE (QEMU provider) | `agentslastexam/ale-qemu:0.2.0` (digest `sha256:cc3ead471566c27f7c394c8748ff2e69124f310dbe35fce5e4a5177bc947232f`) |

```bash
docker pull happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9
docker pull agentslastexam/ale-qemu:0.2.0
docker image inspect --format '{{index .RepoDigests 0}}' agentslastexam/ale-qemu:0.2.0   # check the digest
```

The runners pass the qcow2 path explicitly (OSWorld 2.0: `--path_to_vm`, default from
`mmcoding/paths.py`), so the Docker provider never downloads into a working directory. Each VM
container of OSWorld 2.0 and WeaveBench gets 8 GB RAM and 4 cores for every harness
(`configs/protocol.json`); ALE VMs use the machine type of each task card. The osworld-docker image
declares an anonymous volume; `scripts/docker_gc.sh` removes the ones finished runs leave behind.

## Benchmark data

| Data | Location | How to get it |
|---|---|---|
| OSWorld 2.0 task classes | `$RV_OSWORLD/evaluation_examples/task_class/task_*.py` | gated upstream: `scripts/tools/download_osworld_v2_tasks.py --benchmark-release osworld-v2.1` in the OSWorld-V2 checkout |
| OSWorld 2.0 task assets | `$RV_DATA/assets/osworld_v2/osworld_v2_assets_v2.1` (`$OSWORLD_FILE_BASE_URL`) | `scripts/tools/download_osworld_v2_assets.py --benchmark-release osworld-v2.1 --target-dir <path>` in the OSWorld-V2 checkout |
| WeaveBench tasks | `$RV_DATA/assets/weavebench/tasks` (`$WB_TASKS_ROOT`) | `weavebench-download-dataset --dest <dir>` (tasks land in `<dir>/tasks`) |
| Sandbox assets (installed in every guest) | `$RV_DATA/assets/sandbox/{wheelhouse,debs}` (`$RV_ASSET_DIR`) | `python3 scripts/fetch_sandbox_assets.py` (38 wheels for CPython 3.10 and the `auditd` / `libauparse0` packages of Ubuntu 22.04, checked against `configs/sandbox/assets.sha256`) |

Every run manifest records the sha256 of each sandbox asset.
