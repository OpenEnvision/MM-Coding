"""Run OSWorld 2.0 (OSWorld-V2 release osworld-v2.1, 108 tasks) with one harness on the Docker provider.

The per-task pipeline is the pinned checkout's ``scripts/python/run_multienv_deepseek.py``, used
unchanged: ``run_case`` (setup transport, DSH execution, evidence collection, official evaluation
including the timeout-scoring path, outcome files), the run manifest and the report. The guest and
agent bindings come from ``mmcoding.host.deploy.install``; after the batch the forbidden-channel
audit (``mmcoding.host.audit``) writes ``rv_audit.json`` per task.

usage: scripts/run_osworld2.sh --harness mm_coding --model MODEL [--tasks all] --num_envs N --result_dir DIR
"""

from __future__ import annotations

import os

from mmcoding import paths
from mmcoding.benchmarks import common

BENCH = "osworld2"
# The guest login of the benchmark image; mmcoding.host.deploy reads it at import.
os.environ["OSWORLD_CLIENT_PASSWORD"] = common.protocol(BENCH)["client_password"]
paths.use_upstream("osworld")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from hybrid_agents.deepseek.common import SecretFilter, write_json  # noqa: E402
from scripts.python import run_multienv_deepseek as official  # noqa: E402

from mmcoding.benchmarks import evaluation_transport  # noqa: E402
from mmcoding.host import audit, deploy, sites  # noqa: E402

REPO = official.REPO  # the pinned OSWorld-V2 checkout
# Error kinds after which no further task is started (credentials, quota, host disk, drift).
ACCOUNT_FAILURES = {"auth", "quota", "host_disk", "configuration_drift", "provision"}
SETUP_FAILURE_LIMIT = 3  # consecutive-kind provision/setup failures that pause dispatch


def parse_tasks(tasks: list[str]) -> list[str]:
    """Zero-padded ids, all from the complete task list (``1`` -> ``001``)."""
    known = set(common.load_tasks("all", BENCH))
    try:
        ids = [f"{int(t):03d}" for t in tasks]
    except ValueError as exc:
        raise SystemExit(f"invalid OSWorld 2.0 task id: {exc}") from exc
    unknown = sorted(set(ids) - known)
    if unknown or len(ids) != len(set(ids)):
        raise SystemExit(f"invalid task list (unknown or duplicate ids): {unknown or ids}")
    return ids


def docker_image_identity(release: str) -> dict:
    data = json.loads((REPO / "benchmark_releases" / f"{release}.json").read_text())
    image = data["provider_images"]["docker"]["ubuntu"]
    return {key: image.get(key) for key in ("artifact_path", "artifact_sha256", "runtime_image", "artifact_tag")}


def configure_evaluator(proto: dict) -> dict:
    """Evaluator and user-simulator routes (upstream ``OSWORLD_EVAL_MODEL_*`` / ``OSWORLD_USER_SIM_*``).

    Both models are user-supplied; a missing model is an error before any VM starts. Every
    harness of a comparison must run with the same two routes.
    """
    os.environ.setdefault("OSWORLD_EVAL_MODEL_PROVIDER", proto["evaluator_provider"])
    os.environ.setdefault("OSWORLD_USER_SIM_PROVIDER", os.environ["OSWORLD_EVAL_MODEL_PROVIDER"])
    os.environ["OSWORLD_EVAL_MODEL_NAME"] = common.user_setting(BENCH, "evaluator_model", "OSWORLD_EVAL_MODEL_NAME")
    os.environ["OSWORLD_USER_SIM_MODEL"] = common.user_setting(BENCH, "user_simulator_model", "OSWORLD_USER_SIM_MODEL")
    return {k: os.getenv(k) for k in ("OSWORLD_EVAL_MODEL_PROVIDER", "OSWORLD_EVAL_MODEL_NAME",
                                      "OSWORLD_USER_SIM_PROVIDER", "OSWORLD_USER_SIM_MODEL")}


def config(argv=None):
    """Arguments of one run, validated before any VM starts."""
    proto = common.protocol(BENCH)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_agent_args(parser, BENCH)
    parser.add_argument("--timeout", type=int, default=proto["timeout_s"],
                        help="wall-clock safety cap per task (s); the primary budget is --call_budget")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--setup_parallelism", type=int, default=4)
    parser.add_argument("--evaluation_parallelism", type=int, default=4)
    parser.add_argument("--collection_parallelism", type=int, default=3)
    parser.add_argument("--minimum_free_gib", type=float, default=20)
    parser.add_argument("--volume_size", type=int, default=64, help="minimum guest disk (GB)")
    parser.add_argument("--vm_ram", default=proto["vm_ram"], help="guest RAM (Docker provider RAM_SIZE)")
    parser.add_argument("--vm_cores", default=proto["vm_cores"], help="guest CPU cores (Docker provider CPU_CORES)")
    parser.add_argument("--ssh_user", default="user")
    parser.add_argument("--path_to_vm", default=str(paths.OSWORLD_V2_IMAGE), help="guest qcow2 (read-only base)")
    parser.add_argument("--resume", action="store_true",
                        help="continue in an existing --result_dir: run only tasks without a task directory")
    parser.add_argument(
        "--no_timeout_scoring", action="store_true",
        help="disable the official recovery path that scores a timed-out task (only when the task "
             "declares intermediate_eval_safe) from its quiescent final state",
    )
    args = parser.parse_args(argv)
    common.finish_agent_args(parser, args, BENCH)
    if not 1 <= args.num_envs <= 64:
        parser.error("--num_envs must be in 1..64")
    try:
        args.tasks = parse_tasks(args.tasks)
        args.evaluator = configure_evaluator(proto)
    except SystemExit as exc:
        parser.error(str(exc))
    if not Path(args.ssh_private_key).is_file() or not Path(args.ssh_private_key + ".pub").is_file():
        parser.error(f"SSH key pair for the guest is missing: {args.ssh_private_key}[.pub]")
    if not Path(args.path_to_vm).is_file():
        parser.error(f"VM image missing: {args.path_to_vm} (see docker/IMAGES.md)")
    args.settings = common.settings(args, args.timeout)
    # Fields read by the official run_case / manifest.
    args.region = "docker"
    args.snapshot_name = docker_image_identity(proto["benchmark_release"])["artifact_sha256"]
    args.recovery_mode = not args.no_timeout_scoring
    return args


def harness_manifest(args) -> tuple[dict, dict]:
    # Absolute paths: the official drift guard reads ``REPO / path``, which is then the path itself.
    files = [*deploy.rv_files(), Path(__file__).resolve(), Path(common.__file__).resolve(),
             Path(evaluation_transport.__file__).resolve(), paths.CONFIGS / "protocol.json"]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    info = {
        "harness": args.harness,
        "settings": args.settings.public(),
        "file_hashes": hashes,
        "rv_content_hash": deploy.rv_content_hash(),
        "docker_image": docker_image_identity(common.protocol(BENCH)["benchmark_release"]),
        "volume_size_gb": args.volume_size,
        "vm_resources": {"ram": args.vm_ram, "cores": args.vm_cores},
        "network_guard": args.network_guard,
        "evaluator": args.evaluator,
        "evaluation_transport": evaluation_transport.policy(),
        "task_hidden_paths": deploy.TASK_HIDDEN_PATHS,
        "assets": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in deploy.asset_files()},
        "vm_image": args.path_to_vm,
        "repo_commit": _repo_commit(),
        "note": "Docker provider instead of the official cloud provider; not an official leaderboard submission.",
    }
    return info, hashes


def _repo_commit() -> dict:
    def git(*cmd):
        try:
            return subprocess.run(["git", *cmd], cwd=paths.REPO, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--short"))}


def reap_retained(args, task: str, result: dict) -> dict | None:
    """Remove the container of a VM that the official runner kept for evidence recovery.

    ``run_case`` keeps the VM when quiescence or evidence recovery is incomplete. The collected
    evidence is already on the host, so the container is removed and the batch continues;
    ``retained_vms.json`` records each case.
    """
    if not str(result.get("cleanup_error", "")).startswith("VM retained"):
        return None
    record = {"task": task, "time": time.time()}
    try:
        info = json.loads((args.result_dir / task / "container.json").read_text())
        record["container"] = info.get("container")
        if record["container"]:
            proc = subprocess.run(["docker", "rm", "-f", record["container"]], capture_output=True, text=True,
                                  timeout=120)
            record["removed"] = proc.returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        record["error"] = str(exc)[:200]
    retained = args.result_dir / "retained_vms.json"
    rows = json.loads(retained.read_text()) if retained.exists() else []
    write_json(retained, rows + [record])
    return record


def dispatch(args) -> list[dict]:
    """Run every task; only account / host failures and repeated setup failures pause dispatch.

    Each case gets its own pause flag for the official runner, so one case it flags (for example
    a retained VM) does not stop the remaining tasks.
    """
    outcomes = [json.loads(p.read_text()) for p in sorted(args.result_dir.glob("*/outcome.json"))]
    with mp.Manager() as manager:
        pause = manager.Event()
        args.setup_slot = manager.BoundedSemaphore(args.setup_parallelism)
        args.evaluation_slot = manager.BoundedSemaphore(args.evaluation_parallelism)
        args.collection_slot = manager.BoundedSemaphore(args.collection_parallelism)
        with mp.Pool(args.num_envs) as pool:
            remaining, pending, setup_failures = list(args.tasks), {}, {}
            case_flags = {}  # keeps each case's manager proxy alive until the case returns
            while remaining or pending:
                while remaining and len(pending) < args.num_envs and not pause.is_set():
                    if shutil.disk_usage(args.result_dir).free < args.minimum_free_gib * 2**30:
                        pause.set()
                        write_json(args.result_dir / "dispatch_paused.json", {"reason": "host_disk"})
                        break
                    task = remaining.pop(0)
                    case_flags[task] = manager.Event()
                    pending[task] = pool.apply_async(official.run_case, (args, task, case_flags[task]))
                for task, future in list(pending.items()):
                    if not future.ready():
                        continue
                    result = future.get()
                    outcomes.append(result)
                    del pending[task]
                    case_flags.pop(task, None)
                    reap_retained(args, task, result)
                    kind = result.get("error_kind")
                    if result["status"] == "error" and kind in ACCOUNT_FAILURES:
                        pause.set()
                        write_json(args.result_dir / "dispatch_paused.json", {"reason": kind, "task": task})
                    if result.get("phase") in {"provision", "setup"} and result["status"] == "error":
                        kind = kind or "setup"
                        setup_failures[kind] = setup_failures.get(kind, 0) + 1
                        if setup_failures[kind] >= SETUP_FAILURE_LIMIT:
                            pause.set()
                            write_json(args.result_dir / "dispatch_paused.json",
                                       {"reason": kind, "failures": setup_failures[kind]})
                write_json(args.result_dir / "summary.json", outcomes)
                official.write_report(args.result_dir)
                if pause.is_set() and not pending:
                    break
                time.sleep(5)
    return outcomes


def prepare_result_dir(args) -> bool:
    """Create the result directory (or reuse it with ``--resume``); True when it already existed."""
    existed = args.result_dir.exists()
    if existed and not args.resume:
        raise SystemExit(f"{args.result_dir} exists; pass --resume to continue it")
    args.result_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (args.result_dir / "cache").mkdir(exist_ok=True)
    args.result_dir = args.result_dir.resolve()
    if existed:
        args.tasks = [t for t in args.tasks if not (args.result_dir / t).exists()]
    return existed


def main(argv=None):
    web = sites.load_config()
    if web is not None:
        sites.install_process_resolution(web)
    args = config(argv)
    os.umask(0o077)
    os.environ["OSWORLD_STRICT_SETUP_TRANSPORT"] = "1"
    os.environ["OSWORLD_PRESERVE_VM_CLOCK"] = "true"
    os.environ["ENABLE_TTL"] = "false"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sensitive = tuple(
        v for k, v in os.environ.items() if any(w in k for w in ("API_KEY", "TOKEN", "PASSWORD", "SECRET"))
    ) + (args.api_key,)
    for handler in logging.getLogger().handlers:
        handler.addFilter(SecretFilter(sensitive))

    deploy.install(official, args.harness, vm_ram=args.vm_ram, vm_cores=args.vm_cores, base_url=args.base_url,
                   network_guard=args.network_guard)
    evaluation_transport.install()
    resumed = prepare_result_dir(args)
    info = official.manifest(args)
    harness_info, harness_hashes = harness_manifest(args)
    info["rv_harness"] = harness_info
    info["benchmark_status"] = harness_info["note"]
    args.expected_hashes = {**info["file_hashes"], **harness_hashes}  # drift guard in run_case
    manifest_path = args.result_dir / "manifest.json"
    if resumed and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous.setdefault("resumed", []).append({"started": time.time(), "tasks": args.tasks,
                                                   "settings": harness_info["settings"]})
        write_json(manifest_path, previous)
    else:
        write_json(manifest_path, info)
    try:
        dispatch(args)
    finally:
        report = official.write_report(args.result_dir)
        try:
            audit.analyze_root(args.result_dir)
        except Exception as exc:  # noqa: BLE001 - the audit never hides the run result
            logging.exception("rv audit failed: %s", exc)
    print(json.dumps({"result_dir": str(args.result_dir), "valid_results": report["valid_results"],
                      "errors": report["errors"], "mean_score": report["valid_result_mean_score"]}, indent=2))


if __name__ == "__main__":
    main()
