"""Run WeaveBench (114 tasks) with one harness.

WeaveBench supplies the VM image (a copy with sshd baked in: docker/bake_weavebench_image.py), the
task setup (workspace upload, task env script, warmup), the deliverable archive and the
trajectory-aware agent-as-judge. The agent side is the same as on the other benchmarks: the rv
sandbox (``mmcoding/guest/rv_run.py``), the DSH SDK worker and the harness's tools; WeaveBench's
task prompt is the task instruction.

Per task: boot VM -> WeaveBench setup -> rv deploy (harness) -> agent (call budget) -> archive
/tmp_workspace/results -> DSH trajectory converted to OpenClaw chat.jsonl -> VM removed ->
judge (OpenClaw with the user's judge model, weavebench_judge.py) -> score.json + outcome.json.

Re-running into an existing --result_dir runs only tasks without outcome.json; --rejudge only
judges finished tasks that have no successful score from the selected judge model. Scores are
kept per judge under <task>/judges/<model>/score.json; <task>/score.json is the latest one.

usage: scripts/run_weavebench.sh --harness mm_coding --model MODEL [--tasks all] --num_envs N --result_dir DIR
"""
from __future__ import annotations

import os

from mmcoding import paths
from mmcoding.benchmarks import common

BENCH = "weavebench"
# The guest login of the benchmark image; mmcoding.host.deploy reads it at import.
os.environ["OSWORLD_CLIENT_PASSWORD"] = common.protocol(BENCH)["client_password"]
paths.use_upstream("osworld", "weavebench")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from urllib.parse import quote  # noqa: E402

from hybrid_agents.deepseek.common import DeepSeekError, SecretFilter, scrub, write_json  # noqa: E402

from mmcoding.benchmarks import weavebench_judge  # noqa: E402
from mmcoding.benchmarks.common import dsh_to_openclaw_chat  # noqa: E402
from mmcoding.host import deploy  # noqa: E402

WB_IMAGE = paths.WEAVEBENCH_IMAGE
TASKS_ROOT = paths.WEAVEBENCH_TASKS
DOMAINS = ["WEB", "DAV", "OPS", "DOC", "DES", "GAM", "SPA", "DSK"]
PASSWORD = os.environ["OSWORLD_CLIENT_PASSWORD"]
JUDGE_ATTEMPTS = 3
log = logging.getLogger("rv_weavebench")


# ----------------------------------------------------------------------- task


def discover(task_ids: list[str]) -> dict[str, dict]:
    import weavebench.eval.orchestrator as ds  # noqa: PLC0415

    tasks = {t["task_id"]: t for t in ds.discover_tasks(TASKS_ROOT, DOMAINS)}
    missing = [t for t in task_ids if t not in tasks]
    if missing:
        raise SystemExit(f"unknown WeaveBench tasks (or missing under {TASKS_ROOT}): {missing}")
    return {t: tasks[t] for t in task_ids}


def new_vm(args, task_id: str, out: Path):
    from desktop_env.desktop_env import DesktopEnv  # noqa: PLC0415

    env = DesktopEnv(action_space="pyautogui", provider_name="docker", path_to_vm=str(WB_IMAGE),
                     screen_size=(1920, 1080), headless=True, os_type="Ubuntu", require_a11y_tree=False,
                     enable_proxy=False, client_password=PASSWORD, volume_size=args.volume_size,
                     force_disable_vnc=True, force_disable_recording=True, cache_dir=str(out / "cache"))
    container = env.provider.container
    container.reload()
    ip = next(iter(container.attrs["NetworkSettings"]["Networks"].values()))["IPAddress"]
    write_json(out / "container.json", {"container": container.name, "ip": ip,
                                        "resources": dict(env.provider.environment)})
    return env, ip


def judge(args, task: dict, out: Path) -> dict:
    from weavebench.eval.agent_judge.judge_runner import judge_one  # noqa: PLC0415
    from weavebench.eval.agent_judge.stage_case import stage_case  # noqa: PLC0415

    # Upstream recreates _eval/<case>, state_<case> and ws_<case> destructively; a fresh id per
    # call isolates simultaneous judges of the same rollout without touching AJ_ISOLATED_ROOT.
    case_id = f"rv_{uuid.uuid4().hex}"
    stage = stage_case(case_dir=str(out), bench_root=str(TASKS_ROOT), bench_subdir=".",
                       judge_workspace=str(args.judge_workspace), case_id=case_id,
                       category=task["category"], task_id=task["task_id"])
    result = {"ok": False}
    for attempt in range(JUDGE_ATTEMPTS):
        result = judge_one(stage, case_id, thinking=os.environ.get("AJ_THINKING", "medium"),
                           timeout=int(os.environ.get("AJ_TIMEOUT", "1800")))
        if result.get("ok"):
            break
        # Quota and rate-limit failures clear only after the provider's window; a short pause would
        # waste the next attempt.
        quota = bool(re.search(r"\b429\b|rate limit|quota", result.get("openclaw_stderr_tail") or "", re.I))
        if attempt + 1 < JUDGE_ATTEMPTS:
            time.sleep((300 if quota else 30) * (attempt + 1))
    return {**result, "case_id": case_id, "stage_dir": str(stage)}


_SSH_RC = "__RV_ARCHIVE_RC="

# Some warmups leave answer files in root-only /opt/<task>_gt, and the warmup script and its log
# stay world-readable in /tmp (the script carries the same answers as heredocs). The judge reads
# the host-side gt/ copy, so these are removed after warmup: any harness with a terminal or code
# can sudo on the desktop. /tmp/_warmup.rc and /tmp/_warmup.done (status only) stay.
_GT_FILES = "/tmp/_warmup.sh /tmp/_warmup.log"
_GT_SCRUB = (
    "shopt -s nullglob; for d in /opt/*_gt; do [ -d \"$d\" ] && rm -rf -- \"$d\" && echo \"$d\"; done; "
    f"for f in {_GT_FILES}; do [ -e \"$f\" ] && rm -f -- \"$f\" && echo \"$f\"; done; "
    "for d in /opt/*_gt; do echo \"LEFT:$d\"; done; "
    f"for f in {_GT_FILES}; do [ -e \"$f\" ] && echo \"LEFT:$f\"; done; true"
)


def scrub_vm_gt(guest) -> list[str]:
    text = guest.command(f"echo {shlex.quote(PASSWORD)} | sudo -S -p '' bash -c {shlex.quote(_GT_SCRUB)}",
                         timeout=60)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    left = [line for line in lines if line.startswith("LEFT:")]
    if left:
        raise RuntimeError(f"ground-truth files survived the scrub: {left}")
    return lines


def install_ssh_transport(ds) -> None:
    """Let the orchestrator's exec/fetch go over SSH for envs carrying ``ssh_guest``.

    The in-VM REST server (port 5000) runs as the desktop user, so an agent that kills stray
    Python processes can take it down; the deliverables are still on disk and sshd is independent
    of it. Other envs keep the REST path.
    """
    if getattr(ds, "_rv_ssh_transport", False):
        return
    rest_exec, rest_fetch = ds._vm_exec, ds._vm_fetch

    def vm_exec(env, cmd, shell=False, timeout=120, *rest, **kw):
        guest = getattr(env, "ssh_guest", None)
        if guest is None:
            return rest_exec(env, cmd, shell, timeout, *rest, **kw)
        line = cmd if isinstance(cmd, str) else shlex.join(cmd)
        text = guest.command(f"{line}; printf '\\n{_SSH_RC}%d' $?", timeout=timeout, check=False)
        body, _, code = text.rpartition(_SSH_RC)
        return {"output": body.rstrip("\n"), "returncode": int(code or 1), "error": ""}

    def vm_fetch(env, remote, local):
        guest = getattr(env, "ssh_guest", None)
        if guest is None:
            return rest_fetch(env, remote, local)
        data = guest.read(remote)
        if data is None:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        return True

    ds._vm_exec, ds._vm_fetch, ds._rv_ssh_transport = vm_exec, vm_fetch, True


def archive(ds, wb, guest, out: Path, record: dict) -> None:
    """WeaveBench's deliverable archive over REST, over SSH when the REST server is gone."""
    import requests  # noqa: PLC0415

    try:
        ds._archive_deliverables(wb, out, client_password=PASSWORD)
        return
    except (requests.ConnectionError, requests.Timeout) as exc:
        log.warning("REST archive failed (%s); retrying over SSH", exc)
        record["archive_fallback"] = f"ssh after {type(exc).__name__}"
    install_ssh_transport(ds)
    ds._archive_deliverables(SimpleNamespace(**vars(wb), ssh_guest=guest), out, client_password=PASSWORD)


def run_task(args, task_id: str) -> dict:
    import weavebench.eval.orchestrator as ds  # noqa: PLC0415

    task = args.tasks_info[task_id]
    out = args.result_dir / task_id
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log")
    handler.addFilter(SecretFilter((args.api_key,)))
    logging.getLogger().addHandler(handler)
    record = {"task_id": task_id, "category": task["category"], "harness": args.harness, "model": args.model,
              "started": time.time(), "status": "error", "phase": "provision"}
    env = None
    try:
        with args.setup_slot:
            env, ip = new_vm(args, task_id, out)
            wb = SimpleNamespace(vm_ip=ip, server_port=5000, controller=env.controller)
            record["phase"] = "setup"
            ds.upload_workspace(wb, task["workspace_path"], client_password=PASSWORD)
            ds.upload_skills(wb, task.get("skills", ""), task["skills_path"])
            ds.write_task_env_script(wb, task.get("env", ""))
            ds.run_warmup(wb, task.get("warmup", ""), client_password=PASSWORD)
            try:
                ds.take_init_screenshot(wb, out)
            except Exception as exc:  # noqa: BLE001 - evidence only
                log.warning("init screenshot failed: %s", exc)
            record["phase"] = "deploy"
            deploy.TASK_BY_HOST[ip] = task_id
            guest = deploy.RvGuest(ip, args.ssh_private_key, "user")
            guest.connect()
            if args.scrub_gt:
                record["gt_scrubbed"] = scrub_vm_gt(guest)
            guest.deploy()
        record["phase"] = "agent"
        agent = deploy.RvAgent(guest, args.api_key, common.settings(args, args.timeout))
        agent.env = SimpleNamespace(user_simulator=None)
        agent.result_dir = str(out)
        try:
            final, _ = agent.predict(task["prompt"])
            record["agent_status"] = "completed"
            record["final_response"] = (final or "")[:2000]
        except DeepSeekError as exc:  # budget / wall clock / runtime: the end state is still archived and judged
            record["agent_status"] = exc.error_kind
            record["agent_error"] = str(exc)[:300]
        record["phase"] = "archive"
        archive(ds, wb, guest, out, record)
        record["chat_lines"] = dsh_to_openclaw_chat(out / "dsh", out / "chat.jsonl", args.model, args.provider)
        record["status"] = "completed"
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        record["traceback"] = traceback.format_exc()[-3000:]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                record["cleanup_error"] = str(exc)[:200]
                info = json.loads((out / "container.json").read_text()) if (out / "container.json").exists() else {}
                if info.get("container"):
                    subprocess.run(["docker", "rm", "-f", info["container"]], capture_output=True, timeout=120)
    if record["status"] == "completed" and not args.no_judge:
        record["phase"] = "judge"
        with args.judge_slot:
            score_case(args, task, out, record)
    record["finished"] = time.time()
    record["minutes"] = round((record["finished"] - record["started"]) / 60, 1)
    if record["status"] == "completed":
        record.pop("traceback", None)
    write_judge_json(out / "outcome.json", record)
    logging.getLogger().removeHandler(handler)
    return record


# ---------------------------------------------------------------------- judge


def judge_score_path(out: Path, judge_model: str) -> Path:
    return out / "judges" / quote(judge_model, safe="") / "score.json"


def load_judge_score(out: Path, judge_model: str) -> dict | None:
    """The score of one judge model: its own file, else ``score.json`` if that one is from this judge."""
    for path in (judge_score_path(out, judge_model), out / "score.json"):
        if path.exists():
            score = json.loads(path.read_text())
            if score.get("judge_model") == judge_model:
                return score
    return None


def write_judge_json(path: Path, value) -> None:
    """Atomic write (concurrent judge processes may publish to the same task)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        write_json(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def judge_summary(result_dir: Path, judge_model: str) -> list[dict]:
    """Outcomes with the scores of one judge model; scores of other judges are never mixed in."""
    summary = []
    for path in sorted(result_dir.glob("*/outcome.json")):
        record = json.loads(path.read_text())
        score = load_judge_score(path.parent, judge_model) or {}
        record.update(judge_model=judge_model, judge_ok=bool(score.get("judge_ok")),
                      score=score.get("score") if score.get("judge_ok") else None)
        summary.append(record)
    return summary


def score_case(args, task: dict, out: Path, record: dict) -> dict:
    try:
        aj = judge(args, task, out)
    except Exception as exc:  # noqa: BLE001
        aj = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    score = {"task_id": task["task_id"], "category": task["category"], "harness": args.harness, "model": args.model,
             "judge_model": args.judge_model, "score": aj.get("final_score") if aj.get("ok") else None,
             "is_hack": aj.get("is_hack"), "hack_confidence": aj.get("hack_confidence"),
             "hack_patterns": aj.get("hack_patterns"), "judge_ok": bool(aj.get("ok")),
             "judge_error": aj.get("error"), "scores": aj.get("score_json"), "judge_elapsed_s": aj.get("elapsed_s"),
             "judge_case_id": aj.get("case_id"), "judge_stage_dir": aj.get("stage_dir")}
    if not aj.get("ok"):
        keys = tuple(k for k in (args.api_key, getattr(args, "judge_key", None)) if k)
        score["judge_stderr_tail"] = scrub(aj.get("openclaw_stderr_tail") or "", keys)[-3000:]
    write_judge_json(judge_score_path(out, args.judge_model), score)
    write_judge_json(out / "score.json", score)
    record["score"] = score["score"]
    record["judge_ok"] = score["judge_ok"]
    record["judge_model"] = args.judge_model
    return score


def rejudge(args, task_id: str) -> dict:
    """Judge a finished task with the selected judge model; the agent run is kept as is."""
    out = args.result_dir / task_id
    record = json.loads((out / "outcome.json").read_text())
    with args.judge_slot:
        score_case(args, args.tasks_info[task_id], out, record)
    record.setdefault("rejudged", []).append(time.time())
    write_judge_json(out / "outcome.json", record)
    return record


def needs_rejudge(out: Path, judge_model: str) -> bool:
    """A finished task without a successful score from ``judge_model``."""
    outcome = out / "outcome.json"
    if not outcome.exists() or json.loads(outcome.read_text()).get("status") != "completed":
        return False
    score = load_judge_score(out, judge_model)
    return not score or not score.get("judge_ok")


# ----------------------------------------------------------------------- main


def config(argv=None):
    proto = common.protocol(BENCH)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_agent_args(p, BENCH)
    p.add_argument("--timeout", type=int, default=proto["timeout_s"],
                   help="agent wall-clock cap per task (s); the primary budget is --call_budget")
    p.add_argument("--num_envs", type=int, default=3)
    p.add_argument("--setup_parallelism", type=int, default=3)
    p.add_argument("--judge_parallelism", type=int, default=3)
    p.add_argument("--volume_size", type=int, default=64)
    p.add_argument("--vm_ram", default=proto["vm_ram"])
    p.add_argument("--vm_cores", default=proto["vm_cores"])
    p.add_argument("--judge_model", default=None,
                   help="judge model (default: $WEAVEBENCH_JUDGE_MODEL, else weavebench.judge_model in protocol.json)")
    p.add_argument("--judge_workspace", type=Path, default=None,
                   help="default: $RV_DATA/judge/weavebench/workspaces/<model>__<run> (run = parent of --result_dir)")
    p.add_argument("--no_judge", action="store_true", help="run the agent only; judge later with --rejudge")
    p.add_argument("--keep_gt", dest="scrub_gt", action="store_false",
                   help="leave the warmup's ground-truth files in the VM (default: remove them after warmup)")
    p.add_argument("--rejudge", action="store_true",
                   help="only judge finished tasks without a successful score from the judge model")
    args = p.parse_args(argv)
    common.finish_agent_args(p, args, BENCH)
    args.judge_thinking = proto["judge_thinking"]
    args.judge_timeout = proto["judge_timeout_s"]
    args.judge_config = None
    if not args.no_judge:
        try:
            args.judge_model, args.judge_config = weavebench_judge.judge(args.judge_model)
        except SystemExit as exc:
            p.error(str(exc))
        args.judge_key = os.getenv(args.judge_config["api_key_env"])
    elif args.rejudge:
        p.error("--rejudge and --no_judge exclude each other")
    return args


def judge_environment(args) -> None:
    """The upstream judge runner's settings (``AJ_*``), unless the caller set them."""
    root = paths.WEAVEBENCH_JUDGE
    os.environ.setdefault("AJ_TEMPLATE_WORKSPACE", str(weavebench_judge.workspace_dir()))
    os.environ.setdefault("AJ_ISOLATED_ROOT", str(root / "iso"))
    os.environ.setdefault("AJ_TIMEOUT", str(args.judge_timeout))
    os.environ.setdefault("AJ_THINKING", str(args.judge_thinking))
    os.environ["AJ_TEMPLATE_PROFILE"] = str(weavebench_judge.profile_dir(args.judge_model))


def main(argv=None) -> None:
    args = config(argv)
    task_ids = args.tasks
    if not args.no_judge:
        judge_environment(args)
        problem = weavebench_judge.problem(args.judge_model, args.judge_config)
        if problem:
            raise SystemExit(problem)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(SecretFilter(tuple(k for k in (args.api_key, getattr(args, "judge_key", None)) if k)))
    deploy.configure_docker_vm(args.vm_ram, args.vm_cores)
    deploy.configure_harness(args.harness, args.base_url)
    deploy.RvGuest.network_guard = args.network_guard
    args.tasks_info = discover(task_ids)
    args.result_dir = args.result_dir.resolve()
    if args.judge_workspace is None:
        run = args.result_dir.parent
        args.judge_workspace = paths.WEAVEBENCH_JUDGE / "workspaces" / f"{run.parent.name}__{run.name}"
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.judge_workspace.mkdir(parents=True, exist_ok=True)
    if args.rejudge:
        todo, rejudge_todo = [], [t for t in task_ids if needs_rejudge(args.result_dir / t, args.judge_model)]
    else:
        todo, rejudge_todo = [t for t in task_ids if not (args.result_dir / t / "outcome.json").exists()], []
    manifest_path = args.result_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    judge_info = None if args.no_judge else {"model": args.judge_model, "thinking": os.environ["AJ_THINKING"],
                                              "timeout_s": int(os.environ["AJ_TIMEOUT"]), **args.judge_config}
    manifest = {
        "benchmark": "WeaveBench", "image": str(WB_IMAGE), "tasks": task_ids, "harness": args.harness,
        "model": args.model, "settings": common.settings(args, args.timeout).public(), "judge": judge_info,
        "vm_resources": {"ram": args.vm_ram, "cores": args.vm_cores}, "scrub_gt": args.scrub_gt,
        "network_guard": args.network_guard, "rv_content_hash": deploy.rv_content_hash(), "started": time.time()}
    if not todo and not rejudge_todo and previous is not None:
        return  # nothing left to do; manifest and summary stay as they are
    if previous is None:
        write_judge_json(manifest_path, manifest)
    else:  # keep the original record and log what this invocation did
        previous.setdefault("resumed", []).append({"started": manifest["started"], "run": todo, "rejudge": rejudge_todo,
                                                   "judge": judge_info, "settings": manifest["settings"]})
        write_judge_json(manifest_path, previous)
    with mp.Manager() as manager:
        args.setup_slot = manager.BoundedSemaphore(args.setup_parallelism)
        args.judge_slot = manager.BoundedSemaphore(args.judge_parallelism)
        with mp.Pool(max(1, args.num_envs if todo else args.judge_parallelism)) as pool:
            results = pool.starmap(run_task, [(args, t) for t in todo])
            results += pool.starmap(rejudge, [(args, t) for t in rejudge_todo])
    summary_line = {"harness": args.harness, "tasks": len(task_ids), "done": len(results)}
    if not args.no_judge:
        # The whole result directory, not only this invocation's tasks.
        summary = judge_summary(args.result_dir, args.judge_model)
        write_judge_json(args.result_dir / "judges" / quote(args.judge_model, safe="") / "summary.json", summary)
        write_judge_json(args.result_dir / "summary.json", summary)
        scored = [r["score"] for r in summary if r.get("score") is not None]
        summary_line.update(judge_model=args.judge_model, scored=len(scored),
                            mean_score=round(sum(scored) / len(scored), 3) if scored else None)
    else:
        write_judge_json(args.result_dir / "summary.json",
                         [json.loads(p.read_text()) for p in sorted(args.result_dir.glob("*/outcome.json"))])
    print(json.dumps(summary_line, indent=1))


if __name__ == "__main__":
    main()
