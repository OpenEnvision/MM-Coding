"""Run Agents' Last Exam (ALE v1.1, Linux CPU tasks) with one harness.

ALE supplies the VM (its QEMU provider and the published, sha256-checked ale-ubuntu22-v1.1 qcow2),
the baked task inputs, each task's ``start``/``evaluate`` (graders run host-side) and the encrypted
reference archive. The agent side is the same as on the other benchmarks: the rv sandbox
(``mmcoding/guest/rv_run.py``), the DSH SDK worker and the harness's tools.

Per task: boot VM -> stage inputs + task start() -> move the task's reference.7z to the host and
delete every baked reference.7z from the guest -> rv deploy (harness) -> agent (call budget) ->
archive output/ -> restore reference.7z -> ALE stage_reference + evaluate() -> VM removed.

The archive password is public (ALE docs) and the desktop user has passwordless sudo, so the
reference must not be on the guest while the agent runs (terminals, X input and code all run there).

Re-running into an existing --result_dir runs only tasks without outcome.json; --regrade only grades
again the finished tasks whose grading failed, from their archived output on a fresh VM.

Runs in the ALE checkout's venv (scripts/setup.sh):
  scripts/run_ale.sh --harness mm_coding --model MODEL [--tasks all] --num_envs N --result_dir DIR
"""
from __future__ import annotations

import os

from mmcoding import paths
from mmcoding.benchmarks import common

BENCH = "ale"
# The guest login of the ALE image (sudo needs no password); mmcoding.host.deploy reads it at import.
os.environ["OSWORLD_CLIENT_PASSWORD"] = common.protocol(BENCH)["client_password"]
paths.use_upstream("ale", "osworld")
ALE_REPO = paths.ALE

import argparse  # noqa: E402
import asyncio  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shlex  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from hybrid_agents.deepseek.common import DeepSeekError, SecretFilter, write_json  # noqa: E402

from mmcoding.benchmarks.common import dsh_to_openclaw_chat  # noqa: E402
from mmcoding.host import deploy  # noqa: E402

# ALE's QEMU provider: the published v1.1 Ubuntu disk, revision-pinned, cached under $RV_DATA/images/ale/qemu.
QEMU_SNAPSHOTS = {
    "cpu-free-ubuntu": {
        "image": "ale-ubuntu22-v1-1",
        "disk_source": "hf://agents-last-exam/ale-images-qcow2/ale-ubuntu22-v1.1.qcow2",
        "image_revision": "40ba46b348a684b7d1b34d18e308aed54737f41a",
        "root": str(paths.ALE_QEMU_ROOT),
        "runner_image": "agentslastexam/ale-qemu:0.2.0",
        "runner_pull_policy": "missing",
        "shm_size": "2g",
        "ready_timeout_s": 900,
    }
}


def qemu_provider():
    from ale_run.environments.providers.qemu import QemuProvider  # noqa: PLC0415

    return QemuProvider({"snapshots": QEMU_SNAPSHOTS})


log = logging.getLogger("rv_ale")
# ALE example_exp.yaml prompt_suffix, verbatim (what ALE appends to the task for its own agents).
PROMPT_SUFFIX = (
    "Complete the task fully autonomously; no user will intervene.\n"
    "When finished, terminate all child processes you started.\n"
    "The user will evaluate your performance based on your delivered artifacts."
)
OUTPUT_ARCHIVE_CAP = 200 * 1024 * 1024
JUDGE_TIMEOUT_S = int(os.getenv("RV_ALE_JUDGE_TIMEOUT_S", str(common.protocol(BENCH)["judge_timeout_s"])))
_RC = "__RV_RC="


def ssh(guest, command: str, timeout: float = 300) -> tuple[int, str]:
    text = guest.command(f"{command}; printf '\\n{_RC}%d' $?", timeout=timeout, check=False)
    body, _, code = text.rpartition(_RC)
    return int(code or 1), body.rstrip("\n")


def scrub_references(guest, task_data_root: str) -> list[str]:
    """Delete every baked reference.7z / decrypted reference dir; fail if any survives."""
    root = shlex.quote(task_data_root)
    script = (
        f"find {root} \\( -name reference.7z -o -name .reference_extract \\) -print -exec rm -rf -- {{}} + ; "
        f"find {root} -mindepth 4 -maxdepth 4 -type d -name reference -print -exec rm -rf -- {{}} + ; "
        f"find {root} \\( -name reference.7z -o -name .reference_extract \\) -printf 'LEFT:%p\\n'; "
        f"find {root} -mindepth 4 -maxdepth 4 -type d -name reference -printf 'LEFT:%p\\n'; true"
    )
    code, out = ssh(guest, f"sudo -n bash -c {shlex.quote(script)}", timeout=600)
    lines = [line for line in out.splitlines() if line.strip()]
    left = [line for line in lines if line.startswith("LEFT:")]
    if code != 0 or left:
        raise RuntimeError(f"reference scrub failed (rc={code}): {left[:5]}")
    return lines


def archive_output(guest, remote_dir: str, out: Path) -> dict:
    code, listing = ssh(guest, f"cd {shlex.quote(remote_dir)} 2>/dev/null && find . -type f -printf '%s %p\\n' | head -5000")
    files = [line.split(" ", 1) for line in listing.splitlines() if " " in line]
    total = sum(int(size) for size, _ in files if size.isdigit())
    report = {"files": len(files), "bytes": total, "archived": False}
    (out / "output_listing.txt").write_text(listing)
    if code == 0 and files and total <= OUTPUT_ARCHIVE_CAP:
        tmp = f"/tmp/rv_output_{os.getpid()}.tar.gz"
        ssh(guest, f"tar -C {shlex.quote(remote_dir)} -czf {tmp} . && chmod 644 {tmp}", timeout=900)
        data = guest.read(tmp)
        if data:
            (out / "output.tar.gz").write_bytes(data)
            report["archived"] = True
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                report["members"] = len(tar.getmembers())
        ssh(guest, f"rm -f {tmp}")
    return report


def agent_wall_timeout(args, meta: dict) -> int:
    """Agent wall cap: ``--agent_timeout`` for every task if given, else the task card's timeout
    (7200 s when absent) bounded by ``--max_timeout``."""
    return getattr(args, "agent_timeout", None) or min(int(meta.get("timeout_s") or 7200), args.max_timeout)


async def run_task_async(args, task_id: str, out: Path, record: dict) -> None:
    from ale_run.environments import task_data as tdp  # noqa: PLC0415
    from ale_run.environments.env import ALEEnv  # noqa: PLC0415
    from ale_run.orchestration.lifecycle import _build_env_spec  # noqa: PLC0415
    from ale_run.tasks.driver import TaskDriver  # noqa: PLC0415
    from ale_run.tasks.loader import TaskLoader  # noqa: PLC0415

    os.chdir(ALE_REPO)  # task modules resolve paths relative to the repo
    install_judge_timeout()
    meta = TaskLoader("tasks/" + task_id).load(0)
    td = meta["task_data"]
    timeout = agent_wall_timeout(args, meta)
    record.update(timeout_s=timeout, task_card_timeout_s=meta.get("timeout_s"),
                  timeout_override=getattr(args, "agent_timeout", None), machine_type=meta.get("machine_type"))
    env = ALEEnv(provider=qemu_provider(), spec=_build_env_spec(meta))
    backend = tdp.select("baked_in_sandbox")
    ref_local = out / "reference.7z.host"
    try:
        await asyncio.to_thread(args.setup_slot.acquire)
        try:
            record["phase"] = "provision"
            await env.reset_async()
            md = env.sandbox.metadata
            write_json(out / "container.json", {k: md.get(k) for k in ("container_name", "vcpus", "memory_gb",
                                                                        "image", "image_provenance", "cua_port")})
            record["phase"] = "setup"
            await backend.stage_input(env.sandbox, td, source="baked_in_sandbox")
            driver = TaskDriver(task_path="tasks/" + task_id, session=env.session, variant=0,
                                os_type=env.sandbox.os, session_rebuilder=env.reset_session)
            await driver.setup()
            pub = Path(args.ssh_private_key + ".pub").read_text().strip()
            r = await env.sandbox.run_command(
                f"umask 077; mkdir -p ~/.ssh; grep -qxF {shlex.quote(pub)} ~/.ssh/authorized_keys 2>/dev/null "
                f"|| echo {shlex.quote(pub)} >> ~/.ssh/authorized_keys", timeout=30)
            if r.returncode != 0:
                raise RuntimeError("could not authorize SSH key")
            import subprocess  # noqa: PLC0415
            ip = subprocess.run(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                                 md["container_name"]], capture_output=True, text=True, check=True).stdout.strip()
            record["phase"] = "deploy"
            deploy.TASK_BY_HOST[ip] = task_id
            guest = deploy.RvGuest(ip, args.ssh_private_key, "user")
            await asyncio.to_thread(guest.connect)
            base = str(Path(td.input_dir).parent)
            ref_remote = base + "/reference.7z"
            code, _ = ssh(guest, f"test -f {shlex.quote(ref_remote)}")
            if code == 0:
                data = await asyncio.to_thread(guest.read, ref_remote)
                if not data:
                    raise RuntimeError("could not copy reference.7z to the host")
                ref_local.write_bytes(data)
                ref_local.chmod(0o600)
            record["reference_on_host"] = ref_local.exists()
            scrubbed = await asyncio.to_thread(scrub_references, guest, env.sandbox.task_data_root)
            record["references_scrubbed"] = len(scrubbed)
            try:
                shot = await env.session.screenshot()
                if shot:
                    (out / "init_screenshot.png").write_bytes(shot)
            except Exception as exc:  # noqa: BLE001 - evidence only
                log.warning("init screenshot failed: %s", exc)
            await asyncio.to_thread(guest.deploy)
        finally:
            args.setup_slot.release()

        record["phase"] = "agent"
        settings = common.settings(args, timeout)
        agent = deploy.RvAgent(guest, args.api_key, settings)
        agent.env = SimpleNamespace(user_simulator=None)
        agent.result_dir = str(out)
        prompt = meta["description"].rstrip() + "\n\n" + PROMPT_SUFFIX
        (out / "task_prompt.txt").write_text(prompt)
        try:
            final, _ = await asyncio.to_thread(agent.predict, prompt)
            record["agent_status"] = "completed"
            record["final_response"] = (final or "")[:2000]
        except DeepSeekError as exc:  # budget / wall clock / runtime: the end state is still graded
            record["agent_status"] = exc.error_kind
            record["agent_error"] = str(exc)[:300]

        record["phase"] = "archive"
        await asyncio.to_thread(guest.connect)
        record["output"] = await asyncio.to_thread(archive_output, guest, td.remote_output_dir, out)
        record["chat_lines"] = dsh_to_openclaw_chat(out / "dsh", out / "chat.jsonl", args.model, args.provider)

        record["phase"] = "evaluate"
        if ref_local.exists():
            await asyncio.to_thread(guest.put, ref_remote, ref_local.read_bytes(), 0o644)
        await env.reset_session()
        driver = TaskDriver(task_path="tasks/" + task_id, session=env.session, variant=0,
                            os_type=env.sandbox.os, session_rebuilder=env.reset_session)
        if td.requires_task_data:
            record["reference_stage"] = await backend.stage_reference(env.sandbox, td, source="baked_in_sandbox")
        t0 = time.time()
        result = await asyncio.wait_for(driver.evaluate(), timeout=args.eval_timeout)
        record["eval_s"] = round(time.time() - t0, 1)
        score = result.get("score") if isinstance(result, dict) else None
        write_json(out / "eval_result.json", result if isinstance(result, dict) else {"raw": str(result)})
        record["score"] = float(score) if isinstance(score, (int, float)) else None
        record["eval_error"] = result.get("error") if isinstance(result, dict) else None
        record["status"] = "completed" if record["score"] is not None else "eval_failed"
    finally:
        try:
            await env.close_async("delete")
        except Exception as exc:  # noqa: BLE001
            record["cleanup_error"] = str(exc)[:300]
        ref_local.unlink(missing_ok=True)


def install_judge_timeout() -> None:
    """Raise ALE's hardcoded 120 s judge-client timeout (multi-image document reviews need longer)."""
    from tasks.utils import evaluation  # noqa: PLC0415

    original = getattr(evaluation._resolve_client_kwargs, "_rv_original", evaluation._resolve_client_kwargs)

    def patched(*a, **kw):
        return {**original(*a, **kw), "timeout": JUDGE_TIMEOUT_S}

    patched._rv_original = original
    evaluation._resolve_client_kwargs = patched


async def regrade_async(args, task_id: str, out: Path, record: dict) -> None:
    """Grade an archived run again on a fresh VM (grading failures only; the agent run is kept)."""
    from ale_run.environments import task_data as tdp  # noqa: PLC0415
    from ale_run.environments.env import ALEEnv  # noqa: PLC0415
    from ale_run.orchestration.lifecycle import _build_env_spec  # noqa: PLC0415
    from ale_run.tasks.driver import TaskDriver  # noqa: PLC0415
    from ale_run.tasks.loader import TaskLoader  # noqa: PLC0415

    os.chdir(ALE_REPO)
    install_judge_timeout()
    meta = TaskLoader("tasks/" + task_id).load(0)
    td = meta["task_data"]
    env = ALEEnv(provider=qemu_provider(), spec=_build_env_spec(meta))
    backend = tdp.select("baked_in_sandbox")
    try:
        await asyncio.to_thread(args.setup_slot.acquire)
        try:
            await env.reset_async()
            await backend.stage_input(env.sandbox, td, source="baked_in_sandbox")
            driver = TaskDriver(task_path="tasks/" + task_id, session=env.session, variant=0,
                                os_type=env.sandbox.os, session_rebuilder=env.reset_session)
            await driver.setup()
        finally:
            args.setup_slot.release()
        remote_tar = f"/tmp/rv_regrade_{os.getpid()}.tar.gz"
        await env.sandbox.write_file(remote_tar, (out / "output.tar.gz").read_bytes())
        r = await env.sandbox.run_command(
            f"rm -rf {shlex.quote(td.remote_output_dir)} && mkdir -p {shlex.quote(td.remote_output_dir)} && "
            f"tar -C {shlex.quote(td.remote_output_dir)} -xzf {remote_tar} && rm -f {remote_tar}", timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"could not restore archived output: {(r.stderr or r.stdout or '')[:300]}")
        if td.requires_task_data:
            record["reference_stage"] = await backend.stage_reference(env.sandbox, td, source="baked_in_sandbox")
        t0 = time.time()
        result = await asyncio.wait_for(driver.evaluate(), timeout=args.eval_timeout)
        record["eval_s"] = round(time.time() - t0, 1)
        score = result.get("score") if isinstance(result, dict) else None
        write_json(out / "eval_result.json", result if isinstance(result, dict) else {"raw": str(result)})
        record["score"] = float(score) if isinstance(score, (int, float)) else None
        record["eval_error"] = result.get("error") if isinstance(result, dict) else None
        record["status"] = "completed" if record["score"] is not None else "eval_failed"
    finally:
        try:
            await env.close_async("delete")
        except Exception as exc:  # noqa: BLE001
            record["cleanup_error"] = str(exc)[:300]


def regrade(args, task_id: str) -> dict:
    out = args.result_dir / task_id.replace("/", "__")
    record = json.loads((out / "outcome.json").read_text())
    entry = {"started": time.time(), "previous_status": record.get("status"), "previous_error": record.get("eval_error")}
    try:
        asyncio.run(regrade_async(args, task_id, out, record))
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    entry["finished"] = time.time()
    record.setdefault("regraded", []).append(entry)
    write_json(out / "outcome.json", record)
    return record


def needs_regrade(out: Path) -> bool:
    outcome = out / "outcome.json"
    return (outcome.exists() and json.loads(outcome.read_text()).get("status") == "eval_failed"
            and (out / "output.tar.gz").exists())


def run_task(args, task_id: str) -> dict:
    out = args.result_dir / task_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log")
    handler.addFilter(SecretFilter((args.api_key,)))
    logging.getLogger().addHandler(handler)
    record = {"task_id": task_id, "harness": args.harness, "model": args.model, "started": time.time(),
              "status": "error", "phase": "start"}
    try:
        asyncio.run(run_task_async(args, task_id, out, record))
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        record["traceback"] = traceback.format_exc()[-4000:]
    record["finished"] = time.time()
    record["minutes"] = round((record["finished"] - record["started"]) / 60, 1)
    write_json(out / "outcome.json", record)
    logging.getLogger().removeHandler(handler)
    return record


def config(argv=None):
    proto = common.protocol(BENCH)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_agent_args(p, BENCH)
    p.add_argument("--max_timeout", type=int, default=proto["timeout_s"],
                   help="agent wall cap (s); the task card's timeout applies when smaller")
    p.add_argument("--agent_timeout", type=int, default=None,
                   help="one agent wall cap (s) for every task instead of the task card's timeout")
    p.add_argument("--eval_timeout", type=int, default=proto["eval_timeout_s"])
    p.add_argument("--num_envs", type=int, default=2)
    p.add_argument("--setup_parallelism", type=int, default=2)
    p.add_argument("--regrade", action="store_true",
                   help="only grade again the finished tasks whose grading failed (archived output, fresh VM)")
    args = p.parse_args(argv)
    if args.agent_timeout is not None and args.agent_timeout <= 0:
        p.error("--agent_timeout must be positive")
    common.finish_agent_args(p, args, BENCH)
    if not os.getenv("ALE_REFERENCE_ARCHIVE_PASSWORD"):
        p.error("ALE_REFERENCE_ARCHIVE_PASSWORD is required for grading (public; see ALE secret/.env.example)")
    return args


def main(argv=None) -> None:
    args = config(argv)
    task_ids = args.tasks
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(SecretFilter((args.api_key,)))
    deploy.configure_harness(args.harness, args.base_url)
    deploy.RvGuest.network_guard = args.network_guard
    args.result_dir = args.result_dir.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    if args.regrade:
        todo, regrade_todo = [], [t for t in task_ids if needs_regrade(args.result_dir / t.replace("/", "__"))]
    else:
        todo, regrade_todo = [t for t in task_ids if not (args.result_dir / t.replace("/", "__") / "outcome.json").exists()], []
    manifest_path = args.result_dir / "manifest.json"
    settings = common.settings(args, args.agent_timeout or args.max_timeout).public()
    entry = {"benchmark": "ALE v1.1 (Linux CPU tasks, QEMU provider, ale-ubuntu22-v1.1)", "tasks": task_ids,
             "harness": args.harness, "model": args.model, "settings": settings, "prompt_suffix": PROMPT_SUFFIX,
             "grading": "ALE task evaluate() host-side; reference.7z restored only after the agent",
             "network_guard": args.network_guard, "rv_content_hash": deploy.rv_content_hash(),
             "started": time.time()}
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is None:
        write_json(manifest_path, entry)
    elif todo or regrade_todo:
        previous.setdefault("resumed", []).append({"started": entry["started"], "run": todo, "regrade": regrade_todo,
                                                   "settings": settings})
        write_json(manifest_path, previous)
    with mp.Manager() as manager:
        args.setup_slot = manager.BoundedSemaphore(args.setup_parallelism)
        with mp.Pool(max(1, min(args.num_envs, len(todo) + len(regrade_todo) or 1))) as pool:
            pool.starmap(run_task, [(args, t) for t in todo])
            pool.starmap(regrade, [(args, t) for t in regrade_todo])
    summary = [json.loads(path.read_text()) for path in sorted(args.result_dir.glob("*/outcome.json"))]
    write_json(args.result_dir / "summary.json", summary)
    scored = [r["score"] for r in summary if r.get("score") is not None]
    threshold = common.protocol(BENCH)["pass_threshold"]
    print(json.dumps({"harness": args.harness, "tasks": len(task_ids), "scored": len(scored),
                      "passed": sum(s >= threshold for s in scored),
                      "mean_score": round(sum(scored) / len(scored), 3) if scored else None}, indent=1))


if __name__ == "__main__":
    main()
