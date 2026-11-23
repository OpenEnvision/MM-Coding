"""Shared runner pieces: CLI arguments, run settings, task lists and trajectory conversion."""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path

from mmcoding import harnesses, paths

EFFORTS = ("default", "off", "low", "high", "max")
BENCHES = ("osworld2", "weavebench", "ale")


@lru_cache(maxsize=None)
def protocol(bench: str) -> dict:
    """Run defaults of one benchmark (``configs/protocol.json``)."""
    return json.loads((paths.CONFIGS / "protocol.json").read_text())[bench]


def user_setting(bench: str, key: str, env: str) -> str:
    """A setting the user must supply: ``$env`` if set, else ``protocol.json`` ``<bench>.<key>``.

    Raises ``SystemExit`` with instructions when neither is set: no judge or evaluator model is
    chosen on the user's behalf.
    """
    value = os.getenv(env) or protocol(bench).get(key)
    if not value:
        raise SystemExit(f"{bench}: {key} is not set; export {env} or set \"{key}\" in the "
                         f"\"{bench}\" block of configs/protocol.json")
    return str(value)


def _harness(value: str) -> str:
    try:
        return harnesses.normalize(value)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_agent_args(parser: argparse.ArgumentParser, bench: str) -> None:
    """Arguments every runner shares (harness, policy model route, budget, vision model, tasks)."""
    proto = protocol(bench)
    parser.add_argument("--harness", type=_harness, required=True,
                        help=f"one of {', '.join(harnesses.HARNESSES)} (paper names such as MM-Coding work too)")
    parser.add_argument("--model", required=True, help="policy model id, as listed in configs/models.yaml")
    parser.add_argument("--provider", default="openai", help="model route id in configs/models.yaml")
    parser.add_argument("--base_url", default=os.getenv("RV_LLM_BASE_URL"),
                        help="OpenAI-compatible endpoint of the policy model (default: $RV_LLM_BASE_URL)")
    parser.add_argument("--api_key_env", default="RV_LLM_API_KEY",
                        help="environment variable holding the policy model key")
    parser.add_argument("--reasoning_effort", choices=EFFORTS, default=proto["reasoning_effort"])
    parser.add_argument("--max_tokens", type=int, default=proto["max_tokens"])
    parser.add_argument("--call_budget", type=int, default=proto["call_budget"],
                        help=f"max model calls per task (protocol: {proto['call_budget']})")
    parser.add_argument("--vision_model", default=proto.get("vision_model") or "",
                        help="vision model answering ask_vision (vision_tool harness); empty: --model")
    parser.add_argument("--vision_reasoning_effort", choices=("", *EFFORTS),
                        default=proto.get("vision_reasoning_effort", ""),
                        help="vision model reasoning effort; empty: --reasoning_effort")
    parser.add_argument("--vision_coords", choices=("", "pixel", "norm1000"), default=proto.get("vision_coords", ""),
                        help="coordinate convention of the vision model; empty: per-model profile")
    parser.add_argument("--vision_thinking", choices=("", "on", "off"), default=proto.get("vision_thinking", ""),
                        help="vision model thinking; empty: per-model profile")
    parser.add_argument("--network-guard", "--network_guard", dest="network_guard", action="store_true",
                        help="route guest web traffic through the host network guard while the agent runs "
                             "(off by default)")
    parser.add_argument("--ssh_private_key", default=str(paths.SSH_KEY))
    parser.add_argument("--tasks", default="all",
                        help=f"'all' (default: configs/tasks/{bench}.json), @file.json or comma-separated ids")
    parser.add_argument("--result_dir", type=Path, default=None,
                        help="default: $RV_RESULTS/<bench>/<model>/<timestamp>/<harness>")


def finish_agent_args(parser: argparse.ArgumentParser, args, bench: str) -> None:
    """Resolve tasks, the policy credentials and the result directory after parsing."""
    try:
        args.tasks = load_tasks(args.tasks, bench)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"--tasks: {exc}")
    args.api_key = os.getenv(args.api_key_env)
    if not args.api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    if not args.base_url:
        parser.error("--base_url (or RV_LLM_BASE_URL) is required")
    if args.result_dir is None:
        args.result_dir = default_result_dir(bench, args.model, args.harness)


def settings(args, timeout: int):
    """``RvSettings`` from the shared arguments (import deferred: the host side needs the OSWorld checkout)."""
    from mmcoding.host import deploy

    return deploy.RvSettings(
        args.model, args.base_url, args.reasoning_effort, args.max_tokens, timeout,
        harness=args.harness, provider=args.provider, call_budget=args.call_budget,
        vision_model=args.vision_model, vision_reasoning_effort=args.vision_reasoning_effort,
        vision_coords=args.vision_coords, vision_thinking=args.vision_thinking,
    )


def load_tasks(spec: str, bench: str) -> list[str]:
    """Task ids from ``all`` (the complete list of the benchmark), ``@path.json`` or ``a,b,c``.

    A JSON file holds either a list of ids or an object with a ``tasks`` list.
    """
    spec = spec.strip()
    source = paths.task_file(bench) if spec == "all" else Path(spec[1:]).expanduser() if spec.startswith("@") else None
    if source is not None:
        data = json.loads(source.read_text())
        tasks = [str(t) for t in (data["tasks"] if isinstance(data, dict) else data)]
    else:
        tasks = [t.strip() for t in spec.split(",") if t.strip()]
    if not tasks:
        raise ValueError(f"no tasks in {spec!r}")
    duplicates = sorted({t for t in tasks if tasks.count(t) > 1})
    if duplicates:
        raise ValueError(f"duplicate tasks {duplicates}")
    return tasks


def default_result_dir(bench: str, model: str, harness: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return paths.result_dir(bench, model, stamp, harness)


# ----------------------------------------------------------------- trajectory


def _read_events(dsh: Path) -> list[dict]:
    for name in ("notifications.jsonl", "notifications.jsonl.gz"):
        path = dsh / name
        if path.exists():
            raw = gzip.decompress(path.read_bytes()) if name.endswith(".gz") else path.read_bytes()
            events = []
            for line in raw.decode("utf-8", "replace").splitlines():
                try:
                    events.append(((json.loads(line).get("payload") or {}).get("event")) or {})
                except ValueError:
                    continue
            return events
    return []


def _result_text(blocks) -> str:
    parts = []
    for block in blocks or []:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "image":
            att = block.get("attachment") or {}
            parts.append(f"[image {att.get('width')}x{att.get('height')}]")
    return "\n".join(parts)


def dsh_to_openclaw_chat(dsh: Path, out: Path, model: str, provider: str = "openai") -> int:
    """Convert the DSH event log of one run into WeaveBench's OpenClaw v3 ``chat.jsonl``.

    The trajectory-aware judge reads user messages, assistant text / reasoning / tool calls and
    tool results; images become ``[image WxH]`` placeholders. Returns the number of lines written.
    """
    events = _read_events(dsh)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    lines = [{"type": "session", "version": 3, "id": dsh.parent.name, "timestamp": now, "cwd": "/home/user"},
             {"type": "model_change", "id": "m0", "parentId": None, "timestamp": now, "provider": provider,
              "modelId": model}]
    for ev in events:
        kind, data = ev.get("type"), ev.get("data") or {}
        if kind == "user/message":
            text = "\n".join(b.get("text", "") for b in data.get("content") or [] if b.get("type") == "text")
            lines.append({"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": text}]}})
        elif kind == "assistant/message":
            content = []
            for b in (data.get("message") or {}).get("content") or []:
                if b.get("type") == "reasoning":
                    content.append({"type": "thinking", "thinking": b.get("text", "")})
                elif b.get("type") == "text":
                    content.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "tool-call":
                    try:
                        args = json.loads(b.get("arguments") or "{}")
                    except ValueError:
                        args = {"raw": b.get("arguments")}
                    content.append({"type": "tool_use", "id": b.get("id"), "name": b.get("name"), "input": args})
            lines.append({"type": "message", "message": {"role": "assistant", "content": content}})
        elif kind == "tool/result":
            content = []
            for b in (data.get("message") or {}).get("content") or []:
                if b.get("type") == "tool-result":
                    content.append({"type": "tool_result", "tool_use_id": b.get("toolCallId"),
                                    "content": _result_text(b.get("content")), "is_error": bool(b.get("isError"))})
            lines.append({"type": "message", "message": {"role": "user", "content": content}})
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
    return len(lines)
