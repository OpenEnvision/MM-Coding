#!/usr/bin/env python3
"""Guest worker of the harnesses (mmcoding/harnesses.py).

Runs the *official* OSWorld-V2 DSH worker (``hybrid_agents/deepseek/worker.py``,
uploaded unchanged into the sibling ``code`` directory) inside the agent sandbox
started by ``rv-run worker``. Only the DSH composition and the run wrapper
differ from the official worker:

* provider route + tool policy: ``patches/common.patch.yml`` + ``<harness>.patch.yml``
  (+ ``ask_user.patch.yml`` when the task has a user simulator);
* model-call budget: an awaited native ``agent/request`` hook (``call_budget.mjs``)
  stops BEFORE the next provider request after the decision budget is consumed;
  ``budget.json`` records the count; budget exhaustion is reported like a timeout
  (``error_kind == "timeout"``);
* the runtime gets no DISPLAY/XAUTHORITY, except DISPLAY=:0 in the X harnesses
  (os_helix / mm_coding / vision_tool / cli), whose sandbox exposes the X socket;
* os_helix: requests go through the local helix_shim.py (ten most recent screenshots,
  one action per step, max_tokens recomputation);
* vision_tool: the vision MCP server answers with ``vision_model`` /
  ``vision_reasoning_effort`` (default: the run's own model and reasoning effort),
  exported as ``RV_VISION_MODEL`` / ``RV_VISION_EFFORT``;
* output-length stops: a turn that ends because one response hit the output
  token cap (``max-tokens``) is continued in the same session with one fixed
  message, at most ``MAX_CONTINUATIONS`` times, while budget remains (every harness).
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../osworld-dsh/rv
CODE = HERE.parent / "code"  # official worker.py / common.py / helpers.py
sys.path.insert(0, str(CODE))

import worker as official  # noqa: E402
from common import error_kind, scrub, write_json  # noqa: E402

SDK_VERSION = "0.1.5rc1"
# Copies of mmcoding/harnesses.py (this file runs inside the VM without the package; tests check they agree).
HARNESSES = ("gui", "os_helix", "mm_coding", "vision_tool", "cli")
X_HARNESSES = ("os_helix", "mm_coding", "vision_tool", "cli")
MAX_CONTINUATIONS = 3
CONTINUE_PROMPT = (
    "Your previous response was cut off because it reached the output length limit. Continue the task "
    "from where you stopped. Keep each response short: split long files, code or reasoning across several "
    "tool calls instead of producing them in a single response."
)


def count_model_call(notification, session_id: str | None) -> bool:
    """True when the notification is one root-session model call (``step/start``)."""
    if getattr(notification, "method", None) != "session.event":
        return False
    payload = getattr(notification, "payload", None) or {}
    if session_id is not None and payload.get("sessionId") != session_id:
        return False
    event = payload.get("event") or {}
    return isinstance(event, dict) and event.get("type") == "step/start"


def screenshot_png() -> tuple[bytes, tuple[int, int]]:
    """One desktop screenshot, taken in a child process (python-xlib outside the DSH worker)."""
    code = f"import sys;sys.path.insert(0,{str(HERE)!r});from rv_desktop import Desktop;" \
           "sys.stdout.buffer.write(Desktop(0.0).screenshot_png()[0])"
    return subprocess.check_output(["/usr/bin/python3", "-c", code]), (0, 0)


def rv_harness_class(settings: dict, rv: dict, attempt: Path):
    from deepseek_harness import DeepSeekHarness

    harness = settings["harness"]
    if harness not in HARNESSES:
        raise ValueError(f"unknown harness {harness!r}")
    patches = [HERE / "patches" / "common.patch.yml", HERE / "patches" / f"{harness}.patch.yml"]
    if rv.get("ask_enabled"):
        patches.append(HERE / "patches" / "ask_user.patch.yml")
    effort = settings.get("reasoning_effort")
    budget = int(settings["call_budget"])
    offset = int(rv.get("call_offset") or 0)
    budget_file = attempt / "budget.json"
    # Notifications are asynchronous. Enforce the budget INSIDE the runtime's
    # awaited request waterfall, not by raising in a Python notification reader.
    write_json(attempt / "call_budget.json", {"call_budget": budget, "call_offset": offset})
    control_patch = attempt / "call_budget.patch.yml"
    control_patch.write_text(
        "- insert:\n    - id: rv-call-budget\n"
        f"      name: '{HERE / 'call_budget.mjs'}'\n"
    )
    patches.append(control_patch)

    class RvHarness(DeepSeekHarness):
        def __init__(self, **kwargs) -> None:
            # The official worker exported DEEPSEEK_BASE_URL/API_KEY for its
            # route; the harness route reads the same values under its own names.
            os.environ["RV_LLM_BASE_URL"] = os.environ["DEEPSEEK_BASE_URL"]
            os.environ["RV_LLM_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]
            if harness == "os_helix":
                sys.path.insert(0, str(HERE))
                from helix_shim import HelixShim
                self._helix_shim = HelixShim(os.environ["DEEPSEEK_BASE_URL"], os.environ["DEEPSEEK_API_KEY"],
                                             attempt, screenshot_png)
                os.environ["RV_LLM_BASE_URL"] = self._helix_shim.url
            os.environ["RV_HOME"] = str(HERE)
            os.environ["OSWORLD_DSH_CODE"] = str(CODE)
            for name in ("DISPLAY", "XAUTHORITY"):
                os.environ.pop(name, None)
            if harness in X_HARNESSES:
                os.environ["DISPLAY"] = ":0"  # admitted by the X server's localuser rule
            vision_effort = settings.get("vision_reasoning_effort") or effort
            os.environ["RV_VISION_MODEL"] = str(settings.get("vision_model") or settings.get("model") or "")
            os.environ["RV_VISION_EFFORT"] = "" if vision_effort in (None, "default", "off") else str(vision_effort)
            os.environ["RV_VISION_COORDS"] = str(settings.get("vision_coords") or "")
            os.environ["RV_VISION_THINKING"] = str(settings.get("vision_thinking") or "")
            kwargs.update(
                provider=settings["provider"],
                profile="sdk",
                patches=tuple(str(p) for p in patches),
                reasoning_effort=None if effort == "default" else effort,
                initialize_timeout_seconds=180,
                # The wall-clock alarm in main() and the bounded systemd safety
                # timer own the task deadline.
                request_timeout_seconds=None,
            )
            super().__init__(**kwargs)

        def run(self, input, *, session_id=None, on_notification=None):  # noqa: A002
            calls = {"n": 0, "started": time.time()}

            def guarded(notification) -> None:
                if on_notification is not None:
                    on_notification(notification)
                if count_model_call(notification, session_id):
                    calls["n"] += 1
                    # Telemetry only; the native plugin exclusively owns
                    # budget.json so an asynchronous reader cannot overwrite
                    # its exhausted record.

            result = super().run(input, session_id=session_id, on_notification=guarded)
            continuations = []
            while (result.finish_reason == "max-tokens" and len(continuations) < MAX_CONTINUATIONS
                   and offset + calls["n"] < budget):
                continuations.append({"after_calls": calls["n"], "time": round(time.time(), 1)})
                write_json(attempt / "continuations.json", continuations)
                more = super().run(CONTINUE_PROMPT, session_id=result.session_id, on_notification=guarded)
                result = type(result)(
                    session_id=more.session_id,
                    final_response=more.final_response,
                    finish_reason=more.finish_reason,
                    events=[*result.events, *more.events],
                    notifications=[*result.notifications, *more.notifications],
                )
            return result

    return RvHarness


def main() -> None:
    os.umask(0o077)
    path = Path(sys.argv[1])
    config = json.loads(path.read_text())
    path.unlink()  # one-time secret payload, as in the official worker
    root = Path(config["attempt"])
    # Optional separate endpoint of the vision model (vision_tool, when the policy model's route cannot
    # serve the vision model). Absent = the policy model's route.
    vision_endpoint = (config.get("rv") or {}).get("vision_endpoint") or {}
    secrets = tuple(s for s in (config["api_key"], vision_endpoint.get("api_key")) if s)
    if vision_endpoint.get("base_url") and vision_endpoint.get("api_key"):
        os.environ["RV_VISION_BASE_URL"] = str(vision_endpoint["base_url"])
        os.environ["RV_VISION_API_KEY"] = str(vision_endpoint["api_key"])

    started = time.time()
    wall_budget = config["settings"]["timeout"]

    def timeout(_signum, _frame):
        remaining = wall_budget - (time.time() - started)
        if remaining > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.1, remaining))
            return
        raise TimeoutError("DSH task wall timeout")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(wall_budget)
    try:
        for package in ("deepseek-harness-sdk", "deepseek-harness-runtime-bin"):
            if importlib.metadata.version(package) != SDK_VERSION:
                raise RuntimeError(f"Unexpected {package} version")
        harness_class = rv_harness_class(config["settings"], config.get("rv") or {}, root)
        official.run(config, harness_class=harness_class)
        status = {"status": "completed"}
    except Exception as exc:
        status = {
            "status": "error",
            "error_kind": getattr(exc, "error_kind", error_kind(exc)),
            "error_type": type(exc).__name__,
            "message": scrub(str(exc), secrets),
            "exception_chain": scrub(traceback.format_exc(), secrets),
        }
        if "RV_BUDGET_EXHAUSTED" in str(exc):
            status["budget_exhausted"] = True
    finally:
        signal.alarm(0)
    write_json(root / "status.json", status)


if __name__ == "__main__":
    main()
