"""Host side of the five harnesses (``mmcoding.harnesses``).

Thin adapters over the official OSWorld-V2 DSH integration (``hybrid_agents.deepseek`` in the
pinned OSWorld-V2 checkout), so the official runner's ``run_case`` (setup transport, evaluation,
evidence, outcome files, timeout scoring) runs our harnesses unchanged:

* ``RvSettings``        official settings + harness / provider route / model-call budget / vision
                        answerer settings;
* ``rv_runtime_prompt`` one prompt for every harness; harnesses differ only in the capability
                        paragraph that describes their tools (``mmcoding/prompts``);
* ``RvGuest``           official SSH ``Guest``; deploy installs the sandbox (``guest/rv_run.py
                        setup``), launch starts the worker in the sandbox as a transient system
                        unit, stop/quiescence use the system manager, collect adds audit / broker /
                        bus-proxy evidence;
* ``RvAgent``           official ``DeepSeekAgent``; records whether the task has a user simulator
                        (to mount ``ask_user``) and carries the per-task call budget across phases;
* ``docker_new_env`` / ``docker_terminate_case``  Docker provider instead of AWS.

Every harness uses the same model, prompt skeleton, call budget, sandbox and evaluator. The
optional network guard (``mmcoding.host.guard``) is off unless ``install(..., network_guard=True)``.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import os
import re
import shlex
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import paramiko

from hybrid_agents.deepseek import agent as official_agent
from hybrid_agents.deepseek.agent import DeepSeekAgent
from hybrid_agents.deepseek.common import DeepSeekError, HarnessSettings, scrub, task_value, write_json
from hybrid_agents.deepseek.transport import GUEST_ROOT, UV_VERSION, Guest

from mmcoding import harnesses as registry
from mmcoding import paths
from mmcoding.host import sites

HOST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = HOST_DIR.parent
OFFICIAL_DIR = Path(official_agent.__file__).resolve().parent
GUEST_DIR = PACKAGE_DIR / "guest"
PATCH_DIR = PACKAGE_DIR / "profiles"
PROMPT_DIR = PACKAGE_DIR / "prompts"
ASSET_DIR = paths.SANDBOX_ASSETS  # wheelhouse/ + debs/ (large, data directory)
SANDBOX_LISTS = paths.CONFIGS / "sandbox"  # requirements.txt / constraints.txt (versioned)
MODELS_FILE = paths.CONFIGS / "models.yaml"  # user-edited model list of the provider route
MODELS_EXAMPLE = paths.CONFIGS / "models.example.yaml"  # used when models.yaml does not exist
RV_GUEST = GUEST_ROOT + "/rv"
ASSETS_GUEST = GUEST_ROOT + "/rv-assets"
EVIDENCE_GUEST = GUEST_ROOT + "/rv-evidence"
RV_RUN = "/usr/local/sbin/rv-run"
GUEST_SUFFIXES = {".py", ".json", ".txt", ".mjs"}  # guest files uploaded and hashed into the provenance

SDK_VERSION = "0.1.5rc1"
HARNESSES = registry.HARNESSES
X_HARNESSES = registry.X_HARNESSES  # sandboxed code may use the X display
BROKER = registry.BROKER  # desktop broker tool profile per harness (absent: no broker)
EFFORTS = ("default", "off", "low", "high", "max")
CLIENT_PASSWORD = os.getenv("OSWORLD_CLIENT_PASSWORD", "osworld-public-evaluation")
GUEST_TCP_CC = "bbr"
CONTAINER_PORTS = {"server_port": 5000, "chromium_port": 9222, "vnc_port": 8006, "vlc_port": 8080}

# Placeholder lines of profiles/common.patch.yml, replaced at deploy (render_common_patch).
IMAGE_BYTES_LINE = "        maxRequestImageBytes: __RV_MAX_REQUEST_IMAGE_BYTES__"
MODELS_LINE = "        models: __RV_MODELS__"

# Task-specific evaluator details of OSWorld 2.0 (its task classes are gated) live in an optional,
# user-supplied JSON file: {"hidden_paths": {task: [path, ...]}, "evaluator_state_patterns":
# {task: regex}}. Missing file = nothing hidden.
TASK_PRIVATE_FILE = paths.OSWORLD_V2_TASK_PRIVATE


def load_task_private(path: Path | None = None) -> dict:
    path = path or TASK_PRIVATE_FILE
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


# Evaluator-only data that a task's setup places inside the VM. Hidden from the agent sandbox;
# the evaluator runs outside the sandbox and still reads it.
TASK_HIDDEN_PATHS: dict[str, list[str]] = load_task_private().get("hidden_paths", {})
TASK_BY_HOST: dict[str, str] = {}  # vm ip -> task id (one case per worker process)


# ------------------------------------------------------------------------- prompt


def prompt_text(name: str) -> str:
    """A text file of ``mmcoding/prompts``. One trailing newline is dropped; every other character
    (including trailing spaces) is part of the prompt."""
    text = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def _fill(template: str, **values: str) -> str:
    """Single-pass ``{name}`` substitution; inserted values are never rescanned."""
    return re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)], template)


def capability(harness: str) -> str:
    """The capability paragraph of a harness (``prompts/capability/<harness>.txt``)."""
    harness = registry.get(harness).id
    return _fill(prompt_text(f"capability/{harness}.txt"),
                 with_x=prompt_text("with_x.txt"), serve_note=prompt_text("serve_note.txt"))


CAPABILITIES = {h: capability(h) for h in HARNESSES}


def rv_runtime_prompt(instruction: str, remote: str, task_date: str | None, ask: bool, *, harness: str) -> str:
    """One prompt for every harness; only the capability paragraph (tool description) differs."""
    del remote
    context = "You are operating the task's Ubuntu desktop. " + capability(harness) + "\n"
    context += prompt_text("closing.txt") + "\n"
    context += prompt_text("report_note.txt") + "\n"
    if task_date:
        context += _fill(prompt_text("task_date.txt"), task_date=str(task_date)) + "\n"
    if ask:
        context += prompt_text("ask_user.txt") + "\n"
    return context + "\nTASK INSTRUCTION (authoritative, verbatim):\n" + instruction


# ------------------------------------------------------------- model route patch


def models_file() -> Path:
    """``configs/models.yaml`` when present, else the shipped example (the models of the paper)."""
    return MODELS_FILE if MODELS_FILE.is_file() else MODELS_EXAMPLE


def load_models(path: Path | None = None) -> list[dict]:
    """The ``models`` list of a models file: DSH model entries (id, contextWindow, maxTokens, ...)."""
    import yaml

    path = path or models_file()
    data = yaml.safe_load(path.read_text())
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list) or not models:
        raise ValueError(f"{path}: expected a non-empty 'models' list")
    ids = []
    for entry in models:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"].strip():
            raise ValueError(f"{path}: every model needs a non-empty string id")
        ids.append(entry["id"])
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate model id")
    return models


def render_common_patch(harness: str, models: list[dict] | None = None, template: str | None = None) -> str:
    """``profiles/common.patch.yml`` with the harness's image byte bound and the model list (JSON flow
    style, which is valid YAML) in place of the two placeholder lines."""
    template = template if template is not None else (PATCH_DIR / "common.patch.yml").read_text()
    models = models if models is not None else load_models()
    lines = template.split("\n")
    for line in (IMAGE_BYTES_LINE, MODELS_LINE):
        if lines.count(line) != 1:
            raise ValueError(f"common.patch.yml must contain the line {line.strip()!r} exactly once")
    image_bytes = registry.get(harness).max_request_image_bytes
    lines[lines.index(IMAGE_BYTES_LINE)] = IMAGE_BYTES_LINE.replace("__RV_MAX_REQUEST_IMAGE_BYTES__", str(image_bytes))
    lines[lines.index(MODELS_LINE)] = MODELS_LINE.replace("__RV_MODELS__", json.dumps(models))
    return "\n".join(lines)


# --------------------------------------------------------------------- provenance


def rv_files() -> list[Path]:
    """Every file that defines the harnesses' behaviour, for the provenance hash."""
    files = [PACKAGE_DIR / "harnesses.py", HOST_DIR / "deploy.py", HOST_DIR / "audit.py", HOST_DIR / "sites.py"]
    files += sorted(p for p in GUEST_DIR.iterdir() if p.is_file() and p.suffix in GUEST_SUFFIXES)
    files += sorted(PATCH_DIR.glob("*.yml"))
    files += sorted(PROMPT_DIR.rglob("*.txt"))
    return files + [models_file()]


def rv_content_hash() -> str:
    digest = hashlib.sha256()
    for path in rv_files():
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def asset_files() -> list[Path]:
    """Files installed in every guest: wheels and debs (data directory) + the pinned lists (repository)."""
    files = sorted((ASSET_DIR / "wheelhouse").glob("*.whl")) + sorted((ASSET_DIR / "debs").glob("*.deb"))
    return files + [SANDBOX_LISTS / "requirements.txt", SANDBOX_LISTS / "constraints.txt"]


# ----------------------------------------------------------------------- settings


@dataclass(frozen=True)
class RvSettings(HarnessSettings):
    harness: str = "gui"
    provider: str = "openai"
    call_budget: int = 500
    # Vision-Tool answerer; empty = the run's own model / reasoning effort.
    vision_model: str = ""
    vision_reasoning_effort: str = ""
    vision_coords: str = ""    # "" (per-model profile) | pixel | norm1000
    vision_thinking: str = ""  # "" (per-model profile) | on | off

    def __post_init__(self) -> None:  # replaces the DeepSeek-specific checks
        if self.harness not in HARNESSES:
            raise ValueError(f"harness must be one of {HARNESSES}")
        if self.reasoning_effort not in EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {EFFORTS}")
        if self.vision_reasoning_effort not in ("", *EFFORTS):
            raise ValueError(f"vision_reasoning_effort must be empty or one of {EFFORTS}")
        if self.vision_coords not in ("", "pixel", "norm1000") or self.vision_thinking not in ("", "on", "off"):
            raise ValueError("vision_coords must be pixel|norm1000 and vision_thinking on|off (or empty)")
        if min(self.max_tokens, self.timeout, self.stream_idle_timeout, self.call_budget) <= 0:
            raise ValueError("Token, timeout and budget limits must be positive")
        if not self.model or not self.base_url.startswith("https://"):
            raise ValueError("A model and HTTPS provider endpoint are required")

    def public(self) -> dict[str, Any]:
        spec = registry.get(self.harness)
        servers = [f"desktop (rv_desktop.py --profile {spec.broker} via socat relay)"] if spec.broker else []
        if spec.helix:
            servers.append("helix (helix_code_mcp.py: run_code)")
        if spec.vision_tool:
            servers.append("vision (vision_mcp.py: text-only, perception-only answers from a stateless answerer call)")
        endpoint = os.environ.get("RV_VISION_BASE_URL", "").strip()
        return {
            **asdict(self),
            "sdk_version": SDK_VERSION,
            "profile": "sdk",
            "patches": ["common.patch.yml", f"{self.harness}.patch.yml", "ask_user.patch.yml (simulator tasks only)"],
            "agent_loop": "DSH SDK agent" + (" with guest/helix_shim.py" if spec.helix else ""),
            "mcp_servers": servers + ["report", "ask (simulator tasks only)"],
            "network": "open internet (no private ranges)",
            "network_guard": bool(RvGuest.network_guard),
            "budget_unit": "root-session model calls (step/start events)",
            "budget_scope": "per task, cumulative across phases (call_offset)",
            "vision_answerer": ({"model": self.vision_model or self.model,
                                 "reasoning_effort": self.vision_reasoning_effort or self.reasoning_effort,
                                 "coords": self.vision_coords or "profile",
                                 "thinking": self.vision_thinking or "profile",
                                 "state": "stateless",
                                 # Recorded only for a separate answerer endpoint (absent = the policy route).
                                 **({"endpoint": endpoint} if endpoint else {})}
                                if spec.vision_tool else None),
            "sandbox": "rv-run: PrivateUsers + NoNewPrivileges + netns rv + ICE/Xauthority/AT-SPI/"
                       "systemd/pipewire hidden + filtered session bus; X11 socket "
                       + ("reachable (DISPLAY=:0)" if self.harness in X_HARNESSES else "hidden"),
            "guest_tcp_congestion_control": GUEST_TCP_CC,
            "rv_content_sha256": rv_content_hash(),
        }


# -------------------------------------------------------------------------- guest


def _sudo(command: str) -> str:
    return f"echo {shlex.quote(CLIENT_PASSWORD)} | sudo -S -p '' {command}"


def vision_endpoint() -> dict:
    """Optional separate endpoint of the Vision-Tool answerer, from the host environment:
    ``RV_VISION_BASE_URL`` (https) and ``RV_VISION_API_KEY``. Empty when unset: the answerer then
    uses the policy model's route."""
    base = os.environ.get("RV_VISION_BASE_URL", "").strip()
    if not base:
        return {}
    key = os.environ.get("RV_VISION_API_KEY", "")
    if not base.startswith("https://") or not key:
        raise ValueError("RV_VISION_BASE_URL needs https:// and a non-empty RV_VISION_API_KEY")
    return {"base_url": base, "api_key": key}


class RvGuest(Guest):
    """Official SSH guest; the agent runs inside the rv sandbox as a system unit."""

    harness = "gui"
    network_guard = False  # optional network guard (mmcoding.host.guard), set by install()
    ask_enabled = False
    call_offset = 0  # model calls of earlier phases of this task (set by RvAgent.predict)

    # The official agent polls a *user* unit; the rv worker is a system unit.
    def command(self, command: str, timeout: float = 60, check: bool = True, *, read_only: bool = False) -> str:
        prefix = "systemctl --user is-active osworld-dsh-"
        if command.startswith(prefix):
            command = "systemctl is-active osworld-dsh-" + command[len(prefix):]
        return super().command(command, timeout, check, read_only=read_only)

    def ensure_tls_clock(self) -> dict:
        """Keep the guest OS clock at real time (TLS certificates); the task reference date stays in
        the prompt. Some task setup commands move the guest clock to the task date."""
        before = float(self.command("date +%s", read_only=True).strip())
        now = time.time()
        if abs(before - now) > 300:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now))
            self.command(_sudo("timedatectl set-ntp false"))
            self.command(_sudo("env TZ=UTC date -s " + shlex.quote(stamp)))
        after = float(self.command("date +%s", read_only=True).strip())
        if abs(after - time.time()) > 300:
            raise DeepSeekError("Guest TLS clock could not be synchronized", "environment")
        record = {"before_epoch": before, "after_epoch": after, "changed": abs(before - now) > 300,
                  "policy": "real OS clock for TLS; task reference date retained in prompt"}
        self.put(f"{RV_GUEST}/tls_clock.json", json.dumps(record))
        return record

    def setup_args(self, hidden: list[str]) -> str:
        """Arguments of ``rv-run setup`` for this guest's harness."""
        args = f"--harness {self.harness} --broker {BROKER.get(self.harness) or 'none'}"
        return args + "".join(f" --hidden {shlex.quote(p)}" for p in hidden)

    def deploy(self) -> str:
        root = GUEST_ROOT
        task = TASK_BY_HOST.get(self.host)
        hidden = TASK_HIDDEN_PATHS.get(task or "", [])
        self.command(
            f"umask 077; mkdir -p {root}/code {root}/attempts {RV_GUEST}/patches "
            f"{ASSETS_GUEST}/wheelhouse {ASSETS_GUEST}/debs"
        )
        for name in ("worker.py", "common.py", "helpers.py"):  # official, unchanged
            self.put(f"{root}/code/{name}", (OFFICIAL_DIR / name).read_bytes())
        for path in sorted(p for p in GUEST_DIR.iterdir() if p.is_file() and p.suffix in GUEST_SUFFIXES):
            self.put(f"{RV_GUEST}/{path.name}", path.read_bytes(), mode=0o700)
        for path in sorted(PATCH_DIR.glob("*.yml")):
            data = render_common_patch(self.harness).encode() if path.name == "common.patch.yml" else path.read_bytes()
            self.put(f"{RV_GUEST}/patches/{path.name}", data)
        for path in asset_files():
            sub = {".whl": "wheelhouse/", ".deb": "debs/"}.get(path.suffix, "")
            self.put(f"{ASSETS_GUEST}/{sub}{path.name}", path.read_bytes(), mode=0o644)
        self.command(f"chmod 755 {root} {RV_GUEST} {ASSETS_GUEST} {ASSETS_GUEST}/wheelhouse {ASSETS_GUEST}/debs")
        tls_clock = self.ensure_tls_clock()
        # uv + SDK venv: identical to the official deploy.
        self.command(
            "export PATH=/home/user/.local/bin:$PATH; "
            "if ! command -v uv >/dev/null; then "
            "set -o pipefail; "
            f"curl -fsSL --connect-timeout 20 --max-time 120 https://astral.sh/uv/{UV_VERSION}/install.sh "
            "| env UV_INSTALL_DIR=/home/user/.local/bin UV_NO_MODIFY_PATH=1 sh; fi",
            timeout=180,
        )
        self.command(
            "export PATH=/home/user/.local/bin:$PATH; "
            "command -v uv >/dev/null && "
            f"uv venv --allow-existing --python 3.12 {root}/venv && "
            f"uv pip install --python {root}/venv/bin/python "
            f"deepseek-harness-sdk=={SDK_VERSION}",
            timeout=900,
        )
        # BBR in the root namespace (rv-run setup sets it in the agent namespace).
        self.command(
            _sudo("sh -c 'modprobe tcp_bbr && sysctl -qw net.core.default_qdisc=fq "
                  f"net.ipv4.tcp_congestion_control={GUEST_TCP_CC}'"),
            timeout=60,
        )
        # Root helper: root-owned copy + a NOPASSWD rule for exactly that file.
        self.command(
            _sudo("sh -c 'install -o root -g root -m 0755 "
                  f"{RV_GUEST}/rv_run.py {RV_RUN} && "
                  f"printf \"user ALL=(root) NOPASSWD: {RV_RUN}\\n\" > /etc/sudoers.d/rv-run && "
                  "chmod 0440 /etc/sudoers.d/rv-run'"),
            timeout=60,
        )
        out = self.command(f"sudo -n {RV_RUN} setup {self.setup_args(hidden)}", timeout=1200)
        report = json.loads(out.strip().splitlines()[-1])
        report["task"] = task
        report["tls_clock"] = tls_clock
        report["network_guard"] = bool(self.network_guard)
        self.put(f"{EVIDENCE_GUEST}/setup.json", json.dumps(report, indent=1))
        if report.get("desktop_broker") != BROKER.get(self.harness):
            raise DeepSeekError("desktop broker not running", "environment")
        if report.get("network", "open") != "open":
            raise DeepSeekError("agent network policy not applied", "environment")
        if report.get("ns_congestion_control") != GUEST_TCP_CC:
            raise DeepSeekError("agent namespace is not on BBR", "environment")
        if self.network_guard:
            self.enable_network_guard(task)
        return root

    def enable_network_guard(self, task: str | None) -> None:
        """Redirect this case's traffic through the guard proxy until the agent is quiescent."""
        from mmcoding.host import guard

        try:
            self.guard_report = guard.harden(self, _sudo, task=task or "")
        except Exception as exc:
            raise DeepSeekError("network guard not active: " + type(exc).__name__, "environment") from exc
        self.put(f"{EVIDENCE_GUEST}/network_guard.json", json.dumps(self.guard_report, indent=1))

    def launch(self, attempt: str, config: dict) -> str:
        if not attempt.startswith(GUEST_ROOT + "/attempts/") or "/" in Path(attempt).name:
            raise ValueError("Invalid attempt path")
        config = {**config, "rv": {"ask_enabled": bool(self.ask_enabled), "call_offset": int(self.call_offset)}}
        endpoint = vision_endpoint()
        if endpoint and registry.get(config["settings"].get("harness", self.harness)).vision_tool:
            config["rv"]["vision_endpoint"] = endpoint
        quoted = shlex.quote(attempt)
        self.command(
            f"umask 077; mkdir -p {quoted}/questions {quoted}/answers {quoted}/services && "
            f"touch {quoted}/worker.stdout {quoted}/worker.stderr"
        )
        self.put(attempt + "/request.json", json.dumps(config))
        unit = "osworld-dsh-" + Path(attempt).name
        runtime_max = int(config["settings"]["timeout"]) + 90
        self.command(f"sudo -n {RV_RUN} worker --unit {shlex.quote(unit)} --attempt {quoted} "
                     f"--runtime-max {runtime_max}")
        return unit

    def stop(self, unit: str) -> None:
        if not unit.startswith("osworld-dsh-") or not unit.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Refusing to stop an unrelated service")
        self.command(f"sudo -n {RV_RUN} stop --unit {shlex.quote(unit)}", check=False, timeout=90)

    def stop_and_confirm(self, unit: str) -> dict:
        """Official quiescence proof, against the system manager."""
        for attempt in range(3):
            try:
                self.stop(unit)
                break
            except (EOFError, OSError, paramiko.SSHException, DeepSeekError):
                if attempt == 2:
                    raise
                self.close()
        raw = self.command(
            f"systemctl show {shlex.quote(unit)} -p ActiveState -p MainPID -p ControlPID -p ControlGroup",
            check=False,
            read_only=True,
        )
        state = dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)
        group = state.get("ControlGroup", "")
        populated = False
        if group:
            code = (
                "from pathlib import Path; p=Path('/sys/fs/cgroup') / "
                + repr(group.lstrip("/"))
                + "; print(p.exists() and any(f.read_text().strip() for f in p.rglob('cgroup.procs')))"
            )
            populated = self.command("python3 -c " + shlex.quote(code), read_only=True).strip() != "False"
        confirmed = (
            state.get("ActiveState") in {"inactive", "failed", ""}
            and state.get("MainPID", "0") == "0"
            and state.get("ControlPID", "0") == "0"
            and not populated
        )
        result = {**state, "cgroup_populated": populated, "confirmed": confirmed, "unit": unit}
        if not confirmed:
            raise DeepSeekError("SDK process group is not quiescent", "quiescence")
        if self.network_guard:  # task evaluation runs without the guard
            from mmcoding.host import guard

            guard.unharden(self)
        return result

    def collect(self, remote: str, local: Path, secrets: tuple[str, ...]) -> None:
        # The optional separate answerer key (vision_endpoint) is redacted like the policy key.
        extra = vision_endpoint().get("api_key")
        secrets = tuple(secrets) + ((extra,) if extra and extra not in secrets else ())
        super().collect(remote, local, secrets)
        self.collect_evidence(local.parent / "rv_evidence", secrets)

    def collect_evidence(self, local: Path, secrets: tuple[str, ...] = ()) -> None:
        self.command(f"sudo -n {RV_RUN} collect", timeout=300)
        local.mkdir(parents=True, exist_ok=True, mode=0o700)
        for name in self.files(EVIDENCE_GUEST):
            data = self.read(f"{EVIDENCE_GUEST}/{name}")
            if data is None:
                continue
            if name.endswith((".json", ".jsonl", ".log", ".txt")):
                data = scrub(data.decode(errors="replace"), secrets).encode()
            target = local / name
            if len(data) > 1_000_000:
                data = gzip.compress(data, compresslevel=5, mtime=0)
                target = target.with_name(target.name + ".gz")
            target.write_bytes(data)
            target.chmod(0o600)


class RvAgent(DeepSeekAgent):
    """Official agent + simulator flag and per-task cumulative call budget."""

    def predict(self, instruction: str, obs=None):
        self.guest.ask_enabled = getattr(self.env, "user_simulator", None) is not None
        offset = int(getattr(self, "task_calls", 0))
        self.guest.call_offset = offset
        try:
            return super().predict(instruction, obs)
        finally:
            self.task_calls = self._calls_after(offset)

    def _calls_after(self, offset: int) -> int:
        """Cumulative task model calls from the collected ``budget.json`` (global count)."""
        try:
            budget = json.loads((Path(self.last_output) / "dsh" / "budget.json").read_text())
            calls = int(budget["model_calls"])
        except (OSError, ValueError, KeyError, TypeError):
            return offset
        return max(offset, calls)


# ------------------------------------------------------------------------- docker


def authorize_ssh_key(controller: Any, public_key: str) -> None:
    """Add our key to ``user``'s authorized_keys through the guest control server.

    OSWorld 2.0's server runs it through ``/run_bash_script``; images whose copy of that endpoint
    fails fall back to ``/execute``, which every image generation provides.
    """
    key = public_key.strip()
    if not key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-")) or "'" in key:
        raise ValueError("unexpected SSH public key format")
    script = (
        "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; "
        f"grep -qxF '{key}' ~/.ssh/authorized_keys || echo '{key}' >> ~/.ssh/authorized_keys; echo ok"
    )
    result = controller.run_bash_script(script, timeout=30)
    if result and "ok" in str(result.get("output", "")):
        return
    import requests

    try:
        reply = requests.post(f"{controller.http_server}/execute",
                              json={"command": ["bash", "-c", script], "shell": False}, timeout=60).json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f"could not authorize the SSH key in the guest: {exc}") from exc
    if "ok" not in str(reply.get("output", "")):
        raise RuntimeError("could not authorize the SSH key in the guest")


def prepare_package_manager(guest) -> None:
    """Quiet the image's automatic updates before the official task setup installs packages."""
    guest.command(_sudo("systemctl mask --runtime --now packagekit.service apt-daily.timer apt-daily-upgrade.timer"), timeout=120)
    # Never remove lock files or kill an apt transaction: wait until the POSIX locks are free.
    code = '''import fcntl, os, time
files=['/var/lib/apt/lists/lock','/var/lib/dpkg/lock','/var/lib/dpkg/lock-frontend']
deadline=time.monotonic()+180
while True:
 handles=[]
 try:
  for path in files:
   if not os.path.exists(path): continue
   handle=open(path,'r+'); handles.append(handle)
   fcntl.lockf(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  print('package locks released'); break
 except BlockingIOError:
  if time.monotonic()>=deadline: raise TimeoutError('package manager busy before task setup')
  time.sleep(1)
 finally:
  for handle in handles: handle.close()
'''
    guest.command(_sudo("python3 -c " + shlex.quote(code)), timeout=210)


def rebind_task_setup_controller(env, ip: str, task: str) -> None:
    """OSWorld 2.0 task 082 probes its app at ``setup_controller.vm_ip:3000`` from the host."""
    if task != "082":
        return
    controller = env.setup_controller
    controller.vm_ip = ip
    controller.server_port = CONTAINER_PORTS["server_port"]
    controller.chromium_port = CONTAINER_PORTS["chromium_port"]
    controller.vlc_port = CONTAINER_PORTS["vlc_port"]
    controller.http_server = f"http://{ip}:{controller.server_port}"
    controller.http_server_setup_root = controller.http_server + "/setup"


def install_website_dns(ip: str, ssh_key: str, web_config) -> None:
    """Install guest/website_dns.py and its configuration (used by ``sites`` for GitLab names)."""
    guest = Guest(ip, ssh_key)
    try:
        guest.connect()
        guest.put("/tmp/rv-web-dns.py", (GUEST_DIR / "website_dns.py").read_bytes())
        guest.command(_sudo(f"install -m 0755 /tmp/rv-web-dns.py {sites.DNS_SCRIPT}"))
        dns_config = {"hostnames": list(web_config.hostnames),
                      "gitlab_domain": web_config.gitlab_domain, "forwarders": ["127.0.0.53"]}
        guest.put("/tmp/rv-web-dns.json", json.dumps(dns_config))
        guest.command(_sudo("install -m 0644 /tmp/rv-web-dns.json /usr/local/lib/rv-web-dns.json"))
    finally:
        guest.close()


def docker_new_env(args, example=None):
    """Docker counterpart of the official ``new_env`` (which targets AWS)."""
    from desktop_env.desktop_env import DesktopEnv

    volume = max(int(task_value(example, "volume_size", 0) or 0), int(args.volume_size))
    env = DesktopEnv(
        action_space="pyautogui",
        provider_name="docker",
        path_to_vm=getattr(args, "path_to_vm", None),
        screen_size=(1920, 1080),
        headless=True,
        os_type="Ubuntu",
        require_a11y_tree=False,
        enable_proxy=False,
        client_password=CLIENT_PASSWORD,
        volume_size=volume,
        force_disable_vnc=True,
        force_disable_recording=True,
        cache_dir=str(args.result_dir / "cache" / task_value(example, "id", "smoke")),
    )
    container = env.provider.container
    container.reload()
    ip = next(iter(container.attrs["NetworkSettings"]["Networks"].values()))["IPAddress"]
    authorize_ssh_key(env.controller, Path(args.ssh_private_key + ".pub").read_text())
    env.vm_ip = ip
    for attr, port in CONTAINER_PORTS.items():
        setattr(env, attr, port)
    env.rv_container = container.name
    task_id = str(task_value(example, "id", "smoke"))
    output = Path(args.result_dir) / task_id
    if output.is_dir():  # lets the dispatcher remove a VM the official runner retained
        write_json(output / "container.json", {"container": container.name, "ip": ip,
                                               "resources": dict(env.provider.environment)})
    rebind_task_setup_controller(env, ip, str(task_value(example, "id", "")))
    try:
        package_guest = Guest(ip, args.ssh_private_key)
        try:
            package_guest.connect()
            prepare_package_manager(package_guest)
            if RvGuest.network_guard:  # trust the guard CA before task setup starts browsers
                from mmcoding.host import guard

                guard.install_ca(package_guest, _sudo)
        finally:
            package_guest.close()
        web_config = sites.load_config()
        if web_config is not None:
            install_website_dns(ip, args.ssh_private_key, web_config)
        env.rv_website_tunnel = sites.attach(env, args.ssh_private_key, CLIENT_PASSWORD)
    except BaseException:
        env.close()
        if output.is_dir():
            write_json(output / "cleanup.json", {"container": container.name, "state": "removed",
                                                 "phase": "provision_failed_before_agent"})
        raise
    TASK_BY_HOST[ip] = task_id
    return env


def docker_terminate_case(env, args, output) -> None:
    name = getattr(env, "rv_container", None)
    try:
        env.close()
    finally:
        tunnel = getattr(env, "rv_website_tunnel", None)
        if tunnel is not None:
            tunnel.close()
    write_json(output / "cleanup.json", {"container": name, "state": "removed"})


# ------------------------------------------------------------------------ install


def configure_docker_vm(ram: str = "8G", cores: str = "4") -> None:
    """Set the Docker provider's guest RAM / CPU cores (identical for every harness).

    The provider's default of 4 GB is too small for code that substitutes for perception (large
    images, local OCR / ASR models) and gets it OOM-killed.
    """
    from desktop_env.providers.docker.provider import DockerProvider

    original = getattr(DockerProvider.__init__, "_rv_original", DockerProvider.__init__)

    def patched(self, *a, **kw):
        original(self, *a, **kw)
        self.environment["RAM_SIZE"] = str(ram)
        self.environment["CPU_CORES"] = str(cores)

    patched._rv_original = original
    DockerProvider.__init__ = patched


def configure_harness(harness: str, base_url: str | None = None) -> None:
    """Bind one harness into the agent prompt and the guest adapter."""
    harness = registry.normalize(harness)
    if base_url is not None and urlparse(base_url).scheme != "https":
        raise ValueError("the model endpoint must use https://")
    official_agent.runtime_prompt = functools.partial(rv_runtime_prompt, harness=harness)
    RvGuest.harness = harness


def install_judge_dimension_transport() -> None:
    """Keep full-frame judge evidence within the judge endpoint's 8000-pixel image dimension limit.

    Images above 8000 px on their longest side are scaled down proportionally (never cropped); then
    every image is scaled down further until the request fits the byte bound. Originals and prompts
    are preserved next to the request.
    """
    from io import BytesIO
    import base64
    from PIL import Image
    from hybrid_agents.deepseek import judge_images

    original = getattr(judge_images.bound_request, "_rv_dimension_original", judge_images.bound_request)

    def bounded(kwargs, paths, output, **limits):
        from copy import deepcopy
        target = Path(output)
        max_bytes = limits.get("max_bytes", 8_000_000)
        max_width = limits.get("max_width", 1920)
        if max_bytes <= 0 or max_width <= 0:
            raise ValueError("judge transport limits must be positive")
        payload = deepcopy(kwargs)
        nodes = [item["image_url"] for message in payload["messages"]
                 if isinstance(message.get("content"), list) for item in message["content"]
                 if item.get("type") == "image_url"]
        if len(nodes) != len(paths):
            raise ValueError("judge image count/order mismatch")
        records, sources = [], []
        original_bytes = len(judge_images.json_bytes(payload))
        for index, path in enumerate(paths):
            data = Path(path).read_bytes()
            with Image.open(BytesIO(data)) as image:
                size = list(image.size)
            saved = target / "originals" / f"image-{index}{Path(path).suffix}"
            judge_images.private_write(saved, data)
            sources.append(data)
            records.append({"source_path": str(path), "original_path": str(saved),
                            "original_sha256": hashlib.sha256(data).hexdigest(),
                            "original_bytes": len(data), "original_dimensions": size,
                            "sent_path": str(saved), "sent_sha256": hashlib.sha256(data).hexdigest(),
                            "sent_dimensions": size})

        def represent(index, scale, label):
            data = sources[index]
            with Image.open(BytesIO(data)) as image:
                size = tuple(max(1, int(side * scale)) for side in image.size)
                resized = image.resize(size, Image.Resampling.LANCZOS)
                buffer = BytesIO()
                resized.save(buffer, format="PNG")
                encoded = buffer.getvalue()
                resized.close()
            saved = target / "representations" / f"{label}-image-{index}.png"
            judge_images.private_write(saved, encoded)
            records[index].update(sent_path=str(saved), sent_sha256=hashlib.sha256(encoded).hexdigest(),
                                  sent_dimensions=list(size))
            nodes[index]["url"] = "data:image/png;base64," + base64.b64encode(encoded).decode()

        # Dimension bound first, then converge on both bounds: a tall JPEG may grow as PNG, and
        # narrow images must not bypass the byte reduction.
        for index, record in enumerate(records):
            largest = max(record["original_dimensions"])
            if largest > 8000:
                represent(index, 8000 / largest, "max-dimension-8000")
        width = max_width
        round_index = 0
        while len(judge_images.json_bytes(payload)) > max_bytes:
            if width < 480:
                raise ValueError("Cannot fit judge images without excessive downscaling")
            for index, record in enumerate(records):
                w, h = record["original_dimensions"]
                if max(record["sent_dimensions"]) <= width:
                    continue
                represent(index, min(1., width / w, width / h), f"byte-round-{round_index}-{width}")
            width = int(width * .75)
            round_index += 1
        body = judge_images.json_bytes(payload)
        judge_images.private_write(target / "request.json", body)
        judge_images.private_write(target / "images.json", judge_images.json_bytes({
            "original_request_bytes": original_bytes, "sent_request_bytes": len(body),
            "max_request_bytes": max_bytes, "initial_max_width": max_width,
            "max_image_dimension": 8000, "crop": False, "images": records,
            "dimension_repair": "proportional full frame, no crop; originals and prompts preserved"}))
        return payload

    bounded._rv_dimension_original = original
    judge_images.bound_request = bounded


def install_evaluator_timeout_transport() -> None:
    """Do not cut long official evaluator commands at the I/O layer's 90 s default."""
    from hybrid_agents.deepseek.io_recovery import RecoveryRequests

    original = getattr(RecoveryRequests.request, "_rv_timeout_original", RecoveryRequests.request)

    def request(self, method, url, **kwargs):
        if not self.setup and method.upper() == "POST" and urlparse(url).path == "/setup/execute":
            payload = kwargs.get("json")
            if payload is None and kwargs.get("data") is not None:
                try:
                    payload = json.loads(kwargs["data"])
                except (ValueError, TypeError):
                    payload = None
            command_timeout = payload.get("timeout") if isinstance(payload, dict) else None
            if isinstance(command_timeout, (int, float)) and 0 < command_timeout <= 3600:
                # The command keeps its official deadline; this is slack for the response transport.
                current = kwargs.get("timeout")
                read = current[1] if isinstance(current, tuple) else current
                kwargs["timeout"] = (10, max(command_timeout + 20, read or 0))
        return original(self, method, url, **kwargs)

    request._rv_timeout_original = original
    RecoveryRequests.request = request


@contextmanager
def phase_evidence_adapters(example, agent, guest, output, setup_slot=None, evaluation_slot=None,
                            secrets=()):
    """Bind the official phase wrappers to the actual method, not BaseTask's dict.

    BaseTask.__setattr__ stores ordinary names as mapping entries. An inherited get_phases method
    wins lookup over those entries, so the upstream temporary assignment never invokes its
    evidence/transport wrappers. Keep the exact official setup/evaluator callbacks while installing
    a real instance override.
    """
    from contextlib import nullcontext
    from hybrid_agents.deepseek import phases as official_phases

    original = getattr(example, "get_phases", None)
    if not callable(original):
        yield
        return
    agent.phase_mode = True
    namespace = vars(example)
    had_instance_method = "get_phases" in namespace
    previous_instance_method = namespace.get("get_phases")

    def get_phases():
        result = []
        for index, entry in enumerate(original(), 1):
            phase = dict(entry)
            setup, evaluate = phase["setup"], phase["evaluate"]

            def bounded_setup(controller, *args, _setup=setup, _index=index, **kwargs):
                with setup_slot or nullcontext():
                    with (official_phases.setup_deadline(), official_phases.setup_transport(
                            controller, guest, output / "phases" / f"{_index:02d}" / "setup_commands", secrets)):
                        return _setup(controller, *args, **kwargs)

            def bounded_evaluate(env, _evaluate=evaluate):
                official_phases.screenshot(guest, agent.last_output / "final_screen.png")
                with evaluation_slot or nullcontext():
                    return _evaluate(env)

            phase.update(setup=bounded_setup, evaluate=bounded_evaluate)
            result.append(phase)
        return result

    object.__setattr__(example, "get_phases", get_phases)
    try:
        yield
    finally:
        if had_instance_method:
            object.__setattr__(example, "get_phases", previous_instance_method)
        else:
            object.__delattr__(example, "get_phases")


def install(runner_module: Any, harness: str, *, vm_ram: str = "8G", vm_cores: str = "4",
            base_url: str | None = None, network_guard: bool = False) -> None:
    """Bind the rv adapters into the official runner module and agent prompt."""
    configure_docker_vm(vm_ram, vm_cores)
    configure_harness(harness, base_url)
    RvGuest.network_guard = bool(network_guard)
    install_evaluator_timeout_transport()
    install_judge_dimension_transport()
    runner_module.new_env = docker_new_env
    runner_module.terminate_case = docker_terminate_case
    runner_module.Guest = RvGuest
    runner_module.DeepSeekAgent = RvAgent
    runner_module.phase_adapters = phase_evidence_adapters


__all__ = [
    "BROKER",
    "CAPABILITIES",
    "CLIENT_PASSWORD",
    "HARNESSES",
    "TASK_HIDDEN_PATHS",
    "X_HARNESSES",
    "RvAgent",
    "RvGuest",
    "RvSettings",
    "asset_files",
    "capability",
    "configure_docker_vm",
    "configure_harness",
    "docker_new_env",
    "docker_terminate_case",
    "install",
    "load_models",
    "phase_evidence_adapters",
    "render_common_patch",
    "rv_content_hash",
    "rv_files",
    "rv_runtime_prompt",
    "vision_endpoint",
]
