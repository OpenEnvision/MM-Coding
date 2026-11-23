"""Offline tests of the host side (mmcoding.host): prompt, settings, patch rendering, deploy and
launch against a fake guest, Docker provisioning pieces, transports, audit, network guard and the
OSWorld 2.0 website bridge. No VM, container, network or model is used."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import shlex
import socket
import sys
import time
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests  # noqa: F401  (loaded before no_network replaces http.client.HTTPConnection)
import yaml

from mmcoding import harnesses
from mmcoding.host import audit, deploy
from mmcoding.host import sites as web

REPO = Path(__file__).resolve().parents[1]

# sha256 of rv_runtime_prompt for two input sets per harness, computed once with the implementation
# that produced the paper's runs: the prompt each harness sees is byte-identical to those runs.
PROMPT_INPUTS = (("Open {with_x} and {task_date}.", None, False), ("Do X.\nThen Y.", "2024-03-15", True))
PROMPT_SHA256 = {
    "gui": ("2031aa20d4778840a3c3b2fd0a5a22fd0b5a9779d3523ba825d337b1df118daf",
            "6f0f28e38a7c970aba0f92c441c58f6e3e9229561973d63a99989e85b3c0465c"),
    "os_helix": ("8c1544db7c6f4cbe5a12f107b39983ca3ab056af7def2468a09dc6b0ee7d2cf9",
                 "d81fd7be2153a8238818f01ea1cf02a02cbd895054c57ecb22fccadd2098b80b"),
    "mm_coding": ("b4a8dc4abab3786d1b40b7530c88042f6da69da0050f75f359f6b0db72a3bc70",
                  "72475da176b9151783e27c8744ce731184cbebc6871936977cdc5bcd7c7b7889"),
    "vision_tool": ("b06608905c5cf5cbb9b213dab4910c4c3dff5d1070ec085218576082ba4238ce",
                    "1799c3dfaa3fa9786a4cf70fd6e2627d44afc02d10546fa25769b29f1720820b"),
    "cli": ("ffe814becc808f78c9826cb0b2be610051b1b49ee513484e3fdfcedfc16263a1",
            "54ad838f675599cb1efa92ec1ed2ad0ee361795a334196d30bda6538be39eedf"),
}
SETTINGS_FIELDS = {"model", "base_url", "reasoning_effort", "max_tokens", "timeout", "stream_idle_timeout",
                   "harness", "provider", "call_budget", "vision_model", "vision_reasoning_effort",
                   "vision_coords", "vision_thinking"}


def settings(**kw):
    values = {"model": "model-a", "base_url": "https://x.example/v1", "reasoning_effort": "high",
              "max_tokens": 1024, "timeout": 100}
    return deploy.RvSettings(**{**values, **kw})


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real network access is forbidden")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(web.http.client, "HTTPConnection", forbidden)


# ------------------------------------------------------------------ prompt


@pytest.mark.parametrize("harness", harnesses.HARNESSES)
def test_prompt_is_byte_identical_to_the_paper_runs(harness):
    for (instruction, date, ask), expected in zip(PROMPT_INPUTS, PROMPT_SHA256[harness]):
        prompt = deploy.rv_runtime_prompt(instruction, "/remote", date, ask, harness=harness)
        assert hashlib.sha256(prompt.encode()).hexdigest() == expected


def test_prompt_differs_only_in_capability_paragraph():
    prompts = {h: deploy.rv_runtime_prompt("Do X.", "/remote", "2024-01-02", False, harness=h)
               for h in harnesses.HARNESSES}
    stripped = {h: p.replace(deploy.CAPABILITIES[h], "<CAP>") for h, p in prompts.items()}
    assert len(set(stripped.values())) == 1
    assert prompts["gui"].endswith("TASK INSTRUCTION (authoritative, verbatim):\nDo X.")
    assert all("report_infeasible" in p for p in prompts.values())
    assert "Task reference date: 2024-01-02." in prompts["cli"]
    assert "mcp__ask__ask_user" not in prompts["cli"]
    assert "mcp__ask__ask_user" in deploy.rv_runtime_prompt("x", "", None, True, harness="cli")


def test_prompt_accepts_paper_names_and_keeps_instruction_verbatim():
    text = "Use {serve_note} and {capability} literally."
    prompt = deploy.rv_runtime_prompt(text, "", None, False, harness="MM-Coding")
    assert prompt == deploy.rv_runtime_prompt(text, "", None, False, harness="mm_coding")
    assert prompt.endswith("\n" + text)


def test_capabilities_describe_the_tools():
    caps = deploy.CAPABILITIES
    assert set(caps) == set(harnesses.HARNESSES)
    for text in caps.values():
        assert "prefer" not in text.lower() and "faster" not in text.lower()
        assert "{" not in text and "}" not in text
    for harness in harnesses.X_HARNESSES - {"os_helix"}:
        assert "can use the graphical display" in caps[harness] and "rv-serve" in caps[harness]
    assert "no shell, file or code-execution tools" in caps["gui"] and "terminal windows" in caps["gui"]
    assert "mcp__helix__run_code" in caps["os_helix"] and caps["os_helix"].endswith("parallel. ")
    assert "mcp__desktop__screenshot" in caps["mm_coding"] and "read_image" in caps["mm_coding"]
    vt = caps["vision_tool"]
    assert vt.startswith("You cannot view images yourself") and "mcp__vision__ask_vision" in vt
    assert "read_image" not in vt and "screenshot tool" not in vt
    assert "cannot view images" in caps["cli"] and "read_image" not in caps["cli"]
    assert deploy.prompt_text("with_x.txt") in caps["cli"]


def test_every_harness_has_its_prompt_and_profile_files():
    for harness in harnesses.HARNESSES:
        assert (deploy.PROMPT_DIR / "capability" / f"{harness}.txt").is_file()
        assert (deploy.PATCH_DIR / f"{harness}.patch.yml").is_file()
    assert (deploy.PATCH_DIR / "ask_user.patch.yml").is_file()


# ---------------------------------------------------------------- settings


def test_settings_validation_and_public_fields():
    s = settings(harness="cli", call_budget=5)
    pub = s.public()
    assert pub["harness"] == "cli" and pub["call_budget"] == 5 and pub["provider"] == "openai"
    assert pub["mcp_servers"] == ["report", "ask (simulator tasks only)"]
    assert pub["patches"][:2] == ["common.patch.yml", "cli.patch.yml"] and pub["vision_answerer"] is None
    assert pub["network_guard"] is False and "reachable" in pub["sandbox"]
    assert {f.name for f in dataclasses.fields(deploy.RvSettings)} == SETTINGS_FIELDS
    defaults = settings()
    assert (defaults.harness, defaults.provider, defaults.call_budget) == ("gui", "openai", 500)
    assert "hidden" in defaults.public()["sandbox"]
    mm = settings(harness="mm_coding").public()
    assert mm["mcp_servers"][0].startswith("desktop (rv_desktop.py --profile observe ")
    helix = settings(harness="os_helix").public()
    assert helix["mcp_servers"][0].startswith("desktop (rv_desktop.py --profile act_guarded ")
    assert any(m.startswith("helix (") for m in helix["mcp_servers"])
    for bad in ({"harness": "desktop"}, {"harness": "MM-Coding"}, {"harness": "cli", "call_budget": 0},
                {"reasoning_effort": "ultra"}, {"vision_coords": "abs"}, {"vision_thinking": "yes"},
                {"vision_reasoning_effort": "ultra"}):
        with pytest.raises(ValueError):
            settings(**bad)
    with pytest.raises(ValueError):
        deploy.RvSettings("model-a", "http://x.example/v1", "high", 1024, 100)


def test_vision_answerer_settings(monkeypatch):
    monkeypatch.delenv("RV_VISION_BASE_URL", raising=False)
    pub = settings(harness="vision_tool", vision_model="vision-b", vision_reasoning_effort="off",
                   vision_coords="norm1000", vision_thinking="off").public()
    assert pub["vision_answerer"] == {"model": "vision-b", "reasoning_effort": "off", "coords": "norm1000",
                                      "thinking": "off", "state": "stateless"}
    assert pub["mcp_servers"][0].startswith("vision (") and "reachable" in pub["sandbox"]
    assert settings(harness="vision_tool").public()["vision_answerer"]["model"] == "model-a"
    monkeypatch.setenv("RV_VISION_BASE_URL", "https://vision.example/v1")
    assert settings(harness="vision_tool").public()["vision_answerer"]["endpoint"] == "https://vision.example/v1"


def test_vision_endpoint_from_environment(monkeypatch):
    monkeypatch.delenv("RV_VISION_BASE_URL", raising=False)
    assert deploy.vision_endpoint() == {}
    monkeypatch.setenv("RV_VISION_BASE_URL", "https://vision.example/v1")
    monkeypatch.delenv("RV_VISION_API_KEY", raising=False)
    with pytest.raises(ValueError):
        deploy.vision_endpoint()
    monkeypatch.setenv("RV_VISION_API_KEY", "vk")
    assert deploy.vision_endpoint() == {"base_url": "https://vision.example/v1", "api_key": "vk"}
    monkeypatch.setenv("RV_VISION_BASE_URL", "http://vision.example/v1")
    with pytest.raises(ValueError):
        deploy.vision_endpoint()


def test_configure_harness_binds_prompt_and_guest(monkeypatch):
    monkeypatch.setattr(deploy.official_agent, "runtime_prompt", deploy.official_agent.runtime_prompt)
    monkeypatch.setattr(deploy.RvGuest, "harness", "gui")
    deploy.configure_harness("Vision-Tool", "https://x.example/v1")
    assert deploy.RvGuest.harness == "vision_tool"
    assert deploy.official_agent.runtime_prompt("Do X.", "", None, False) == \
        deploy.rv_runtime_prompt("Do X.", "", None, False, harness="vision_tool")
    with pytest.raises(KeyError):
        deploy.configure_harness("vision")
    with pytest.raises(ValueError):
        deploy.configure_harness("cli", "http://x.example/v1")


def test_install_binds_adapters_and_guard_flag(monkeypatch):
    for name in ("configure_docker_vm", "install_evaluator_timeout_transport", "install_judge_dimension_transport"):
        monkeypatch.setattr(deploy, name, lambda *a, **k: None)
    monkeypatch.setattr(deploy.official_agent, "runtime_prompt", deploy.official_agent.runtime_prompt)
    monkeypatch.setattr(deploy.RvGuest, "harness", "gui")
    monkeypatch.setattr(deploy.RvGuest, "network_guard", False)
    runner = SimpleNamespace()
    deploy.install(runner, "os_helix")
    assert deploy.RvGuest.network_guard is False and deploy.RvGuest.harness == "os_helix"
    assert runner.Guest is deploy.RvGuest and runner.DeepSeekAgent is deploy.RvAgent
    assert runner.new_env is deploy.docker_new_env and runner.terminate_case is deploy.docker_terminate_case
    assert runner.phase_adapters is deploy.phase_evidence_adapters
    deploy.install(runner, "cli", network_guard=True)
    assert deploy.RvGuest.network_guard is True


# ------------------------------------------------------- common patch rendering


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:js", lambda loader, node: loader.construct_scalar(node))


def _provider(text: str) -> dict:
    patch = yaml.load(text, Loader=_Loader)
    return next(p for p in patch if p.get("id") == "llm-pi-ai")["config"]["providers"]["openai"]


@pytest.mark.parametrize("harness", harnesses.HARNESSES)
def test_common_patch_rendering(harness):
    text = deploy.render_common_patch(harness)
    assert "__RV_MODELS__" not in text and "__RV_MAX_REQUEST_IMAGE_BYTES__" not in text
    provider = _provider(text)
    assert provider["maxRequestImageBytes"] == harnesses.get(harness).max_request_image_bytes
    assert provider["models"] == deploy.load_models()
    assert provider["displayName"] == "OpenAI-compatible endpoint"
    line = next(l for l in text.splitlines() if l.startswith("        models: "))
    assert json.loads(line.split(": ", 1)[1]) == deploy.load_models()  # one JSON flow line


def test_common_patch_placeholders_must_appear_once():
    template = (deploy.PATCH_DIR / "common.patch.yml").read_text()
    for broken in (template.replace(deploy.MODELS_LINE, ""), template + "\n" + deploy.IMAGE_BYTES_LINE + "\n"):
        with pytest.raises(ValueError):
            deploy.render_common_patch("gui", template=broken)
    rendered = deploy.render_common_patch("gui", models=[{"id": "m", "maxTokens": 8}], template=template)
    assert _provider(rendered)["models"] == [{"id": "m", "maxTokens": 8}]


def test_models_file_choice_and_validation(tmp_path, monkeypatch):
    example = deploy.load_models(deploy.MODELS_EXAMPLE)
    assert len(example) == 4 and len({m["id"] for m in example}) == 4
    assert all(m["contextWindow"] == 262144 for m in example)
    own = tmp_path / "models.yaml"
    monkeypatch.setattr(deploy, "MODELS_FILE", own)
    assert deploy.models_file() == deploy.MODELS_EXAMPLE
    own.write_text("models:\n  - id: mine\n    maxTokens: 10\n")
    assert deploy.models_file() == own and deploy.load_models() == [{"id": "mine", "maxTokens": 10}]
    for bad in ("models: []\n", "- id: a\n", "models:\n  - id: a\n  - id: a\n", "models:\n  - maxTokens: 1\n"):
        own.write_text(bad)
        with pytest.raises(ValueError):
            deploy.load_models()


def test_provenance_covers_prompts_and_profiles():
    names = {p.name for p in deploy.rv_files()}
    assert {"deploy.py", "harnesses.py", "common.patch.yml", "closing.txt", "os_helix.txt"} <= names
    assert len(deploy.rv_content_hash()) == 64
    lists = [p for p in deploy.asset_files() if p.suffix == ".txt"]
    assert [p.name for p in lists] == ["requirements.txt", "constraints.txt"]
    assert all(p.parent == deploy.SANDBOX_LISTS for p in lists)


# --------------------------------------------------------------- fake guest


class Recorder:
    """Replaces the SSH transport of ``Guest``: records commands and uploads."""

    def __init__(self, harness, broker=None, network="open"):
        self.commands, self.files = [], {}
        reported = deploy.BROKER.get(harness) if broker is None else broker
        self.report = {"desktop_broker": reported, "ns_congestion_control": "bbr", "network": network}

    def command(self, guest, command, timeout=60, check=True, read_only=False):
        self.commands.append(command)
        if command == "date +%s":
            return str(int(time.time()))
        if command.startswith(f"sudo -n {deploy.RV_RUN} setup "):
            return "setup log\n" + json.dumps(self.report)
        return ""

    def put(self, guest, path, content, mode=0o600):
        self.files[path] = content


@pytest.fixture
def fake_guest(monkeypatch):
    def make(harness, **kw):
        rec = Recorder(harness, **kw)
        monkeypatch.setattr(deploy.Guest, "command", lambda self, *a, **k: rec.command(self, *a, **k))
        monkeypatch.setattr(deploy.Guest, "put", lambda self, *a, **k: rec.put(self, *a, **k))
        monkeypatch.setattr(deploy, "asset_files", lambda: [])
        monkeypatch.setattr(deploy.RvGuest, "harness", harness)
        monkeypatch.setattr(deploy.RvGuest, "network_guard", False)
        return deploy.RvGuest("10.0.0.7", "/key"), rec
    return make


@pytest.fixture
def no_guard_import(monkeypatch):
    """Any use of the guard module fails the test."""
    import mmcoding.host

    class Forbidden(types.ModuleType):
        def __getattr__(self, name):
            raise AssertionError(f"guard.{name} used while the network guard is off")

    module = Forbidden("mmcoding.host.guard")
    monkeypatch.setitem(sys.modules, "mmcoding.host.guard", module)
    monkeypatch.setattr(mmcoding.host, "guard", module, raising=False)


def test_guest_command_rewrites_user_unit_poll(fake_guest):
    guest, rec = fake_guest("gui")
    guest.command("systemctl --user is-active osworld-dsh-task-abc")
    guest.command("systemctl --user is-active other")
    assert rec.commands == ["systemctl is-active osworld-dsh-task-abc", "systemctl --user is-active other"]


@pytest.mark.parametrize("harness,expected", [
    ("gui", "--harness gui --broker act"), ("os_helix", "--harness os_helix --broker act_guarded"),
    ("mm_coding", "--harness mm_coding --broker observe"), ("vision_tool", "--harness vision_tool --broker none"),
    ("cli", "--harness cli --broker none"),
])
def test_deploy_command_sequence(fake_guest, no_guard_import, monkeypatch, harness, expected):
    guest, rec = fake_guest(harness)
    monkeypatch.setitem(deploy.TASK_BY_HOST, "10.0.0.7", "042")
    monkeypatch.setitem(deploy.TASK_HIDDEN_PATHS, "042", ["/home/user/eval data"])
    assert guest.deploy() == deploy.GUEST_ROOT
    setup = [c for c in rec.commands if " setup " in c]
    assert setup == [f"sudo -n {deploy.RV_RUN} setup {expected} --hidden '/home/user/eval data'"]
    order = [next(i for i, c in enumerate(rec.commands) if key in c)
             for key in ("mkdir -p", "date +%s", "install.sh", "deepseek-harness-sdk==", "tcp_bbr",
                         "/etc/sudoers.d/rv-run", " setup ")]
    assert order == sorted(order)
    patches = {Path(p).name: v for p, v in rec.files.items() if p.startswith(deploy.RV_GUEST + "/patches/")}
    assert set(patches) == {p.name for p in deploy.PATCH_DIR.glob("*.yml")}
    assert patches["common.patch.yml"].decode() == deploy.render_common_patch(harness)
    assert f"{deploy.RV_GUEST}/rv_run.py" in rec.files
    assert {f"{deploy.GUEST_ROOT}/code/{n}" for n in ("worker.py", "common.py", "helpers.py")} <= set(rec.files)
    report = json.loads(rec.files[f"{deploy.EVIDENCE_GUEST}/setup.json"])
    assert report["task"] == "042" and report["network_guard"] is False and "tls_clock" in report
    assert f"{deploy.RV_GUEST}/tls_clock.json" in rec.files


@pytest.mark.parametrize("override", [{"broker": "act"}, {"broker": None}, {"network": "restricted"}])
def test_deploy_fails_closed_on_setup_report(fake_guest, no_guard_import, override):
    guest, rec = fake_guest("mm_coding", **override)
    if "broker" in override and override["broker"] is None:
        rec.report["desktop_broker"] = None
    with pytest.raises(deploy.DeepSeekError):
        guest.deploy()


def test_deploy_with_network_guard_calls_harden(fake_guest, monkeypatch):
    import mmcoding.host

    calls = []
    fake = types.SimpleNamespace(harden=lambda guest, sudo, task="": calls.append(("harden", task)) or {"ok": 1},
                                 unharden=lambda guest: calls.append(("unharden",)))
    monkeypatch.setitem(sys.modules, "mmcoding.host.guard", fake)
    monkeypatch.setattr(mmcoding.host, "guard", fake, raising=False)
    guest, rec = fake_guest("cli")
    monkeypatch.setattr(deploy.RvGuest, "network_guard", True)
    guest.deploy()
    assert calls == [("harden", "")]
    assert json.loads(rec.files[f"{deploy.EVIDENCE_GUEST}/network_guard.json"]) == {"ok": 1}

    def failing(guest, sudo, task=""):
        raise RuntimeError("proxy down")
    fake.harden = failing
    with pytest.raises(deploy.DeepSeekError, match="network guard not active"):
        guest.deploy()


def _launch_config(harness):
    s = settings(harness=harness, vision_model="vision-b")
    return {"attempt": deploy.GUEST_ROOT + "/attempts/task-1", "cwd": "/home/user", "api_key": "k",
            "settings": s.public(), "prompt": "p"}


@pytest.mark.parametrize("harness", harnesses.HARNESSES)
def test_launch_request_schema(fake_guest, monkeypatch, harness):
    monkeypatch.setenv("RV_VISION_BASE_URL", "https://vision.example/v1")
    monkeypatch.setenv("RV_VISION_API_KEY", "vk")
    guest, rec = fake_guest(harness)
    guest.ask_enabled, guest.call_offset = True, 7
    attempt = deploy.GUEST_ROOT + "/attempts/task-1"
    assert guest.launch(attempt, _launch_config(harness)) == "osworld-dsh-task-1"
    request = json.loads(rec.files[attempt + "/request.json"])
    expected = {"ask_enabled": True, "call_offset": 7}
    if harness == "vision_tool":
        expected["vision_endpoint"] = {"base_url": "https://vision.example/v1", "api_key": "vk"}
    assert request["rv"] == expected
    s = request["settings"]
    assert s["harness"] == harness and s["provider"] == "openai" and s["call_budget"] == 500
    assert {"vision_model", "vision_reasoning_effort", "vision_coords", "vision_thinking"} <= set(s)
    assert SETTINGS_FIELDS <= set(s)
    worker = rec.commands[-1]
    assert worker.startswith(f"sudo -n {deploy.RV_RUN} worker --unit osworld-dsh-task-1 ")
    assert worker.endswith(f"--runtime-max {100 + 90}")
    with pytest.raises(ValueError):
        guest.launch("/tmp/elsewhere", _launch_config(harness))


def test_launch_without_vision_endpoint(fake_guest, monkeypatch):
    monkeypatch.delenv("RV_VISION_BASE_URL", raising=False)
    guest, rec = fake_guest("vision_tool")
    attempt = deploy.GUEST_ROOT + "/attempts/task-2"
    guest.launch(attempt, _launch_config("vision_tool"))
    assert json.loads(rec.files[attempt + "/request.json"])["rv"] == {"ask_enabled": False, "call_offset": 0}


def test_stop_and_confirm_without_guard(fake_guest, no_guard_import, monkeypatch):
    guest, rec = fake_guest("cli")
    replies = {"systemctl show": "ActiveState=inactive\nMainPID=0\nControlPID=0\nControlGroup=\n"}
    monkeypatch.setattr(deploy.Guest, "command", lambda self, c, timeout=60, check=True, read_only=False:
                        rec.commands.append(c) or next((v for k, v in replies.items() if c.startswith(k)), ""))
    assert guest.stop_and_confirm("osworld-dsh-task-1")["confirmed"] is True
    with pytest.raises(ValueError):
        guest.stop("sshd")
    replies["systemctl show"] = "ActiveState=active\nMainPID=12\n"
    with pytest.raises(deploy.DeepSeekError):
        guest.stop_and_confirm("osworld-dsh-task-1")


def test_agent_carries_call_budget_across_phases(tmp_path, monkeypatch):
    outputs = iter([5, 12])

    def fake_predict(self, instruction, obs=None):
        out = tmp_path / f"p{self.guest.call_offset}"
        (out / "dsh").mkdir(parents=True)
        (out / "dsh" / "budget.json").write_text(json.dumps({"model_calls": next(outputs)}))
        self.last_output = out
        return "done", []

    monkeypatch.setattr(deploy.DeepSeekAgent, "predict", fake_predict)
    agent = deploy.RvAgent.__new__(deploy.RvAgent)
    agent.guest = SimpleNamespace()
    agent.env = SimpleNamespace(user_simulator=object())
    agent.predict("x")
    assert agent.guest.ask_enabled is True and agent.task_calls == 5
    agent.env = SimpleNamespace()
    agent.predict("y")
    assert agent.guest.call_offset == 5 and agent.guest.ask_enabled is False and agent.task_calls == 12


# ---------------------------------------------------------------- TLS clock


def test_restore_clock_after_task_changes_date(monkeypatch):
    calls, files = [], {}
    times = iter(["1754049600", "1792731600"])
    monkeypatch.setattr(deploy.time, "time", lambda: 1792731600)
    guest = deploy.RvGuest("example", "/k")
    guest.command = lambda cmd, **kw: next(times) if cmd == "date +%s" else calls.append(cmd) or ""
    guest.put = lambda path, text: files.update({path: json.loads(text)})
    result = guest.ensure_tls_clock()
    assert result["changed"] and result["after_epoch"] == 1792731600
    assert len(calls) == 2 and "set-ntp false" in calls[0] and "TZ=UTC date -s" in calls[1]
    assert files[deploy.RV_GUEST + "/tls_clock.json"]["policy"].startswith("real OS clock")


def test_clock_keeps_healthy_guest_unchanged(monkeypatch):
    monkeypatch.setattr(deploy.time, "time", lambda: 1792731600)
    guest = deploy.RvGuest("example", "/k")
    guest.command = lambda cmd, **kw: "1792731600" if cmd == "date +%s" else pytest.fail("no date mutation")
    guest.put = lambda *args: None
    assert guest.ensure_tls_clock()["changed"] is False


def test_bad_clock_fails_closed(monkeypatch):
    monkeypatch.setattr(deploy.time, "time", lambda: 1792731600)
    guest = deploy.RvGuest("example", "/k")
    guest.command = lambda cmd, **kw: "1754049600" if cmd == "date +%s" else ""
    guest.put = lambda *args: None
    with pytest.raises(deploy.DeepSeekError, match="TLS clock"):
        guest.ensure_tls_clock()


# ------------------------------------------------------------ docker provider


def test_package_manager_bootstrap_waits_without_removing_locks():
    seen = []
    guest = SimpleNamespace(command=lambda command, **kwargs: seen.append((command, kwargs)))
    deploy.prepare_package_manager(guest)
    assert len(seen) == 2
    assert "mask --runtime --now packagekit.service apt-daily.timer apt-daily-upgrade.timer" in seen[0][0]
    assert "fcntl.lockf" in seen[1][0] and "LOCK_NB" in seen[1][0]
    assert "rm " not in seen[1][0] and "kill" not in seen[1][0]
    assert seen[0][1]["timeout"] == 120 and seen[1][1]["timeout"] == 210


def test_task082_controller_rebind_uses_case_ip_and_coherent_guest_ports():
    for ip in ("10.201.0.2", "10.201.0.3"):
        controller = SimpleNamespace(vm_ip="localhost", server_port=5001, chromium_port=9223,
                                     vlc_port=8081, http_server="http://localhost:5001")
        env = SimpleNamespace(setup_controller=controller)
        deploy.rebind_task_setup_controller(env, ip, "082")
        assert controller.vm_ip == ip and controller.server_port == 5000
        assert controller.chromium_port == 9222 and controller.vlc_port == 8080
        assert controller.http_server == f"http://{ip}:5000"
        assert controller.http_server_setup_root == f"http://{ip}:5000/setup"
    untouched = SimpleNamespace(vm_ip="localhost")
    deploy.rebind_task_setup_controller(SimpleNamespace(setup_controller=untouched), "10.201.0.4", "001")
    assert vars(untouched) == {"vm_ip": "localhost"}


def test_authorize_ssh_key_validates_and_falls_back():
    with pytest.raises(ValueError):
        deploy.authorize_ssh_key(None, "not a key")
    seen = []
    controller = SimpleNamespace(run_bash_script=lambda script, timeout: seen.append(script) or {"output": "ok\n"})
    deploy.authorize_ssh_key(controller, "ssh-ed25519 AAAA test\n")
    assert "authorized_keys" in seen[0] and "ssh-ed25519 AAAA test" in seen[0]


class FakeEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        container = SimpleNamespace(name="vm_042", reload=lambda: None,
                                    attrs={"NetworkSettings": {"Networks": {"n": {"IPAddress": "10.0.0.9"}}}})
        self.provider = SimpleNamespace(container=container, environment={"RAM_SIZE": "8G"})
        self.controller = SimpleNamespace()
        self.setup_controller = SimpleNamespace()

    def close(self):
        self.closed = True


@pytest.fixture
def docker_case(tmp_path, monkeypatch):
    module = types.ModuleType("desktop_env.desktop_env")
    module.DesktopEnv = FakeEnv
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", module)
    guests = []

    class PackageGuest:
        def __init__(self, ip, key):
            self.ip, self.commands, self.closed = ip, [], False
            guests.append(self)

        def connect(self):
            pass

        def command(self, command, timeout=60, **kw):
            self.commands.append(command)
            return ""

        def put(self, *a, **kw):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(deploy, "Guest", PackageGuest)
    monkeypatch.setattr(deploy, "authorize_ssh_key", lambda controller, key: None)
    monkeypatch.setattr(web, "load_config", lambda config=None: None)
    monkeypatch.setattr(web, "attach", lambda env, key, password, config=None: None)
    monkeypatch.setattr(deploy.RvGuest, "network_guard", False)
    (tmp_path / "key.pub").write_text("ssh-ed25519 AAAA k\n")
    (tmp_path / "042").mkdir()
    args = SimpleNamespace(volume_size=10, result_dir=tmp_path, ssh_private_key=str(tmp_path / "key"))
    return args, {"id": "042"}, guests


def test_docker_new_env_without_guard(docker_case, no_guard_import, monkeypatch):
    args, example, guests = docker_case
    monkeypatch.delitem(deploy.TASK_BY_HOST, "10.0.0.9", raising=False)
    env = deploy.docker_new_env(args, example)
    assert env.kwargs["provider_name"] == "docker" and env.kwargs["client_password"] == deploy.CLIENT_PASSWORD
    assert env.vm_ip == "10.0.0.9" and env.server_port == 5000 and env.rv_container == "vm_042"
    assert env.rv_website_tunnel is None and deploy.TASK_BY_HOST["10.0.0.9"] == "042"
    assert len(guests) == 1 and guests[0].closed and len(guests[0].commands) == 2
    assert not any("ca-certificates" in c for c in guests[0].commands)
    assert json.loads((args.result_dir / "042" / "container.json").read_text())["container"] == "vm_042"
    deploy.TASK_BY_HOST.pop("10.0.0.9")
    deploy.docker_terminate_case(env, args, args.result_dir / "042")
    assert env.closed
    assert json.loads((args.result_dir / "042" / "cleanup.json").read_text()) == {"container": "vm_042",
                                                                                  "state": "removed"}


def test_docker_new_env_installs_guard_ca_only_when_enabled(docker_case, monkeypatch):
    import mmcoding.host

    args, example, guests = docker_case
    installed = []
    fake = types.SimpleNamespace(install_ca=lambda guest, sudo: installed.append(guest.ip))
    monkeypatch.setitem(sys.modules, "mmcoding.host.guard", fake)
    monkeypatch.setattr(mmcoding.host, "guard", fake, raising=False)
    monkeypatch.setattr(deploy.RvGuest, "network_guard", True)
    deploy.docker_new_env(args, example)
    deploy.TASK_BY_HOST.pop("10.0.0.9")
    assert installed == ["10.0.0.9"]


def test_docker_new_env_closes_vm_on_provision_failure(docker_case, monkeypatch):
    args, example, _ = docker_case
    envs = []
    monkeypatch.setattr(sys.modules["desktop_env.desktop_env"], "DesktopEnv",
                        lambda **kw: envs.append(FakeEnv(**kw)) or envs[-1])

    def broken(guest):
        raise RuntimeError("apt busy")
    monkeypatch.setattr(deploy, "prepare_package_manager", broken)
    with pytest.raises(RuntimeError, match="apt busy"):
        deploy.docker_new_env(args, example)
    assert envs[0].closed and "10.0.0.9" not in deploy.TASK_BY_HOST
    cleanup = json.loads((args.result_dir / "042" / "cleanup.json").read_text())
    assert cleanup["phase"] == "provision_failed_before_agent"


# --------------------------------------------------------------- transports


def test_evaluator_transport_respects_long_command_deadline(monkeypatch):
    from hybrid_agents.deepseek.io_recovery import RecoveryRequests
    seen = []

    def base(self, method, url, **kwargs):
        seen.append(kwargs)
        return "unchanged result"
    monkeypatch.setattr(RecoveryRequests, "request", base)
    deploy.install_evaluator_timeout_transport()
    recovery = SimpleNamespace(setup=False)
    assert RecoveryRequests.request(recovery, "POST", "http://localhost:5000/setup/execute",
                                    data=json.dumps({"command": ["python3", "script.py"], "timeout": 300})) == "unchanged result"
    assert seen[-1]["timeout"] == (10, 320)
    RecoveryRequests.request(recovery, "POST", "http://localhost:5000/setup/execute",
                             json={"timeout": 300}, timeout=(10, 600))
    assert seen[-1]["timeout"] == (10, 600)
    RecoveryRequests.request(recovery, "GET", "http://localhost:5000/file")
    assert "timeout" not in seen[-1]
    RecoveryRequests.request(SimpleNamespace(setup=True), "POST", "http://localhost:5000/setup/execute",
                             json={"timeout": 300})
    assert "timeout" not in seen[-1]
    deploy.install_evaluator_timeout_transport()
    RecoveryRequests.request(recovery, "POST", "http://localhost:5000/setup/execute", json={"timeout": 300})
    assert len(seen) == 5 and seen[-1]["timeout"] == (10, 320)


def test_dimension_transport_bounds_tall_image_under_byte_limit(monkeypatch, tmp_path):
    from PIL import Image
    from hybrid_agents.deepseek import judge_images
    path = tmp_path / "tall.png"
    Image.new("RGB", (100, 9000), "white").save(path)
    data = path.read_bytes()
    kwargs = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "original rubric"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
    ]}]}
    monkeypatch.setattr(judge_images, "bound_request", judge_images.bound_request)
    deploy.install_judge_dimension_transport()
    payload = judge_images.bound_request(kwargs, [str(path)], tmp_path / "bounded")
    record = json.loads((tmp_path / "bounded/images.json").read_text())
    assert record["crop"] is False and record["images"][0]["original_dimensions"] == [100, 9000]
    assert max(record["images"][0]["sent_dimensions"]) == 8000
    assert path.read_bytes() == data
    assert payload["messages"][0]["content"][0]["text"] == "original rubric"
    assert kwargs["messages"][0]["content"][1]["image_url"]["url"].endswith(base64.b64encode(data).decode())


def test_dimension_transport_rechecks_bytes_after_tall_jpeg_conversion(monkeypatch, tmp_path):
    from PIL import Image
    from hybrid_agents.deepseek import judge_images
    path = tmp_path / "tall.jpg"
    image = Image.effect_noise((240, 9001), 90).convert("RGB")
    image.save(path, quality=15)
    image.close()
    data = path.read_bytes()
    kwargs = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "rubric unchanged"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}},
    ]}]}
    monkeypatch.setattr(judge_images, "bound_request", judge_images.bound_request)
    deploy.install_judge_dimension_transport()
    payload = judge_images.bound_request(kwargs, [str(path)], tmp_path / "jpeg-bound", max_bytes=500000)
    assert len(judge_images.json_bytes(payload)) <= 500000
    record = json.loads((tmp_path / "jpeg-bound/images.json").read_text())
    assert max(record["images"][0]["sent_dimensions"]) <= 8000
    assert path.read_bytes() == data and record["crop"] is False


# ------------------------------------------------------------------- phases


def test_real_basetask_phase_wrapper_captures_screen_and_restores(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from desktop_env.task_base import MultiPhaseTask
    from hybrid_agents.deepseek import phases

    calls = []

    class Task(MultiPhaseTask):
        def get_phases(self):
            return [{"instruction": "unchanged", "gate_min_score": .35,
                     "setup": self.phase_setup, "evaluate": self.phase_evaluate}]

        def phase_setup(self, controller, **kwargs):
            calls.append(("setup", controller, kwargs))
            return "setup unchanged"

        def phase_evaluate(self, env):
            calls.append(("evaluate", env))
            return .2

    @contextmanager
    def deadline():
        calls.append(("deadline",))
        yield

    @contextmanager
    def transport(*args):
        calls.append(("transport", args))
        yield

    monkeypatch.setattr(phases, "setup_deadline", deadline)
    monkeypatch.setattr(phases, "setup_transport", transport)
    monkeypatch.setattr(phases, "screenshot", lambda guest, path: calls.append(("screen", guest, path)))
    task = Task()
    # Ordinary assignment updates the task's mapping while attribute lookup still resolves the
    # class method.
    original = task.get_phases
    task.get_phases = lambda: pytest.fail("mapping assignment is not a method override")
    assert task.get_phases == original
    mapping_before = dict(task)
    agent = SimpleNamespace(phase_mode=False, last_output=tmp_path / "phases/01")
    with deploy.phase_evidence_adapters(task, agent, "guest", tmp_path, secrets=("secret",)):
        wrapped = task.get_phases()[0]
        assert wrapped["instruction"] == "unchanged" and wrapped["gate_min_score"] == .35
        assert wrapped["setup"]("controller", use_proxy=False) == "setup unchanged"
        assert wrapped["evaluate"]("env") == .2
        assert calls[-2:] == [("screen", "guest", agent.last_output / "final_screen.png"), ("evaluate", "env")]
    assert agent.phase_mode is True and task.get_phases == original
    assert dict(task) == mapping_before and "get_phases" not in vars(task)


def test_phase_wrapper_restores_true_override_on_error(tmp_path):
    from desktop_env.task_base import MultiPhaseTask

    class Task(MultiPhaseTask):
        def get_phases(self):
            return []

    task = Task()
    original = lambda: []
    object.__setattr__(task, "get_phases", original)
    agent = SimpleNamespace(phase_mode=False)
    with pytest.raises(RuntimeError, match="expected"):
        with deploy.phase_evidence_adapters(task, agent, None, tmp_path):
            raise RuntimeError("expected")
    assert task.get_phases is original


def test_nonphased_wrapper_leaves_task_and_agent_unchanged(tmp_path):
    task = SimpleNamespace()
    agent = SimpleNamespace(phase_mode=False)
    with deploy.phase_evidence_adapters(task, agent, None, tmp_path):
        assert agent.phase_mode is False and not hasattr(task, "get_phases")


# -------------------------------------------------------------------- audit


AUDIT_LOG = "\n".join([
    "type=USER msg=audit(100.000:1): pid=1 uid=0 auid=4294967295 ses=4294967295 msg='text=rv_agent_start unit=u exe=\"/usr/sbin/auditctl\"'",
    'type=SYSCALL msg=audit(101.000:2): arch=c000003e syscall=59 success=yes exit=0 ppid=10 pid=11 auid=4294967295 uid=1000 ses=4294967295 comm="bash" exe="/usr/bin/bash" key="rv_exec"',
    'type=EXECVE msg=audit(101.000:2): argc=3 a0="bash" a1="-c" a2=78646F746F6F6C20636C69636B2031',
    'type=SYSCALL msg=audit(102.000:3): arch=c000003e syscall=59 success=yes exit=0 ppid=20 pid=21 auid=1000 uid=1000 ses=2 comm="bash" exe="/usr/bin/bash" key="rv_exec"',
    'type=EXECVE msg=audit(102.000:3): argc=1 a0="bash"',
    'type=SYSCALL msg=audit(103.000:4): arch=c000003e syscall=59 success=yes exit=0 ppid=21 pid=22 auid=1000 uid=1000 ses=2 comm="python3" exe="/usr/bin/python3.10" key="rv_exec"',
    'type=EXECVE msg=audit(103.000:4): argc=2 a0="python3" a1="evil.py"',
    "type=USER msg=audit(200.000:9): pid=1 uid=0 auid=4294967295 ses=4294967295 msg='text=rv_agent_stop unit=u exe=\"/usr/sbin/auditctl\"'",
])


def test_audit_parses_hex_args_and_marks():
    execs, marks, _ = audit.parse_audit(AUDIT_LOG)
    assert [m["mark"] for m in marks] == ["rv_agent_start", "rv_agent_stop"]
    assert execs[0]["argv"] == ["bash", "-c", "xdotool click 1"]


def test_audit_flags_display_attempts_and_notes_terminal_exec(tmp_path):
    task = tmp_path / "001"
    ev = task / "rv_evidence"
    ev.mkdir(parents=True)
    (ev / "audit.log").write_text(AUDIT_LOG)
    (ev / "session_agent_start.json").write_text(json.dumps({"audit_session": 2}))
    (ev / "ps_agent_start.txt").write_text("20 1 1000 50 gnome-terminal- /usr/libexec/gnome-terminal-server\n")
    report_cli = audit.analyze(task, "cli")
    assert report_cli["harness"] == "cli"
    assert any(a["type"] == "display_control_exec" for a in report_cli["attempts"])
    report_gui = audit.analyze(task, "GUI")
    assert report_gui["harness"] == "gui"
    assert not any(v["type"] == "terminal_exec" for v in report_gui["violations"])
    assert any(n["type"] == "terminal_exec" and "evil.py" in n["cmd"] for n in report_gui["notes"])
    assert any(v["type"] == "gui_unexpected_sandbox_exec" for v in report_gui["violations"])
    with pytest.raises(KeyError):
        audit.analyze(task, "desktop")


def test_audit_root_reads_harness_from_manifest(tmp_path):
    (tmp_path / "001").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"rv_harnesses": {"harness": "mm_coding"}}))
    assert audit.analyze_root(tmp_path) == [{"task": "001", "harness": "mm_coding",
                                             "notes": ["no rv_evidence collected"], "summary": {}}]


def test_bus_denial_parser():
    text = ("1.0 vm xdg-dbus-proxy[1]: C5: -> org.gnome.Shell call org.gnome.Shell.Eval at /org/gnome/Shell\n"
            "1.0 vm xdg-dbus-proxy[1]: *HIDDEN* (ping)\n"
            "1.1 vm xdg-dbus-proxy[1]: C6: -> ca.desrt.dconf call ca.desrt.dconf.Writer.Change at /x\n")
    assert audit.parse_bus_denials(text) == [
        {"dest": "org.gnome.Shell", "method": "org.gnome.Shell.Eval", "path": "/org/gnome/Shell", "t": "1.0"}]


# ------------------------------------------------------------- network guard


@pytest.fixture
def guard_modules(tmp_path, monkeypatch):
    cfg = json.loads((REPO / "configs" / "guard.example.json").read_text())
    cfg.update(registry=str(tmp_path / "registry"), log_dir="$GUARD_TEST_DIR/logs", ca_cert=str(tmp_path / "ca.pem"),
               nss_debs={}, task_allow=str(tmp_path / "allow.json"), passthrough_hosts=["llm.example"])
    (tmp_path / "guard.json").write_text(json.dumps(cfg))
    (tmp_path / "ca.pem").write_text("-----BEGIN CERTIFICATE-----\nTEST\n-----END CERTIFICATE-----\n")
    (tmp_path / "allow.json").write_text(json.dumps({"tasks": {"042": ["github.com/xlang-ai/instructor-embedding"]}}))
    monkeypatch.setenv("RV_GUARD_CONFIG", str(tmp_path / "guard.json"))
    monkeypatch.setenv("GUARD_TEST_DIR", str(tmp_path))
    for name in ("mmcoding.host.guard", "mmcoding.host.guard_addon"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import importlib
    addon = importlib.import_module("mmcoding.host.guard_addon")
    guard = importlib.import_module("mmcoding.host.guard")
    yield guard, addon
    for name in ("mmcoding.host.guard", "mmcoding.host.guard_addon"):
        sys.modules.pop(name, None)


def test_guard_example_config_resolves_paths(monkeypatch):
    monkeypatch.delenv("RV_GUARD_CONFIG", raising=False)
    from mmcoding.host import guard_addon
    cfg = guard_addon.load_config(REPO / "configs" / "guard.example.json")
    assert cfg["port"] == 8893 and cfg["rules_version"] == "1" and cfg["task_allow"] is None
    assert cfg["log_dir"] == REPO / "data" / "guard" / "logs"
    assert cfg["ca_cert"] == Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"
    assert not any(key.startswith("_") for key in cfg)


def test_guard_rules(guard_modules):
    guard, addon = guard_modules
    assert addon.REGISTRY.name == "registry" and addon.GLOBAL_LOG.parent.name == "logs"
    assert addon.decide("https://raw.githubusercontent.com/xlang-ai/OSWorld/main/README.md",
                        "raw.githubusercontent.com", []) == "osworld_source"
    assert addon.decide("https://r.jina.ai/https%253A%252F%252Fgithub.com%252Fxlang-ai", "r.jina.ai", []) == "osworld_source"
    assert addon.decide("http://mailhub.osworld.test/inbox", "mailhub.osworld.test", []) is None
    assert addon.decide("https://github.com/numpy/numpy", "github.com", []) is None
    assert addon.decide("https://example.org/f/0123456789abcdef", "example.org", ["0123456789abcdef"]) == "task_identifier"
    allow = addon.task_allow("042")
    assert addon.decide("https://github.com/xlang-ai/instructor-embedding", "github.com", [], allow) is None
    assert addon.decide("https://github.com/xlang-ai/OSWorld", "github.com", [], allow) == "osworld_source"
    assert addon.task_allow("043") == []
    assert addon.intercept_host("raw.githubusercontent.com") and not addon.intercept_host("pypi.org")
    assert any(p.search("api.llm.example") for p in addon.PASSTHROUGH_SNI)
    assert addon.unescape('{\\"url\\": \\"https:\\/\\/github.com\\/xlang-ai\\"}') == '{"url": "https://github.com/xlang-ai"}'
    assert addon.response_rule("see evaluation_examples/examples/chrome", []) == "response_osworld_listing"
    uuid = "bb5e4c0d-f964-439c-97b6-bdb9747de3f4"
    assert addon.response_rule('{"id": "%s", "evaluator": {}}' % uuid, [uuid]) == "response_task_config"
    assert guard.chains("10.0.0.9") == ("RVG_0a000009", "RVGF_0a000009")


def _pack(payload: bytes) -> bytes:
    size = len(payload)
    header = bytearray([0x80 | (3 << 4) | (size & 0x0F)])
    size >>= 4
    while size:
        header.append((0x80 if size >> 7 else 0) | (size & 0x7F))
        size >>= 7
    body = b"PACK" + (2).to_bytes(4, "big") + (1).to_bytes(4, "big") + bytes(header) + zlib.compress(payload)
    line = b"\x01" + body
    return b"%04x" % (len(line) + 4) + line + b"0000"


def test_guard_inspects_compressed_transfers(guard_modules):
    _, addon = guard_modules
    uuid = "bb5e4c0d-f964-439c-97b6-bdb9747de3f4"
    needles = addon._needles([uuid, "short"])
    assert needles == [uuid.encode()]
    assert addon.git_pack_hit(_pack(b'{"id": "' + uuid.encode() + b'"}'), needles)
    assert not addon.git_pack_hit(_pack(b"ordinary source file " * 4), needles)
    import gzip
    assert addon.archive_hit(gzip.compress(b"task " + uuid.encode()), needles)
    assert not addon.archive_hit(gzip.compress(b"nothing here"), needles)


def test_guard_installs_ca_and_checks_trust(guard_modules):
    guard, _ = guard_modules
    puts, commands = [], []
    reply = {"value": "system_store=ok\ncertifi_patched=1\nnss=/home/user/.pki/nssdb\ndone"}
    guest = SimpleNamespace(put=lambda path, data, mode=0o600: puts.append(path),
                            command=lambda c, timeout=60: commands.append(c) or reply["value"])
    assert "system_store=ok" in guard.install_ca(guest, lambda c: "sudo " + c)
    assert puts == ["/tmp/rv-answer-proxy.crt"] and commands[-1].startswith("sudo bash -c")
    reply["value"] = "system_store=ok\nnss=no_certutil"
    with pytest.raises(guard.GuardError):
        guard.install_ca(guest, lambda c: c)


def test_guard_config_pins_and_requirements(guard_modules, tmp_path, monkeypatch):
    guard, _ = guard_modules
    cfg = guard.config()
    cfg["ca_sha256"] = "0" * 64
    with pytest.raises(guard.GuardError, match="ca_sha256"):
        guard._ca(cfg)
    monkeypatch.setenv("RV_GUARD_TERMS", "{}")
    with pytest.raises(guard.GuardError, match="RV_GUARD_TERMS"):
        guard.harden(SimpleNamespace(host="10.0.0.9"), lambda c: c)


# -------------------------------------------------------- OSWorld 2.0 websites


def config(**overrides):
    value = dict(suffix="osworld.test", host_port=18090, guest_forward_port=18090,
                 hostnames=["mailhub.osworld.test", "studio.streamview.osworld.test"],
                 repo_commit="pinned", submodules={"mailhub_web": "pinned"})
    value.update(overrides)
    return value


@pytest.mark.parametrize("overrides", [
    {"suffix": "evil.test;echo x"}, {"suffix": "*.test"}, {"suffix": "OSWORLD.test"},
    {"host_port": True}, {"host_port": "18090"}, {"host_port": 65536},
    {"guest_forward_port": 80}, {"guest_forward_port": 0},
    {"hostnames": []}, {"hostnames": "mailhub.osworld.test"},
    {"hostnames": ["mailhub.osworld.test\n127.0.0.1 evil"]},
    {"hostnames": ["evil.osworld.test"]}, {"hostnames": ["mailhub.example.test"]},
    {"hostnames": ["osworld.test"]}, {"hostnames": ["*.osworld.test"]},
    {"hostnames": ["mailhub.osworld.test", "mailhub.osworld.test"]},
    {"hostnames": [None]},
])
def test_invalid_config(overrides):
    with pytest.raises(ValueError):
        web.load_config(config(**overrides))


def test_config_loading_defaults_and_explicit_missing(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setattr(web, "DEFAULT_CONFIG", path)
    monkeypatch.delenv("RV_WEBSITE_CONFIG", raising=False)
    assert web.attach(None, None, None) is None
    monkeypatch.setenv("RV_WEBSITE_CONFIG", str(path))
    with pytest.raises(FileNotFoundError):
        web.load_config()
    path.write_text(json.dumps(config()))
    assert web.load_config().hostnames == tuple(config()["hostnames"])
    assert web.load_config(path) == web.load_config(config())
    path.write_text("[]")
    with pytest.raises(ValueError):
        web.load_config(path)
    path.write_text("broken json")
    with pytest.raises(ValueError):
        web.load_config(path)


def test_hosts_replaces_only_fixed_aliases_and_is_idempotent(tmp_path):
    cfg = web.load_config(config())
    original = ("127.0.0.1 localhost\n::1 localhost ip6-localhost\n"
                "10.2.3.4 MAILHUB.osworld.test unrelated # keep comment\n"
                "::1 studio.streamview.osworld.test\n# unrelated comment\n")
    result = web.hosts_text(original, cfg)
    assert "127.0.0.1 localhost\n::1 localhost ip6-localhost\n" in result
    assert "10.2.3.4\tunrelated # keep comment" in result
    assert "::1 studio.streamview" not in result
    assert result.count("mailhub.osworld.test") == 1
    assert web.hosts_text(result, cfg) == result
    path = tmp_path / "hosts"
    path.write_text(original)
    assert web.install_hosts(cfg, path)
    assert path.read_text() == result
    other = web.load_config(config(suffix="apps.example.test", host_port=19090,
                                  guest_forward_port=28090,
                                  hostnames=["mailhub.apps.example.test"]))
    assert "mailhub.osworld.test" not in web.hosts_text(result, other)


class ResultChannel:
    def __init__(self, status=0):
        self.status = status
        self.closed = False

    def recv_ready(self):
        return False

    def recv_stderr_ready(self):
        return False

    def exit_status_ready(self):
        return True

    def recv_exit_status(self):
        return self.status

    def close(self):
        self.closed = True


class Input(io.StringIO):
    def close(self):
        self.written = self.getvalue()
        super().close()


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.cancels = []

    def set_keepalive(self, seconds):
        self.keepalive = seconds

    def request_port_forward(self, address, port, handler):
        self.requests.append((address, port))
        self.handler = handler
        return port

    def cancel_port_forward(self, address, port):
        self.cancels.append((address, port))


class FakeSSH:
    def __init__(self, transport, fail_at=None):
        self.transport = transport
        self.commands = []
        self.inputs = []
        self.channels = []
        self.fail_at = fail_at

    def get_transport(self):
        return self.transport

    def exec_command(self, command, timeout):
        self.commands.append(command)
        status = 1 if self.fail_at == len(self.commands) else 0
        channel = ResultChannel(status)
        stdin, stdout, stderr = Input(), io.StringIO(), io.StringIO()
        stdout.channel = channel
        self.inputs.append(stdin)
        self.channels.append(channel)
        return stdin, stdout, stderr


class FakeGuest:
    def __init__(self, fail_at=None):
        self.transport = FakeTransport()
        self.ssh = FakeSSH(self.transport, fail_at)
        self.connects = self.closes = 0

    def connect(self):
        self.connects += 1

    def close(self):
        self.closes += 1


@pytest.fixture
def fake_attach(monkeypatch):
    guest = FakeGuest()
    calls = []
    def factory(host, key, user):
        calls.append((host, key, user))
        return guest
    monkeypatch.setattr(web, "SSHGuest", factory)
    monkeypatch.setattr(web, "_probe_host", lambda cfg: calls.append("host probe"))
    return guest, calls


def test_attach_separate_transport_loopback_and_cleanup(fake_attach):
    guest, calls = fake_attach
    cfg = config(suffix="different.test", host_port=19090, guest_forward_port=28090,
                 hostnames=["mailhub.different.test"])
    tunnel = web.attach(SimpleNamespace(vm_ip="172.18.0.2"), "/key", "secret ' $", cfg)
    assert calls == [("172.18.0.2", "/key", "user"), "host probe"]
    assert guest.connects == 1
    assert guest.transport.requests == [("127.0.0.1", 28090)]
    commands = guest.ssh.commands
    assert len(commands) == 3
    assert "secret" not in " ".join(commands)
    assert all(x.written == "secret ' $\n" for x in guest.ssh.inputs[:2])
    assert guest.ssh.inputs[2].written == ""
    unit = shlex.split(commands[1])
    assert "--unit=rv-web-loopback" in unit
    assert "TCP-LISTEN:80,bind=127.0.0.1,reuseaddr,fork" in unit
    assert "TCP:127.0.0.1:28090" in unit
    assert "0.0.0.0" not in commands[1]
    probe = shlex.split(commands[2])[-1]
    assert "mailhub.different.test" in probe
    assert "ports = {80, 28090}" in probe
    assert "0100007F" in probe  # Reject sshd GatewayPorts=yes/wildcard listeners.
    assert "/api/state" not in probe
    with tunnel as same:
        assert same is tunnel
    tunnel.close()
    assert guest.closes == 1
    assert guest.transport.cancels == [("127.0.0.1", 28090)]
    assert "systemctl stop rv-web-loopback" in commands[-1]
    assert tunnel._password == ""
    assert all(ch.closed for ch in guest.ssh.channels)


def test_guest_hosts_script_matches_host_helper(fake_attach, monkeypatch, tmp_path):
    guest, _ = fake_attach
    tunnel = web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config())
    script = shlex.split(guest.ssh.commands[0])[-1]
    path = tmp_path / "hosts"
    path.write_text("127.0.0.1 localhost\n")
    from pathlib import Path
    import pathlib
    real_path = Path
    monkeypatch.setattr(pathlib, "Path", lambda p: path if p == "/etc/hosts" else real_path(p))
    exec(compile(script, "<trusted-hosts-helper>", "exec"), {})
    assert path.read_text() == web.hosts_text("127.0.0.1 localhost\n", web.load_config(config()))
    tunnel.close()


@pytest.mark.parametrize("fail_at,should_stop", [(1, False), (2, False), (3, True)])
def test_setup_failure_cancels_owned_resources(fake_attach, fail_at, should_stop):
    guest, _ = fake_attach
    guest.ssh.fail_at = fail_at
    with pytest.raises(RuntimeError, match="guest command failed"):
        web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config())
    assert guest.closes == 1
    assert guest.transport.cancels == [("127.0.0.1", 18090)]
    assert any("systemctl stop" in cmd for cmd in guest.ssh.commands) == should_stop


def test_host_readiness_failure_never_connects(fake_attach, monkeypatch):
    guest, _ = fake_attach
    def fail(cfg):
        raise RuntimeError("not ready")
    monkeypatch.setattr(web, "_probe_host", fail)
    with pytest.raises(RuntimeError, match="not ready"):
        web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config())
    assert guest.connects == 0
    assert guest.closes == 1
    assert not guest.transport.requests


def test_dead_vm_cleanup_does_not_reconnect(fake_attach, monkeypatch):
    guest, _ = fake_attach
    tunnel = web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config())
    def dead(*a, **kw):
        raise OSError("VM terminated")
    monkeypatch.setattr(guest.ssh, "exec_command", dead)
    monkeypatch.setattr(guest.transport, "cancel_port_forward", dead)
    tunnel.close()
    assert guest.connects == guest.closes == 1


@pytest.mark.parametrize("bad", ["pw\ninjected", "pw\r", "pw\0", None])
def test_password_validation_before_guest_creation(fake_attach, bad):
    guest, calls = fake_attach
    with pytest.raises(ValueError):
        web.attach(SimpleNamespace(vm_ip="guest"), "key", bad, config())
    assert not calls


class Stream:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.sent = []
        self.shutdowns = []
        self.closed = False

    def recv(self, count):
        assert count == 65536
        return self.chunks.pop(0)

    def sendall(self, data):
        self.sent.append(data)

    def settimeout(self, timeout):
        self.timeout = timeout

    def shutdown(self, direction):
        self.shutdowns.append(direction)

    def close(self):
        self.closed = True


def test_relay_splices_exact_bytes_and_half_closes(monkeypatch):
    request = b"GET / HTTP/1.1\r\nHost: mailhub.osworld.test\r\nCookie: case=x\r\n\r\n"
    reply = b"HTTP/1.1 200 OK\r\nSet-Cookie: case=y\r\n\r\nbody"
    channel, local = Stream([request, b""]), Stream([reply, b""])
    endpoints = []
    def connect(address, timeout):
        endpoints.append((address, timeout))
        return local
    monkeypatch.setattr(web.socket, "create_connection", connect)
    monkeypatch.setattr(web.select, "select", lambda readers, w, x, t: (list(readers), [], []))
    tunnel = web.WebsiteTunnel(FakeGuest(), web.load_config(config(host_port=19999)), "pw")
    tunnel._streams.add(channel)
    tunnel._relay(channel)
    assert endpoints == [(("127.0.0.1", 19999), 5)]
    assert local.sent == [request]
    assert channel.sent == [reply]
    assert local.shutdowns == channel.shutdowns == [socket.SHUT_WR]
    assert local.closed and channel.closed
    assert not tunnel._streams


def test_relay_connect_failure_closes_channel(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Caddy gone")
    monkeypatch.setattr(web.socket, "create_connection", fail)
    channel = Stream()
    tunnel = web.WebsiteTunnel(FakeGuest(), web.load_config(config()), "pw")
    tunnel._relay(channel)
    assert channel.closed


def test_handler_only_starts_bounded_daemons_and_close_rejects(fake_attach, monkeypatch):
    guest, _ = fake_attach
    tunnel = web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config())
    workers = []
    class Thread:
        def __init__(self, **kw):
            self.kw = kw
            workers.append(self)
        def start(self):
            self.started = True
        def join(self, timeout):
            self.joined = True
    monkeypatch.setattr(web.threading, "Thread", Thread)
    channel = Stream()
    guest.transport.handler(channel, ("ignored", 5), ("127.0.0.1", 18090))
    assert workers[0].kw["daemon"] is True and workers[0].started
    tunnel.MAX_STREAMS = 1
    excess = Stream()
    guest.transport.handler(excess, ("ignored", 5), ("127.0.0.1", 18090))
    assert excess.closed and len(workers) == 1
    tunnel.close()
    assert channel.closed and workers[0].joined
    late = Stream()
    guest.transport.handler(late, ("ignored", 5), ("127.0.0.1", 18090))
    assert late.closed


def test_handler_rejects_nonloopback_destination():
    tunnel = web.WebsiteTunnel(FakeGuest(), web.load_config(config()), "pw")
    channel = Stream()
    tunnel._handle(channel, ("ignored", 0), ("0.0.0.0", 18090))
    assert channel.closed
    assert not tunnel._threads


def test_host_probe_only_reads_loopback_with_original_host(monkeypatch):
    requests = []
    connections = []
    class HTTP:
        def __init__(self, address, port, timeout):
            assert address == "127.0.0.1" and timeout == 5
            self.port = port
            self.closed = False
            connections.append(self)
        def request(self, method, path, headers):
            requests.append((self.port, method, path, headers))
        def getresponse(self):
            return SimpleNamespace(status=200)
        def close(self):
            self.closed = True
    monkeypatch.setattr(web.http.client, "HTTPConnection", HTTP)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 80))])
    web._probe_host(web.load_config(config()))
    assert len(requests) == 4
    assert {r[0] for r in requests} == {80, 18090}
    assert {r[3]["Host"] for r in requests} == set(config()["hostnames"])
    assert all(r[1:3] == ("GET", "/") for r in requests)
    assert all(c.closed for c in connections)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("192.168.1.2", 80))])
    with pytest.raises(RuntimeError, match="not exclusively loopback"):
        web._probe_host(web.load_config(config()))
    assert len(requests) == 4


@pytest.mark.parametrize("listener", ["0100007F", "00000000"])
def test_guest_probe_read_only_and_rejects_wildcard(monkeypatch, listener):
    import builtins
    import itertools
    import time
    requests = []
    class HTTP:
        def __init__(self, address, port, timeout):
            assert (address, port, timeout) == ("127.0.0.1", 80, 2)
        def request(self, method, path, headers):
            requests.append((method, path, headers))
        def getresponse(self):
            return SimpleNamespace(status=200)
        def close(self):
            pass
    def proc_open(path, *args, **kwargs):
        assert not args and not kwargs  # No write modes, no filesystem state API.
        assert path in ("/proc/net/tcp", "/proc/net/tcp6")
        if path.endswith("6"):
            return io.StringIO("header\n")
        return io.StringIO("header\n" + "".join(
            f"0: {listener}:{port:04X} 00000000:0000 0A\n" for port in (80, 18090)))
    monkeypatch.setattr(web.http.client, "HTTPConnection", HTTP)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 80))])
    monkeypatch.setattr(builtins, "open", proc_open)
    clock = itertools.count(0, 100)
    monkeypatch.setattr(time, "monotonic", lambda: next(clock))
    source = web._GUEST_PROBE.replace("HOSTS", repr(config()["hostnames"])).replace("FORWARD_PORT", "18090")
    if listener == "0100007F":
        exec(compile(source, "<guest-readiness>", "exec"), {})
        assert len(requests) == 2
        assert all(r[:2] == ("GET", "/") for r in requests)
    else:
        with pytest.raises(AssertionError, match="not IPv4 loopback"):
            exec(compile(source, "<guest-readiness>", "exec"), {})
        assert not requests


def test_forward_request_failure_closes_owned_connection(fake_attach, monkeypatch):
    guest, _ = fake_attach
    def fail(*args):
        raise OSError("forward denied")
    monkeypatch.setattr(guest.transport, "request_port_forward", fail)
    with pytest.raises(OSError, match="forward denied"):
        web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config())
    assert guest.closes == 1
    assert not guest.transport.cancels and not guest.ssh.commands


@pytest.mark.parametrize("domain", ["evil.test", "osworld.test", "*.gitlab.osworld.test",
                                     "gitlab.osworld.test.evil", "GITLAB.osworld.test", "", 1])
def test_gitlab_domain_is_scoped_to_configured_suffix(domain):
    with pytest.raises(ValueError, match="gitlab_domain"):
        web.load_config(config(gitlab_domain=domain))


def test_gitlab_config_adds_exact_base_not_wildcard_hosts():
    cfg = web.load_config(config(host_port=18092, gitlab_domain="gitlab.osworld.test"))
    assert cfg.host_port == 18092
    assert cfg.hostnames[-1] == "gitlab.osworld.test"
    assert web.load_config(config()).gitlab_domain is None
    with pytest.raises(ValueError):
        web.load_config(config(gitlab_domain="gitlab.osworld.test",
                               hostnames=["project.gitlab.osworld.test"]))
    text = web.hosts_text("127.0.0.1 localhost\n", cfg)
    assert "127.0.0.1\tgitlab.osworld.test" in text and "*" not in text


@pytest.mark.parametrize("name", ["gitlab.osworld.test", "user-abc123.gitlab.osworld.test",
                                   "unique-page.gitlab.osworld.test.", "A.B.GITLAB.OSWORLD.TEST",
                                   b"bytes.gitlab.osworld.test"])
def test_process_resolver_maps_only_gitlab_dns_names(monkeypatch, name):
    seen = []
    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))]
    monkeypatch.setattr(socket, "getaddrinfo", original)
    restore = web.install_process_resolution(config(gitlab_domain="gitlab.osworld.test"))
    result = socket.getaddrinfo(name, "http", family=socket.AF_UNSPEC,
                                type=socket.SOCK_STREAM, proto=6, flags=socket.AI_CANONNAME)
    assert result[0][4][0] == "127.0.0.1"
    assert seen == [(("127.0.0.1", "http"), {"family": socket.AF_UNSPEC,
                     "type": socket.SOCK_STREAM, "proto": 6, "flags": socket.AI_CANONNAME})]
    restore()
    assert socket.getaddrinfo is original


@pytest.mark.parametrize("name", ["mailhub.osworld.test", "example.com", "gitlab.osworld.test.evil",
                                   "notgitlab.osworld.test", "xgitlab.osworld.test", None,
                                   "bad..gitlab.osworld.test", "https://gitlab.osworld.test",
                                   "a/gitlab.osworld.test", "*.gitlab.osworld.test", b"\xff.gitlab.osworld.test"])
def test_process_resolver_preserves_nonmatching_requests(monkeypatch, name):
    seen = []
    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return "unchanged"
    monkeypatch.setattr(socket, "getaddrinfo", original)
    restore = web.install_process_resolution(config(gitlab_domain="gitlab.osworld.test"))
    assert socket.getaddrinfo(name, 8080, socket.AF_INET6, socket.SOCK_DGRAM) == "unchanged"
    assert seen == [((name, 8080, socket.AF_INET6, socket.SOCK_DGRAM), {})]
    restore()


def test_resolver_idempotent_replacement_and_conditional_restore(monkeypatch):
    seen = []
    def original(host, *args, **kwargs):
        seen.append(host)
    monkeypatch.setattr(socket, "getaddrinfo", original)
    old_restore = web.install_process_resolution(config(gitlab_domain="gitlab.osworld.test"))
    first = socket.getaddrinfo
    web.install_process_resolution(config(gitlab_domain="gitlab.osworld.test"))
    assert socket.getaddrinfo is first
    new_restore = web.install_process_resolution(config(suffix="other.test",
        hostnames=["mailhub.other.test"], gitlab_domain="gitlab.other.test"))
    old_restore()  # Never remove a later installation.
    socket.getaddrinfo("x.gitlab.osworld.test", 80)
    socket.getaddrinfo("x.gitlab.other.test", 80)
    assert seen == ["x.gitlab.osworld.test", "127.0.0.1"]
    new_restore()
    assert socket.getaddrinfo is original
    web.install_process_resolution(config())
    assert socket.getaddrinfo is original


def test_gitlab_attach_starts_scoped_guest_dns_and_cleans_owned_resources(fake_attach, monkeypatch):
    guest, calls = fake_attach
    installed = []
    monkeypatch.setattr(web, "install_process_resolution", lambda cfg: installed.append(cfg.gitlab_domain))
    tunnel = web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw",
                        config(host_port=18092, gitlab_domain="gitlab.osworld.test"))
    assert installed == ["gitlab.osworld.test"]
    commands = guest.ssh.commands
    assert any("--unit=rv-web-gitlab-dns" in cmd and
               "/usr/local/lib/rv-web-dns.py --domain gitlab.osworld.test --listen 127.0.0.1 --port 53" in cmd
               for cmd in commands)
    assert any("s.st_uid == 0" in cmd for cmd in commands)
    assert any("--upstream 127.0.0.53" in cmd for cmd in commands)
    assert any(shlex.split(cmd)[-1] == web._GUEST_DNS_INSTALL for cmd in commands)
    assert any("rv-wildcard-readiness." in cmd for cmd in commands)
    assert not any("iptables" in cmd for cmd in commands)
    tunnel.close()
    assert any(shlex.split(cmd)[-1] == web._GUEST_DNS_RESTORE for cmd in commands)
    assert any("systemctl stop rv-web-gitlab-dns" in cmd for cmd in commands)
    assert guest.closes == 1


@pytest.mark.parametrize("fail_at,dns_started,route_started", [(2, False, False), (3, False, False),
                                                               (4, True, True), (5, True, True),
                                                               (6, True, True), (7, True, True)])
def test_gitlab_dns_failures_fail_closed_and_cleanup(fake_attach, monkeypatch, fail_at, dns_started, route_started):
    guest, _ = fake_attach
    monkeypatch.setattr(web, "install_process_resolution", lambda cfg: None)
    guest.ssh.fail_at = fail_at
    with pytest.raises(RuntimeError, match="guest command failed"):
        web.attach(SimpleNamespace(vm_ip="guest"), "key", "pw", config(gitlab_domain="gitlab.osworld.test"))
    commands = guest.ssh.commands
    assert any("systemctl stop rv-web-gitlab-dns" in cmd for cmd in commands) == dns_started
    assert any(shlex.split(cmd)[-1] == web._GUEST_DNS_RESTORE for cmd in commands) == route_started
    assert guest.closes == 1 and len(guest.transport.cancels) == 1


@pytest.mark.parametrize("symlink", [False, True])
def test_guest_resolver_setup_restores_original_without_touching_stub_target(tmp_path, monkeypatch, symlink):
    import pathlib
    original = 'nameserver 127.0.0.53\nsearch example.test\n'
    target = tmp_path / 'systemd-stub'
    target.write_text(original)
    resolv = tmp_path / 'resolv.conf'
    if symlink:
        resolv.symlink_to(target)
    else:
        resolv.write_text(original)
    backup = tmp_path / 'backup.json'
    real_path = pathlib.Path
    def translated(value):
        return {'/etc/resolv.conf': resolv, '/run/rv-web-gitlab-resolver.json': backup}.get(value, real_path(value))
    monkeypatch.setattr(pathlib, 'Path', translated)
    exec(compile(web._GUEST_DNS_INSTALL, '<guest-dns-setup>', 'exec'), {})
    assert not resolv.is_symlink() and resolv.read_text().startswith('nameserver 127.0.0.1\n')
    assert target.read_text() == original and backup.exists()
    with pytest.raises(FileExistsError):
        # The resolver is no longer the expected systemd stub; refusal may occur
        # earlier, so use the original contents without removing the ownership file.
        resolv.write_text(original)
        exec(compile(web._GUEST_DNS_INSTALL, '<guest-dns-repeat>', 'exec'), {})
    resolv.write_text('nameserver 127.0.0.1\noptions timeout:2 attempts:2\n# rv-web-gitlab-dns\n')
    exec(compile(web._GUEST_DNS_RESTORE, '<guest-dns-restore>', 'exec'), {})
    assert resolv.read_text() == original and resolv.is_symlink() == symlink
    assert target.read_text() == original and not backup.exists()
    exec(compile(web._GUEST_DNS_RESTORE, '<guest-dns-restore-again>', 'exec'), {})


def test_guest_resolver_refuses_nonstub_or_newer_configuration(tmp_path, monkeypatch):
    import pathlib
    resolv, backup = tmp_path / 'resolv.conf', tmp_path / 'backup.json'
    real_path = pathlib.Path
    monkeypatch.setattr(pathlib, 'Path', lambda value: {
        '/etc/resolv.conf': resolv, '/run/rv-web-gitlab-resolver.json': backup}.get(value, real_path(value)))
    resolv.write_text('nameserver 1.1.1.1\n')
    with pytest.raises(AssertionError, match='expected systemd-resolved'):
        exec(compile(web._GUEST_DNS_INSTALL, '<guest-dns>', 'exec'), {})
    assert not backup.exists()
    backup.write_text(json.dumps({'text': 'nameserver 127.0.0.53\n', 'symlink': None}))
    with pytest.raises(AssertionError, match='ownership changed'):
        exec(compile(web._GUEST_DNS_RESTORE, '<guest-dns>', 'exec'), {})
    assert resolv.read_text() == 'nameserver 1.1.1.1\n' and backup.exists()
