"""Offline tests of the guest scripts (mmcoding/guest) and DSH patches (mmcoding/profiles).

No VM, no network, no model: the guest modules are loaded by file path (they run inside the VM
without the package) and every external effect is replaced by a fake.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mmcoding import harnesses

ROOT = Path(__file__).resolve().parents[1]
GUEST = ROOT / "mmcoding" / "guest"
PROFILES = ROOT / "mmcoding" / "profiles"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"rv_guest_{name}", GUEST / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


desktop = _load("rv_desktop")
rv_run = _load("rv_run")


def _load_worker(monkeypatch=None, written=None):
    """worker_rv with fakes for the official worker modules (uploaded next to it in the VM)."""
    fake_worker = types.ModuleType("worker")
    fake_worker.run = lambda *a, **k: None
    fake_common = types.ModuleType("common")
    fake_common.error_kind = lambda exc: "agent_runtime"
    fake_common.scrub = lambda v, s=(): v
    fake_common.write_json = (lambda p, v: written.__setitem__(Path(p).name, v)) if written is not None \
        else (lambda p, v: None)
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, "worker", fake_worker)
        monkeypatch.setitem(sys.modules, "common", fake_common)
    else:
        sys.modules.setdefault("worker", fake_worker)
        sys.modules.setdefault("common", fake_common)
    return _load("worker_rv")


class FakeHarness:
    seen: dict = {}

    def __init__(self, **kwargs):
        FakeHarness.seen = kwargs


def _settings(harness, **kw):
    return {"harness": harness, "model": "policy-model", "reasoning_effort": "high", "call_budget": 5,
            "provider": "openai", **kw}


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepseek_harness", types.SimpleNamespace(DeepSeekHarness=FakeHarness))
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://x.example/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    for name in ("RV_LLM_BASE_URL", "RV_LLM_API_KEY", "RV_HOME", "DISPLAY", "OSWORLD_DSH_CODE",
                 "RV_VISION_MODEL", "RV_VISION_EFFORT", "RV_VISION_COORDS", "RV_VISION_THINKING"):
        monkeypatch.delenv(name, raising=False)
    written = {}
    module = _load_worker(monkeypatch, written)
    module.written = written
    return module


# ------------------------------------------------------------------ harness tables


def test_guest_harness_tables_equal_the_registry(monkeypatch):
    worker = _load_worker(monkeypatch)
    assert tuple(rv_run.HARNESSES) == tuple(worker.HARNESSES) == harnesses.HARNESSES
    assert set(rv_run.X_HARNESSES) == set(worker.X_HARNESSES) == set(harnesses.X_HARNESSES)
    assert rv_run.BROKER == harnesses.BROKER
    assert tuple(rv_run.BROKER_PROFILES) == harnesses.BROKER_PROFILES
    assert tuple(desktop.PROFILE_TOOLS) == harnesses.BROKER_PROFILES
    assert set(harnesses.BROKER.values()) <= set(desktop.PROFILE_TOOLS)
    assert desktop.GUARDED_PROFILES == {"act_guarded"}


def test_profiles_cover_every_harness():
    names = sorted(p.name for p in PROFILES.glob("*.patch.yml"))
    assert names == sorted([f"{h}.patch.yml" for h in harnesses.HARNESSES]
                           + ["ask_user.patch.yml", "common.patch.yml"])


# ------------------------------------------------------------------ patches


def test_common_patch_template_placeholders():
    common = (PROFILES / "common.patch.yml").read_text()
    lines = common.splitlines()
    assert lines.count("        maxRequestImageBytes: __RV_MAX_REQUEST_IMAGE_BYTES__") == 1
    assert lines.count("        models: __RV_MODELS__") == 1
    assert common.count("__RV_MAX_REQUEST_IMAGE_BYTES__") == 1 and common.count("__RV_MODELS__") == 1
    assert "      openai:" in lines and "displayName: OpenAI-compatible endpoint" in common
    assert "apiKeyEnv: RV_LLM_API_KEY" in common and "process.env.RV_LLM_BASE_URL" in common
    assert "- id:" not in common.split("models: __RV_MODELS__", 1)[1].split("# ---", 1)[0]  # no model list
    assert "INVALID_REQUEST, PI_AI_ERROR]" in common and "maxRetries: 5" in common
    assert "personaPrefix: You are a computer-use agent" in common and "mcp-report" in common


def test_harness_patches_tool_policy():
    body = {h: (PROFILES / f"{h}.patch.yml").read_text() for h in harnesses.HARNESSES}
    gui_tools = ("tool-bash", "tool-jobs", "tool-fs", "tool-fs-search")
    for tool in gui_tools:
        assert f"- id: {tool}" in body["gui"] and f"- id: {tool}" in body["os_helix"]
        assert f"- id: {tool}" not in body["mm_coding"] and f"- id: {tool}" not in body["cli"]
    assert body["gui"].count("disabled: true") == 4
    for h in harnesses.HARNESSES:
        assert ("mcp-desktop" in body[h]) == (h in harnesses.BROKER)
        assert ("serverName: vision" in body[h]) == harnesses.get(h).vision_tool
        assert ("serverName: helix" in body[h]) == harnesses.get(h).helix
    strip = lambda text: [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    assert strip(body["vision_tool"])[:2] == strip(body["cli"])
    assert "attachment-local" in body["cli"] and "- id: tool-todo" in body["os_helix"]
    ask = (PROFILES / "ask_user.patch.yml").read_text()
    assert "serverName: ask" in ask and "process.env.RV_HOME}/ask_user_mcp.py" in ask


def test_os_helix_patch_is_one_clean_list():
    lines = [l for l in (PROFILES / "os_helix.patch.yml").read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    tops = [l for l in lines if l.startswith("- ")]
    assert tops == ["- insert:", "- id: tool-bash", "- id: tool-jobs", "- id: tool-fs", "- id: tool-fs-search",
                    "- id: tool-todo", "- insert:"]
    text = "\n".join(lines)
    assert text.index("id: mcp-desktop") < text.index("id: tool-bash") < text.index("id: mcp-helix")


# ------------------------------------------------------------------ desktop broker


@pytest.mark.parametrize("combo,expected", [
    ("ctrl+s", (frozenset({"ctrl"}), "s", "ctrl+s")),
    ("Enter", (frozenset(), "Return", "Return")),
    ("ctrl+shift+T", (frozenset({"ctrl", "shift"}), "t", "ctrl+shift+T")),
    ("alt+f2", (frozenset({"alt"}), "F2", "alt+F2")),
    ("ctrl+`", (frozenset({"ctrl"}), "grave", "ctrl+grave")),
])
def test_parse_combo(combo, expected):
    assert desktop.parse_combo(combo) == expected


def test_parse_keys_rejects_bad_input():
    with pytest.raises(desktop.BadArgs):
        desktop.parse_keys("")
    with pytest.raises(desktop.BadArgs):
        desktop.parse_combo("s+ctrl")
    with pytest.raises(desktop.BadArgs):
        desktop.parse_keys(" ".join(["a"] * 21))


def test_combo_refusals_depend_on_focus():
    alt_f2 = desktop.parse_combo("alt+F2")
    assert desktop.combo_refusal(alt_f2[0], alt_f2[1], "gedit Gedit")
    f12 = desktop.parse_combo("F12")
    assert desktop.combo_refusal(f12[0], f12[1], "google-chrome Google-chrome")
    assert desktop.combo_refusal(f12[0], f12[1], "libreoffice-calc") is None
    term = desktop.parse_combo("ctrl+`")
    assert desktop.combo_refusal(term[0], term[1], "code Code")
    shift_c = desktop.parse_combo("ctrl+shift+c")
    assert desktop.combo_refusal(shift_c[0], shift_c[1], "libreoffice-writer") is None
    macro = desktop.parse_combo("alt+F11")
    assert desktop.combo_refusal(macro[0], macro[1], "libreoffice-calc")


def test_focus_refusal_terminals_and_consoles():
    assert desktop.focus_refusal("gnome-terminal-server Gnome-terminal", "user@vm: ~")
    assert desktop.focus_refusal("xterm XTerm", "")
    assert desktop.focus_refusal("google-chrome", "DevTools - localhost:8000")
    assert desktop.focus_refusal("gedit Gedit", "notes.txt") is None
    assert desktop.focus_refusal("thunderbird", "Inbox") is None


def test_broker_profiles_and_validation(tmp_path):
    guarded = desktop.Broker("act_guarded", str(tmp_path / "a.jsonl"), 0)
    observe = desktop.Broker("observe", str(tmp_path / "b.jsonl"), 0)
    act = desktop.Broker("act", str(tmp_path / "c.jsonl"), 0)
    actions = {"screenshot", "click", "move", "drag", "scroll", "type_text", "key", "wait"}
    assert set(guarded.tools) == set(act.tools) == actions and observe.tools == ["screenshot"]
    assert guarded.guarded and not act.guarded and not observe.guarded
    with pytest.raises(ValueError):
        desktop.Broker("observe_click", str(tmp_path / "d.jsonl"), 0)
    reply = observe.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                            "params": {"name": "click", "arguments": {"x": 1, "y": 1}}})
    assert reply["result"]["isError"] and "unknown tool" in reply["result"]["content"][0]["text"]
    reply = guarded.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {"name": "click", "arguments": {"x": 5000, "y": 1}}})
    assert reply["result"]["isError"] and "outside the screen" in reply["result"]["content"][0]["text"]
    listed = guarded.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]
    assert all("inputSchema" in t for t in listed) and [t["name"] for t in listed] == list(desktop.TOOLS)
    init = act.handle({"jsonrpc": "2.0", "id": 4, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "rv-desktop-act"
    log = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "bad_args"


def _fake_desk(monkeypatch, broker, typed, focus_class="gnome-terminal-server Gnome-terminal",
               pointer_class="gnome-terminal-server Gnome-terminal"):
    monkeypatch.setattr(broker.desk, "active_window", lambda: {"class": focus_class, "title": "user@vm: ~"})
    monkeypatch.setattr(broker.desk, "window_under_pointer", lambda: {"class": pointer_class, "title": ""})
    monkeypatch.setattr(broker.desk, "xdotool", lambda *a, **k: typed.append(a))
    monkeypatch.setattr(broker, "_shot_content", lambda text: [{"type": "text", "text": text}])


def test_gui_broker_allows_terminals_os_helix_broker_refuses(tmp_path, monkeypatch):
    brokers = {h: desktop.Broker(harnesses.BROKER[h], str(tmp_path / f"{h}.jsonl"), 0) for h in ("gui", "os_helix")}
    typed = []
    for broker in brokers.values():
        _fake_desk(monkeypatch, broker, typed)
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "type_text", "arguments": {"text": "ls /tmp_workspace"}}}
    refused = brokers["os_helix"].handle(call)["result"]
    assert refused["isError"] and "terminal" in refused["content"][0]["text"] and not typed
    click = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "click", "arguments": {"x": 10, "y": 10}}}
    refused = brokers["os_helix"].handle(click)["result"]
    assert refused["isError"] and "terminal window" in refused["content"][0]["text"]
    assert [a[0] for a in typed] == ["mousemove"]  # the pointer moved, the click was refused
    typed.clear()
    allowed = brokers["gui"].handle(call)["result"]
    assert not allowed.get("isError") and typed and typed[0][0] == "type"
    log = [json.loads(line) for line in (tmp_path / "os_helix.jsonl").read_text().splitlines()]
    assert [r["status"] for r in log] == ["refused", "refused"]


def test_guarded_broker_refuses_console_key_combos(tmp_path, monkeypatch):
    broker = desktop.Broker("act_guarded", str(tmp_path / "g.jsonl"), 0)
    typed = []
    _fake_desk(monkeypatch, broker, typed, focus_class="google-chrome Google-chrome")
    call = lambda keys: broker.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                       "params": {"name": "key", "arguments": {"keys": keys}}})["result"]
    assert call("ctrl+shift+i")["isError"] and not typed
    assert not call("ctrl+l Return").get("isError") and typed == [("key", "--delay", "60", "ctrl+l", "Return")]


# ------------------------------------------------------------------ worker


def test_budget_counts_only_root_step_start(monkeypatch):
    worker = _load_worker(monkeypatch)
    ev = lambda t, sid="s1": SimpleNamespace(method="session.event", payload={"sessionId": sid, "event": {"type": t}})
    assert worker.count_model_call(ev("step/start"), "s1")
    assert not worker.count_model_call(ev("step/end"), "s1")
    assert not worker.count_model_call(ev("step/start", "child"), "s1")
    assert not worker.count_model_call(SimpleNamespace(method="session.status", payload={}), "s1")


@pytest.mark.parametrize("harness", harnesses.HARNESSES)
def test_worker_patch_composition_per_harness(worker, tmp_path, monkeypatch, harness):
    shims = []

    class FakeShim:
        def __init__(self, base, key, attempt, capture):
            shims.append((base, key, Path(attempt), capture))
            self.url = "http://127.0.0.1:1/v1"

    monkeypatch.setitem(sys.modules, "helix_shim", types.SimpleNamespace(HelixShim=FakeShim))
    monkeypatch.setattr(sys, "path", list(sys.path))
    worker.rv_harness_class(_settings(harness), {"call_offset": 2}, tmp_path)()
    kw = FakeHarness.seen
    names = [Path(p).name for p in kw["patches"]]
    assert names == ["common.patch.yml", f"{harness}.patch.yml", "call_budget.patch.yml"]
    assert all(Path(p).parent == worker.HERE / "patches" for p in kw["patches"][:2])
    assert kw["provider"] == "openai" and kw["profile"] == "sdk" and kw["reasoning_effort"] == "high"
    assert kw["request_timeout_seconds"] is None
    assert worker.written["call_budget.json"] == {"call_budget": 5, "call_offset": 2}
    plugin = (tmp_path / "call_budget.patch.yml").read_text()
    assert "id: rv-call-budget" in plugin and str(worker.HERE / "call_budget.mjs") in plugin
    assert os.environ["RV_HOME"] == str(worker.HERE) and os.environ["RV_LLM_API_KEY"] == "k"
    assert (os.environ.get("DISPLAY") == ":0") == (harness in harnesses.X_HARNESSES)
    if harness == "os_helix":
        (base, key, attempt, capture), = shims
        assert (base, key, attempt) == ("https://x.example/v1", "k", tmp_path) and callable(capture)
        assert os.environ["RV_LLM_BASE_URL"] == "http://127.0.0.1:1/v1"
    else:
        assert not shims and os.environ["RV_LLM_BASE_URL"] == "https://x.example/v1"


def test_worker_adds_ask_patch_and_rejects_unknown_harness(worker, tmp_path):
    worker.rv_harness_class(_settings("cli"), {"ask_enabled": True}, tmp_path)()
    assert [Path(p).name for p in FakeHarness.seen["patches"]] == [
        "common.patch.yml", "cli.patch.yml", "ask_user.patch.yml", "call_budget.patch.yml"]
    for bad in ("GUI", "unknown", ""):
        with pytest.raises(ValueError):
            worker.rv_harness_class(_settings(bad), {}, tmp_path)


def test_worker_exports_vision_model_settings(worker, tmp_path):
    worker.rv_harness_class(_settings("vision_tool", model="policy-model"), {}, tmp_path)()
    assert os.environ["RV_VISION_MODEL"] == "policy-model" and os.environ["RV_VISION_EFFORT"] == "high"
    assert os.environ["DISPLAY"] == ":0" and os.environ["RV_LLM_API_KEY"] == "k"
    worker.rv_harness_class(_settings("vision_tool", vision_model="vision-model", vision_reasoning_effort="off",
                                      vision_coords="pixel", vision_thinking="on"), {}, tmp_path)()
    assert os.environ["RV_VISION_MODEL"] == "vision-model" and os.environ["RV_VISION_EFFORT"] == ""
    assert os.environ["RV_VISION_COORDS"] == "pixel" and os.environ["RV_VISION_THINKING"] == "on"


def test_worker_continues_after_output_length_stop(monkeypatch, tmp_path):
    @dataclass
    class Result:
        session_id: str
        final_response: str
        finish_reason: str
        events: list
        notifications: list

    script = ["max-tokens", "max-tokens", "completed"]
    prompts = []

    class Harness:
        def __init__(self, **kwargs):
            pass

        def run(self, input, *, session_id=None, on_notification=None):  # noqa: A002
            prompts.append(input)
            on_notification(SimpleNamespace(method="session.event",
                                            payload={"sessionId": session_id, "event": {"type": "step/start"}}))
            reason = script[len(prompts) - 1]
            return Result(session_id, reason, reason, [{"type": "turn/end"}], [])

    monkeypatch.setitem(sys.modules, "deepseek_harness", types.SimpleNamespace(DeepSeekHarness=Harness))
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://x")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DISPLAY", ":9")
    worker = _load_worker(monkeypatch)
    cls = worker.rv_harness_class(_settings("cli", call_budget=10), {}, tmp_path)
    result = cls().run("task", session_id="s1")
    assert os.environ["DISPLAY"] == ":0"
    assert result.finish_reason == "completed" and len(result.events) == 3
    assert prompts == ["task", worker.CONTINUE_PROMPT, worker.CONTINUE_PROMPT]
    script[:] = ["max-tokens"] * 10
    prompts.clear()
    result = cls().run("task", session_id="s1")
    assert result.finish_reason == "max-tokens" and len(prompts) == 1 + worker.MAX_CONTINUATIONS
    # No continuation once the (cumulative) budget is used up.
    prompts.clear()
    worker.rv_harness_class(_settings("cli", call_budget=10), {"call_offset": 9}, tmp_path)().run("task", session_id="s")
    assert prompts == ["task"]
    worker.rv_harness_class(_settings("gui", call_budget=10), {}, tmp_path)()
    assert "DISPLAY" not in os.environ


def test_worker_main_reports_budget_exhaustion_and_vision_endpoint(monkeypatch, tmp_path):
    written = {}
    worker = _load_worker(monkeypatch, written)
    for package_env in ("RV_VISION_BASE_URL", "RV_VISION_API_KEY"):
        monkeypatch.delenv(package_env, raising=False)
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: worker.SDK_VERSION)
    monkeypatch.setattr(worker.signal, "signal", lambda *a: None)
    monkeypatch.setattr(worker.signal, "alarm", lambda *a: None)
    monkeypatch.setattr(worker.os, "umask", lambda m: 0)
    monkeypatch.setattr(worker, "rv_harness_class", lambda *a: object)

    def run(config, harness_class):
        raise RuntimeError("RV_BUDGET_EXHAUSTED: Model call budget exhausted (5 calls); TIMEOUT")

    monkeypatch.setattr(worker.official, "run", run)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"attempt": str(tmp_path), "api_key": "k", "settings": {"timeout": 60},
                                   "rv": {"vision_endpoint": {"base_url": "https://v.example/v1",
                                                              "api_key": "vk"}}}))
    monkeypatch.setattr(sys, "argv", ["worker_rv.py", str(request)])
    worker.main()
    status = written["status.json"]
    assert status["status"] == "error" and status["budget_exhausted"] is True
    assert not request.exists()
    assert os.environ["RV_VISION_BASE_URL"] == "https://v.example/v1" and os.environ["RV_VISION_API_KEY"] == "vk"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_call_budget_plugin_stops_before_the_next_request(tmp_path):
    (tmp_path / "call_budget.json").write_text(json.dumps({"call_budget": 3, "call_offset": 1}))
    driver = tmp_path / "driver.mjs"
    driver.write_text(f"""
import * as plugin from {json.dumps((GUEST / 'call_budget.mjs').as_uri())};
const handlers = {{}};
await plugin.apply({{ on: (name, fn) => {{ handlers[name] = fn; }} }});
const agent = n => ({{ session: {{ events: Array.from({{ length: n }}, () => ({{ type: 'step/end' }})) }} }});
const out = [];
for (const n of [0, 1, 2]) {{
  try {{ await handlers['agent/request']({{ agent: agent(n) }}, async () => 'sent'); out.push('sent'); }}
  catch (e) {{ out.push(e.message); }}
}}
console.log(JSON.stringify({{ plugin: plugin.name, hooks: Object.keys(handlers), out }}));
""")
    env = {**os.environ, "OSWORLD_DSH_ATTEMPT": str(tmp_path)}
    proc = subprocess.run(["node", str(driver)], capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["plugin"] == "rv-call-budget" and report["hooks"] == ["agent/request"]
    assert report["out"][:2] == ["sent", "sent"] and report["out"][2].startswith("RV_BUDGET_EXHAUSTED")
    budget = json.loads((tmp_path / "budget.json").read_text())
    assert budget["exhausted"] and budget["model_calls"] == 3 and budget["budget"] == 3


# ------------------------------------------------------------------ rv-run sandbox


def test_x_harness_sandbox_exposes_only_the_display():
    plain = rv_run.sandbox_props([], "gui")
    xprops = rv_run.sandbox_props([], "cli")
    assert "TemporaryFileSystem=/tmp/.X11-unix:ro /tmp/.ICE-unix:ro" in plain
    assert "TemporaryFileSystem=/tmp/.ICE-unix:ro" in xprops
    env_plain = next(p for p in plain if p.startswith("Environment="))
    env_x = next(p for p in xprops if p.startswith("Environment="))
    assert "DISPLAY" not in env_plain and "DISPLAY=:0" in env_x
    strip = lambda props: [p for p in props if not p.startswith(("TemporaryFileSystem=", "Environment="))]
    assert strip(plain) == strip(xprops)
    assert "-/run/user/1000/gdm" in next(p for p in xprops if p.startswith("InaccessiblePaths="))
    assert rv_run.sandbox_props([]) == plain  # default harness: gui


@pytest.mark.parametrize("harness", harnesses.HARNESSES)
def test_sandbox_props_per_harness(harness):
    props = rv_run.sandbox_props(["/secret/eval"], harness)
    env = next(p for p in props if p.startswith("Environment="))
    assert ("DISPLAY=:0" in env) == harnesses.get(harness).x_access
    assert "-/secret/eval" in next(p for p in props if p.startswith("InaccessiblePaths="))
    assert "NetworkNamespacePath=/run/netns/rv" in props and "NoNewPrivileges=yes" in props
    assert "PrivateUsers=yes" in props and "User=user" in props


def test_network_guard_ca_reaches_the_sandbox_env(monkeypatch):
    env = lambda: next(p for p in rv_run.sandbox_props([], "cli") if p.startswith("Environment="))
    monkeypatch.setattr(rv_run.Path, "is_file", lambda self: False)
    assert "SSL_CERT_FILE" not in env()
    monkeypatch.setattr(rv_run.Path, "is_file", lambda self: self.name == "rv-answer-proxy.crt")
    text = env()
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in text and "PIP_CERT=" in text
    assert "NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/rv-answer-proxy.crt" in text


def test_forward_rules_drop_private_networks_except_resolvers():
    rules = rv_run.forward_rules(["8.8.8.8", "10.0.0.1"])
    assert rules[:4] == rv_run.dns_rules(["8.8.8.8", "10.0.0.1"])
    assert ["-d", "10.0.0.0/8", "-j", "DROP"] in rules and rules[-1] == ["-j", "ACCEPT"]


def test_rv_run_argument_parsing(monkeypatch):
    seen = {}
    monkeypatch.setattr(rv_run.os, "geteuid", lambda: 0)
    for name in ("cmd_setup", "cmd_worker", "cmd_stop", "cmd_service", "cmd_collect", "cmd_daemon", "cmd_probe"):
        monkeypatch.setattr(rv_run, name, lambda args, name=name: seen.update({name: args}))
    rv_run.main(["setup", "--harness", "os_helix", "--hidden", "/a", "--hidden", "/b"])
    args = seen["cmd_setup"]
    assert args.harness == "os_helix" and args.hidden == ["/a", "/b"] and args.broker is None
    rv_run.main(["setup", "--harness", "mm_coding", "--broker", "none"])
    assert seen["cmd_setup"].broker == "none"
    rv_run.main(["worker", "--unit", "osworld-dsh-x", "--attempt", "/a", "--runtime-max", "60"])
    assert seen["cmd_worker"].runtime_max == 60
    rv_run.main(["service", "--name", "web", "--port", "8000", "--cwd", "/tmp", "--", "python3", "-m", "http.server"])
    assert seen["cmd_service"].argv == ["python3", "-m", "http.server"]
    rv_run.main(["probe", "--", "id", "-u"])
    assert seen["cmd_probe"].argv == ["id", "-u"]
    for argv in (["setup", "--harness", "unknown"], ["setup"], ["setup", "--harness", "gui", "--broker", "other"],
                 ["worker", "--unit", "u"], ["unknown"]):
        with pytest.raises(SystemExit):
            rv_run.main(argv)
    monkeypatch.setattr(rv_run.os, "geteuid", lambda: 1000)
    with pytest.raises(SystemExit, match="root"):
        rv_run.main(["collect"])


def test_setup_desktop_uses_the_harness_broker(monkeypatch):
    runs, state = [], {"harness": "gui", "broker": "act"}
    monkeypatch.setattr(rv_run, "unit_active", lambda unit: False)
    monkeypatch.setattr(rv_run, "systemd_run", lambda unit, props, argv, extra=(): runs.append(argv))
    monkeypatch.setattr(rv_run, "load_state", lambda: state)
    monkeypatch.setattr(rv_run.os.path, "exists", lambda p: True)
    assert rv_run.setup_desktop("os_helix") == {"desktop_broker": "act_guarded"}
    assert runs[-1][runs[-1].index("--profile") + 1] == "act_guarded"
    monkeypatch.setattr(rv_run, "sh", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(rv_run.os, "unlink", lambda p: None)
    assert rv_run.setup_desktop("cli") == {"desktop_broker": None}
    assert rv_run.setup_desktop("gui", None) == {"desktop_broker": None}
    assert rv_run.setup_x_relay("gui") == {"x_relay": None}


def test_rv_run_dpkg_install_waits_for_warmup_lock(monkeypatch):
    replies = [SimpleNamespace(returncode=2, stderr="dpkg: error: dpkg frontend lock was locked by another process"),
               SimpleNamespace(returncode=0, stderr="")]
    calls = []
    monkeypatch.setattr(rv_run, "sh", lambda cmd, check=True, timeout=300: calls.append(cmd) or replies.pop(0))
    monkeypatch.setattr(rv_run.time, "sleep", lambda s: None)
    rv_run.dpkg_install([Path("/x/auditd.deb")])
    assert len(calls) == 2 and "dpkg -i /x/auditd.deb" in calls[0]
    monkeypatch.setattr(rv_run, "sh", lambda cmd, check=True, timeout=300: SimpleNamespace(returncode=1, stderr="bad deb"))
    with pytest.raises(RuntimeError, match="bad deb"):
        rv_run.dpkg_install([Path("/x/auditd.deb")])


def test_task_relay_port_selection():
    ss = "\n".join([
        "LISTEN 0 5 0.0.0.0:4000 0.0.0.0:*", "LISTEN 0 128 127.0.0.1:631 0.0.0.0:*",
        "LISTEN 0 5 0.0.0.0:5000 0.0.0.0:*", "LISTEN 0 50 [::1]:8000 [::]:*", "LISTEN 0 50 [::]:6000 [::]:*",
        "LISTEN 0 50 10.201.0.5:7000 0.0.0.0:*", "LISTEN 0 50 0.0.0.0:41234 0.0.0.0:*",
        "LISTEN 0 50 127.0.0.1:5001 0.0.0.0:*", "LISTEN 0 4096 127.0.0.53%lo:53 0.0.0.0:*",
    ])
    assert rv_run.relay_candidates(ss, {5001}) == {4000: "127.0.0.1", 8000: "[::1]"}
    assert rv_run.FS_MASKS and all(p.startswith("/usr/share/code/") for p in rv_run.FS_MASKS)


# ------------------------------------------------------------------ report / ask_user MCP


def test_report_infeasible_tool_records(tmp_path, monkeypatch, capsys):
    report = _load("report_mcp")
    monkeypatch.setenv("OSWORLD_DSH_ATTEMPT", str(tmp_path))
    report.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    report.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "report_infeasible", "arguments": {"reason": ""}}})
    report.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "report_infeasible", "arguments": {"reason": "no GUI input"}}})
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert replies[0]["result"]["tools"][0]["name"] == "report_infeasible"
    assert replies[1]["result"]["isError"] and not replies[2]["result"].get("isError")
    entries = [json.loads(line) for line in (tmp_path / "infeasible.jsonl").read_text().splitlines()]
    assert [e["reason"] for e in entries] == ["no GUI input"]


def test_ask_user_forwards_to_the_official_bridge(tmp_path, monkeypatch, capsys):
    asked = []
    helpers = types.ModuleType("helpers")
    helpers.ask = lambda attempt, question: asked.append((attempt, question)) or "blue"
    monkeypatch.setitem(sys.modules, "helpers", helpers)
    monkeypatch.setenv("OSWORLD_DSH_ATTEMPT", str(tmp_path))
    ask = _load("ask_user_mcp")
    ask.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    ask.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "ask_user", "arguments": {"question": "Which colour?"}}})
    ask.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "other", "arguments": {}}})
    ask.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [t["name"] for t in replies[0]["result"]["tools"]] == ["ask_user"]
    assert replies[1]["result"]["content"][0]["text"] == "blue" and asked == [(tmp_path, "Which colour?")]
    assert replies[2]["result"]["isError"] and len(replies) == 3


# ------------------------------------------------------------------ OS-Helix


helix = _load("helix_shim")


def _image(n):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{n}"}}


def test_helix_rewrite_keeps_ten_latest_images_and_initial_screenshot():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    messages += [{"role": "tool", "content": [{"type": "text", "text": f"r{i}"}, _image(i)]} for i in range(12)]
    body = {"messages": messages}
    initial = _image("init")
    total, kept = helix.rewrite(body, initial)
    assert (total, kept) == (13, 10) and body["parallel_tool_calls"] is False
    user = body["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "task"} and user[-1]["type"] == "text"  # initial image omitted
    images = [p for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]
              if p.get("type") == "image_url"]
    assert [p["image_url"]["url"][-2:] for p in images] == [f",{i}"[-2:] for i in range(2, 12)]
    small = {"messages": [{"role": "user", "content": [{"type": "text", "text": "t"}]}]}
    assert helix.rewrite(small, initial) == (1, 1) and small["messages"][0]["content"][-1] is initial


def test_helix_static_token_estimate():
    assert helix.est_tokens("abc") == 2 and helix.est_tokens("\u4e2d\u6587") == 3
    assert helix.est_tokens([_image(1), {"type": "image", "data": "x" * 10 ** 6}]) == 2 * helix.IMAGE_TOKENS + 1
    assert helix.WINDOW == 262144 and helix.CAP == 65536 and helix.SAFETY == 8192


class _Upstream:
    def __init__(self, obj):
        self.status, self.headers, self.obj = 200, {"Content-Type": "application/json"}, obj

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return json.dumps(self.obj).encode()


def _shim(monkeypatch, tmp_path, responses):
    """HelixShim with the HTTP server and the upstream replaced by fakes (no sockets)."""
    holder, sent = {}, []

    class Server:
        def __init__(self, address, handler):
            holder["handler"] = handler
            self.server_address = ("127.0.0.1", 1)

        def serve_forever(self):
            pass

    class Opener:
        def open(self, request, timeout):
            sent.append(json.loads(request.data))
            return _Upstream(responses.pop(0))

    monkeypatch.setattr(helix, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(helix.urllib.request, "build_opener", lambda *a: Opener())
    shim = helix.HelixShim("https://up.example/v1/", "k", tmp_path, lambda: (b"PNG", (0, 0)))

    def post(body):
        handler = object.__new__(holder["handler"])
        raw = json.dumps(body).encode()
        handler.path, handler.headers = "/v1/chat/completions", {"Content-Length": str(len(raw))}
        handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
        handler.request_version, handler.requestline, handler.command = "HTTP/1.1", "POST", "POST"
        handler.client_address = ("127.0.0.1", 0)
        handler.do_POST()
        return json.loads(handler.wfile.getvalue().split(b"\r\n\r\n", 1)[1])

    return shim, post, sent


def test_helix_shim_recomputes_max_tokens_and_keeps_one_tool_call(monkeypatch, tmp_path):
    reply = lambda prompt: {"choices": [{"message": {"tool_calls": [{"id": "a"}, {"id": "b"}]}}],
                            "usage": {"prompt_tokens": prompt, "completion_tokens": 100}}
    shim, post, sent = _shim(monkeypatch, tmp_path, [reply(190000), reply(60000), reply(1000)])
    assert shim.url == "http://127.0.0.1:1/v1"
    tools = [{"type": "function", "function": {"name": "run_code"}}]
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    out = post({"messages": messages, "tools": tools, "max_tokens": 1})
    assert out["choices"][0]["message"]["tool_calls"] == [{"id": "a"}]
    first = sent[0]
    est = helix.est_tokens(first["messages"]) + helix.est_tokens(tools) + helix.est_tokens(None)
    assert first["max_tokens"] == min(helix.CAP, helix.WINDOW - est - helix.SAFETY) == helix.CAP
    assert first["parallel_tool_calls"] is False and first["messages"][1]["content"][-1]["type"] == "image_url"
    # The next request of the same session starts from the reported usage.
    more = [*messages, {"role": "assistant", "content": "x" * 3000}]
    post({"messages": more, "tools": tools, "max_tokens": 1})
    est = 190000 + 100 + helix.est_tokens(sent[1]["messages"][2:])
    assert sent[1]["max_tokens"] == helix.WINDOW - est - helix.SAFETY < helix.CAP
    # Never lowered below what pi-ai asked for.
    post({"messages": [*more, {"role": "user", "content": "y"}], "tools": tools, "max_tokens": 250000})
    assert sent[2]["max_tokens"] == 250000
    log = [json.loads(line) for line in (tmp_path / "helix_requests.jsonl").read_text().splitlines()]
    assert [r["est_source"] for r in log] == ["static", "usage", "usage"]
    assert all(r["n_tool_calls_provider"] == 2 and r["n_tool_calls_allowed"] == 1 for r in log)


def test_helix_shim_leaves_requests_without_tools_alone(monkeypatch, tmp_path):
    shim, post, sent = _shim(monkeypatch, tmp_path, [{"choices": [{"message": {"content": "ok"}}]}])
    post({"messages": [{"role": "user", "content": "q"}], "max_tokens": 7})
    assert sent[0]["max_tokens"] == 7 and shim.state is None


def test_helix_code_tool_spec_and_errors(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(GUEST))
    code = _load("helix_code_mcp")
    code.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    code.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "other", "arguments": {}}})
    code.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "run_code", "arguments": {"language": "ruby", "code": "x"}}})
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    tool, = replies[0]["result"]["tools"]
    assert tool["name"] == "run_code" and tool["inputSchema"]["properties"]["timeout_s"]["maximum"] == 600
    assert replies[1]["result"]["isError"] and replies[2]["result"]["isError"]
    assert code.LOCK.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="already running"):
            code.execute({"language": "bash", "code": "true"})
    finally:
        code.LOCK.release()


# ------------------------------------------------------------------ Vision-Tool


def _png(path: Path, size=(40, 30), color=(200, 10, 10)) -> Path:
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


def _vision(tmp_path, monkeypatch):
    vision = _load("vision_mcp")
    monkeypatch.setenv("OSWORLD_DSH_ATTEMPT", str(tmp_path))
    monkeypatch.setenv("RV_VISION_MODEL", "kimi-vision")
    calls = []

    def fake_model(messages, profile=None):
        calls.append(messages)
        return {"answer": "A red square.", "usage": {"prompt_tokens": 10}, "finish_reason": "stop",
                "reasoning_chars": 5}

    monkeypatch.setattr(vision, "call_model", fake_model)
    return vision, calls


def _call(vision, capsys, arguments, msg_id=1):
    vision.handle({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                   "params": {"name": "ask_vision", "arguments": arguments}})
    return json.loads(capsys.readouterr().out.splitlines()[-1])["result"]


def test_ask_vision_lists_one_text_tool(tmp_path, monkeypatch, capsys):
    vision, _ = _vision(tmp_path, monkeypatch)
    vision.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tools = json.loads(capsys.readouterr().out)["result"]["tools"]
    assert [t["name"] for t in tools] == ["ask_vision"] and tools[0] == vision.TOOL
    assert tools[0]["inputSchema"]["required"] == ["question"]
    assert "remembers nothing between calls" in tools[0]["description"]
    assert vision.tool_spec() is vision.TOOL


def test_ask_vision_answers_with_text_only_and_logs(tmp_path, monkeypatch, capsys):
    vision, calls = _vision(tmp_path, monkeypatch)
    image = _png(tmp_path / "ref.png")
    result = _call(vision, capsys, {"question": "What is shown?", "images": [str(image)]})
    assert not result.get("isError")
    assert [c["type"] for c in result["content"]] == ["text"]
    assert result["content"][0]["text"].endswith("A red square.")
    system, user = calls[0]
    assert system["role"] == "system" and "cannot see" in system["content"]
    assert [part["type"] for part in user["content"]] == ["text", "image_url", "text"]
    assert user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user["content"][-1]["text"] == "Question: What is shown?"
    assert "What is shown?" not in system["content"]  # no task context beyond the question
    entry = json.loads((tmp_path / "vision_calls.jsonl").read_text().splitlines()[0])
    assert entry["status"] == "ok" and entry["answer"] == "A red square."
    assert (tmp_path / entry["saved"][0]).exists() and entry["image_info"][0]["size"] == [40, 30]
    assert "state" not in entry and "zooms" not in entry


def test_ask_vision_is_stateless(tmp_path, monkeypatch, capsys):
    vision, calls = _vision(tmp_path, monkeypatch)
    image = _png(tmp_path / "a.png")
    _call(vision, capsys, {"question": "colour?", "images": [str(image)]})
    _call(vision, capsys, {"question": "colour again?", "images": [str(image)]}, msg_id=2)
    assert calls[1][1]["content"][0]["text"].startswith("Image 1")
    assert "colour?" not in json.dumps(calls[1])
    assert "zoom" not in calls[0][0]["content"]


def test_ask_vision_screen_default_and_downscale(tmp_path, monkeypatch, capsys):
    from PIL import Image

    vision, calls = _vision(tmp_path, monkeypatch)
    buf = io.BytesIO()
    Image.new("RGB", (1920, 1080), (0, 0, 0)).save(buf, format="PNG")
    monkeypatch.setattr(vision, "capture_screen", lambda: (buf.getvalue(), (10, 20)))
    big = _png(tmp_path / "big.png", size=(4096, 4096))
    result = _call(vision, capsys, {"question": "Compare.", "images": ["screen", str(big)]})
    assert not result.get("isError")
    labels = [p["text"] for p in calls[0][1]["content"] if p["type"] == "text"]
    assert labels[0] == "Image 1: the current screen (1920x1080, the mouse cursor is drawn)."
    assert labels[1] == "Image 2: an image (2048x2048)." and str(big) not in "".join(labels)  # no paths
    text = result["content"][0]["text"]
    assert "cursor at (10, 20)" in text and "shown downscaled to 2048x2048" in text
    only_screen = _call(vision, capsys, {"question": "Where is the dock?"}, msg_id=2)
    assert not only_screen.get("isError") and len(calls) == 2


@pytest.mark.parametrize("arguments,needle", [
    ({"question": ""}, "question is required"),
    ({"question": "x" * 8001}, "longer than 8000"),
    ({"question": "x", "images": ["rel/path.png"]}, "absolute file path"),
    ({"question": "x", "images": ["/etc/passwd"]}, "only PNG"),
    ({"question": "x", "images": ["/nonexistent/a.png"]}, "cannot read"),
    ({"question": "x", "images": []}, "non-empty list"),
    ({"question": "x", "images": ["screen"] * 9}, "at most 8"),
])
def test_ask_vision_rejects_bad_arguments(tmp_path, monkeypatch, capsys, arguments, needle):
    vision, calls = _vision(tmp_path, monkeypatch)
    result = _call(vision, capsys, arguments)
    assert result["isError"] and needle in result["content"][0]["text"] and not calls
    entry = json.loads((tmp_path / "vision_calls.jsonl").read_text().splitlines()[-1])
    assert entry["status"] == "error"


def test_ask_vision_source_limits(tmp_path, monkeypatch):
    vision, _ = _vision(tmp_path, monkeypatch)
    big = tmp_path / "huge.png"
    big.write_bytes(b"\0" * (vision.MAX_SOURCE_BYTES + 1))
    with pytest.raises(vision.ToolError, match="20 MiB"):
        vision.load_source(str(big))
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(vision.ToolError, match="not a readable image"):
        vision.load_source(str(bad))
    wide = _png(tmp_path / "wide.png", size=(vision.MAX_SOURCE_DIMENSION + 1, 2))
    with pytest.raises(vision.ToolError, match="8192px"):
        vision.load_source(str(wide))
    monkeypatch.setattr(vision, "MAX_SOURCE_PIXELS", 100)
    with pytest.raises(vision.ToolError, match="100-pixel"):
        vision.load_source(str(_png(tmp_path / "px.png", size=(20, 20))))


def test_ask_vision_retries_then_reports_unavailable(tmp_path, monkeypatch, capsys):
    vision, _ = _vision(tmp_path, monkeypatch)
    monkeypatch.setattr(vision, "RETRY_WAITS", (0.01, 0.01))
    image = _png(tmp_path / "a.png")
    attempts = []

    def flaky(messages, profile=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError("slow")
        return {"answer": "ok", "usage": None, "finish_reason": "stop", "reasoning_chars": 0}

    monkeypatch.setattr(vision, "call_model", flaky)
    assert not _call(vision, capsys, {"question": "q", "images": [str(image)]}).get("isError")
    monkeypatch.setattr(vision, "call_model", lambda m, p=None: (_ for _ in ()).throw(TimeoutError("down")))
    result = _call(vision, capsys, {"question": "q", "images": [str(image)]}, msg_id=2)
    assert result["isError"] and "unavailable" in result["content"][0]["text"]


def _noise_png(path: Path, size=(1920, 1080)) -> Path:
    from PIL import Image

    Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3)).save(path)
    return path


def test_ask_vision_reencodes_large_images_and_caps_payload(tmp_path, monkeypatch, capsys):
    vision, calls = _vision(tmp_path, monkeypatch)
    noisy = _noise_png(tmp_path / "photo.png")
    assert noisy.stat().st_size > 4 * 2**20
    data, mime, size, shown = vision.normalize(noisy.read_bytes())
    assert mime == "image/jpeg" and size == shown == (1920, 1080)
    small = _png(tmp_path / "ui.png", size=(1920, 1080))
    data, mime, _, _ = vision.normalize(small.read_bytes())
    assert mime == "image/png" and data == small.read_bytes()  # small PNGs pass unchanged
    monkeypatch.setattr(vision, "MAX_CALL_BYTES", 1024)
    result = _call(vision, capsys, {"question": "q", "images": [str(noisy)]})
    assert result["isError"] and "per call" in result["content"][0]["text"] and not calls


def test_ask_vision_v2_prompt_is_perception_only(monkeypatch):
    vision = _load("vision_mcp")
    monkeypatch.setenv("RV_VISION_MODEL", "qwen-vl-test")
    monkeypatch.delenv("RV_VISION_COORDS", raising=False)
    monkeypatch.delenv("RV_VISION_THINKING", raising=False)
    monkeypatch.delenv("RV_VISION_EFFORT", raising=False)
    qwen = vision.answerer_profile()
    assert qwen["coords"] == "norm1000" and qwen["thinking"] == "off"
    prompt = vision.system_prompt(qwen)
    assert vision.OUT_OF_SCOPE in prompt and "general knowledge" in prompt and "premise" in prompt
    assert "0-1000" in prompt and "<pt>x,y</pt>" in prompt and "zoom" not in prompt
    monkeypatch.setenv("RV_VISION_EFFORT", "high")
    body = vision.request_body([{"role": "user", "content": "x"}], qwen)
    assert body["enable_thinking"] is False and "reasoning_effort" not in body and "tools" not in body
    assert body["model"] == "qwen-vl-test" and body["stream"] is True and body["max_tokens"] == 16384
    kimi = vision.answerer_profile("Kimi-vision")
    assert kimi["coords"] == "pixel" and "pixels of the image as you see it" in vision.system_prompt(kimi)
    body = vision.request_body([{"role": "user", "content": "x"}], kimi)
    assert body["reasoning_effort"] == "high" and "enable_thinking" not in body
    other = vision.answerer_profile("some-model")
    assert other["coords"] == "pixel" and other["thinking"] is None
    monkeypatch.setenv("RV_VISION_COORDS", "pixel")
    monkeypatch.setenv("RV_VISION_THINKING", "on")
    forced = vision.answerer_profile()
    assert forced["coords"] == "pixel" and forced["thinking"] == "on"
    assert vision.request_body([], forced)["enable_thinking"] is True
    assert vision.TOOL["description"].count("general knowledge") == 1


def test_ask_vision_converts_marked_locations_to_original_pixels():
    vision = _load("vision_mcp")
    screen = {"size": (1920, 1080), "shown": (1920, 1080)}
    big = {"size": (4096, 4096), "shown": (2048, 2048)}
    text, n = vision.convert_locations("Yes at <pt>500,500</pt>; bar <box>0,0,1000,100</box>.", [screen], "norm1000")
    assert n == 2 and text == "Yes at (960, 540); bar (0, 0)-(1919, 108)."
    text, n = vision.convert_locations("A <pt i=2>100, 200</pt> B <pt i=1>10,20</pt>", [screen, big], "pixel")
    assert n == 2 and text == "A (200, 400) in image 2 B (10, 20) in image 1"
    text, n = vision.convert_locations("bad <pt i=3>1,2</pt> and <pt>1,2,3</pt>", [screen], "norm1000")
    assert n == 0 and "<pt i=3>" in text
    variants = "<pt x=250,y=500</pt> <pt>(250, 500)</pt> <pt x=\"250\" y=\"500\"></pt>"
    text, n = vision.convert_locations(variants, [screen], "norm1000")
    assert n == 3 and text == "(480, 540) (480, 540) (480, 540)"
    assert vision.to_source_px(5000, 0, big, "pixel") == 4095 and vision.to_source_px(-3, 1, big, "pixel") == 0


def test_ask_vision_v2_answer_is_converted_and_logged(tmp_path, monkeypatch, capsys):
    vision, calls = _vision(tmp_path, monkeypatch)
    monkeypatch.setenv("RV_VISION_MODEL", "qwen-vl-test")
    monkeypatch.setattr(vision, "call_model", lambda m, p=None: {
        "answer": "The OK button is at <pt>500,250</pt>.", "usage": {}, "finish_reason": "stop", "reasoning_chars": 0})
    image = _png(tmp_path / "ui.png", size=(1000, 800))
    result = _call(vision, capsys, {"question": "Where is OK?", "images": [str(image)]})
    text = result["content"][0]["text"]
    assert text.endswith("The OK button is at (500, 200).") and "pixel coordinates of the original images" in text
    entry = json.loads((tmp_path / "vision_calls.jsonl").read_text().splitlines()[-1])
    assert entry["converted"] == 1 and entry["raw_answer"].endswith("<pt>500,250</pt>.")
    assert entry["prompt_version"] == "v2" and entry["profile"]["coords"] == "norm1000"


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def advance(self, seconds):
        self.now += seconds

    def wait(self, seconds):
        self.waits.append(seconds)
        self.advance(seconds)
        return False


def _offline_vision(tmp_path, monkeypatch):
    vision = _load("vision_mcp")
    monkeypatch.setenv("RV_VISION_MODEL", "kimi-vision")
    monkeypatch.setenv("RV_LLM_BASE_URL", "https://offline.invalid/v1")
    monkeypatch.setenv("RV_LLM_API_KEY", "offline-test-only")
    monkeypatch.delenv("RV_VISION_BASE_URL", raising=False)
    monkeypatch.delenv("RV_VISION_API_KEY", raising=False)
    monkeypatch.setenv("OSWORLD_DSH_ATTEMPT", str(tmp_path))
    clock = _Clock()
    monkeypatch.setattr(vision, "RETRY_WAITS", ())  # unexpected fake failures must never cause real backoff
    monkeypatch.setattr(vision.time, "monotonic", lambda: clock.now)
    src = _png(tmp_path / "source.png", size=(400, 300), color=(0, 200, 0))
    return vision, clock, {"question": "What is visible?", "images": [str(src)]}


def _chunk(answer="green", tokens=10):
    return {"choices": [{"delta": {"content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tokens, "prompt_tokens_details": {"cached_tokens": 2}}}


def _fake_http(vision, monkeypatch, scripts):
    """Exercise real request serialization/SSE parsing; scripts contain chunks, callbacks or errors."""
    requests, responses = [], []
    scripts = iter(scripts)

    class Response:
        def __init__(self, script):
            self.script, self.closed, self.shutdowns = script, False, []
            self.fp = types.SimpleNamespace(raw=types.SimpleNamespace(_sock=self))

        def shutdown(self, how):
            self.shutdowns.append(how)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def __iter__(self):
            for item in self.script:
                if callable(item):
                    item()
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield ("data: " + json.dumps(item) + "\n").encode()
            yield b"data: [DONE]\n"

    def urlopen(request, timeout, operation):
        requests.append((json.loads(request.data), timeout, request))
        response = Response(next(scripts))
        responses.append(response)
        return response

    monkeypatch.setattr(vision, "open_request", urlopen)
    return requests, responses


def _vision_log(tmp_path):
    return [json.loads(line) for line in (tmp_path / "vision_calls.jsonl").read_text().splitlines()]


def test_vision_request_uses_the_policy_route_or_a_separate_endpoint(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    requests, _ = _fake_http(vision, monkeypatch, [[_chunk()], [_chunk()]])
    assert vision.run_tool(arguments).endswith("green")
    body, timeout, request = requests[0]
    assert request.full_url == "https://offline.invalid/v1/chat/completions" and timeout == 600
    assert request.get_header("Authorization") == "Bearer offline-test-only"
    assert body["model"] == "kimi-vision" and "tools" not in body
    monkeypatch.setenv("RV_VISION_BASE_URL", "https://vision.invalid/v1/")
    monkeypatch.setenv("RV_VISION_API_KEY", "vision-key")
    vision.run_tool(arguments)
    request = requests[1][2]
    assert request.full_url == "https://vision.invalid/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer vision-key"


@pytest.mark.parametrize("succeeds", [False, True])
def test_vision_preserves_partial_stream_usage_on_retry(tmp_path, monkeypatch, succeeds):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    monkeypatch.setattr(vision, "RETRY_WAITS", (5,))
    operation = vision.Operation()
    monkeypatch.setattr(operation.cancelled, "wait", clock.wait)
    requests, _ = _fake_http(vision, monkeypatch, [
        [_chunk(tokens=7), OSError("lost stream")],
        [_chunk(tokens=11)] if succeeds else [_chunk("", tokens=11)],
    ])
    if succeeds:
        vision.run_tool(arguments, operation)
    else:
        with pytest.raises(vision.ToolError, match="unavailable"):
            vision.run_tool(arguments, operation)
    entry, = _vision_log(tmp_path)
    assert len(requests) == 2 and clock.waits == [5]
    assert entry["usage"] == {"answerer_calls": 2, "prompt_tokens": 18}
    assert entry["requests"][0]["usage"]["prompt_tokens_details"] == {"cached_tokens": 2}
    assert entry["requests"][0]["status"] == "error"
    assert entry["status"] == ("ok" if succeeds else "error")


def test_vision_retry_budget_includes_waits_and_remaining_timeout(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    operation = vision.Operation(timeout=12)
    monkeypatch.setattr(vision, "RETRY_WAITS", (5, 20))
    monkeypatch.setattr(operation.cancelled, "wait", clock.wait)
    requests, _ = _fake_http(vision, monkeypatch, [
        [lambda: clock.advance(4), OSError("try again")],
        [lambda: clock.advance(3), _chunk(tokens=3)],
    ])
    with vision.operation_scope(operation), pytest.raises(vision.ToolError, match="deadline"):
        vision.ask(vision.build_messages("q", []), timeout=12)
    assert [timeout for _, timeout, _ in requests] == [12, 3]
    assert clock.waits == [5] and clock.now == 12
    assert operation.accounting()["usage"] == {"answerer_calls": 2, "prompt_tokens": 3}
    # A retry whose backoff cannot fit must not sleep or launch another request.
    clock.now = 0
    operation = vision.Operation(timeout=4)
    monkeypatch.setattr(operation.cancelled, "wait", clock.wait)
    requests, _ = _fake_http(vision, monkeypatch, [[OSError("down")]])
    with vision.operation_scope(operation), pytest.raises(vision.ToolError, match="unavailable"):
        vision.ask([], timeout=4)
    assert len(requests) == 1 and clock.now == 0


def test_vision_mcp_cancel_interrupts_stream_and_preserves_usage(tmp_path, monkeypatch, capsys):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    cancel = lambda: vision.handle({"method": "notifications/cancelled", "params": {"requestId": 7}})
    requests, responses = _fake_http(vision, monkeypatch, [
        [_chunk("", tokens=9), cancel, _chunk("late", tokens=12)]
    ])
    vision.handle({"id": 7, "method": "tools/call", "params": {"name": "ask_vision", "arguments": arguments}})
    entry, = _vision_log(tmp_path)
    assert capsys.readouterr().out == "" and not vision._active
    assert len(requests) == 1 and len(responses[0].shutdowns) == 1 and responses[0].closed
    assert entry["status"] == "cancelled" and entry["usage"] == {"answerer_calls": 1, "prompt_tokens": 12}
    assert entry["requests"][0]["status"] == "cancelled"


def test_vision_cancellation_during_backoff_stops_retries(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    operation = vision.Operation()
    monkeypatch.setattr(vision, "RETRY_WAITS", (5,))
    monkeypatch.setattr(operation.cancelled, "wait", lambda seconds: operation.cancel() or True)
    requests, _ = _fake_http(vision, monkeypatch, [[_chunk(tokens=2), OSError("lost")]])
    with pytest.raises(vision.Cancelled):
        vision.run_tool(arguments, operation)
    entry, = _vision_log(tmp_path)
    assert len(requests) == 1 and entry["status"] == "cancelled" and entry["usage"]["prompt_tokens"] == 2


def test_vision_cancellation_after_answer_is_not_answered(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    operation = vision.Operation()
    real_convert = vision.convert_locations

    def cancel_after_answer(*args):
        operation.cancel()
        return real_convert(*args)

    monkeypatch.setattr(vision, "convert_locations", cancel_after_answer)
    _fake_http(vision, monkeypatch, [[_chunk()]])
    with pytest.raises(vision.Cancelled):
        vision.run_tool(arguments, operation)
    assert _vision_log(tmp_path)[0]["status"] == "cancelled" and not operation.completed


def test_vision_main_registers_before_dispatch_for_immediate_cancel(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    workers = []

    class DeferredThread:
        def __init__(self, target, args, daemon):
            workers.append(lambda: target(*args))

        def start(self):
            pass

    monkeypatch.setattr(vision.threading, "Thread", DeferredThread)
    call = {"id": "queued", "method": "tools/call", "params": {"name": "ask_vision", "arguments": arguments}}
    cancel = {"method": "notifications/cancelled", "params": {"requestId": "queued"}}
    monkeypatch.setattr(vision.sys, "stdin", io.StringIO(json.dumps(call) + "\n" + json.dumps(cancel) + "\n"))
    requests, _ = _fake_http(vision, monkeypatch, [])
    vision.main()
    assert vision._active["queued"].cancelled.is_set()
    workers[0]()
    assert not vision._active and not requests
    assert _vision_log(tmp_path)[0]["status"] == "cancelled"


def test_vision_stream_deadline_watchdog_interrupts_socket(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    timers = []

    class Timer:
        def __init__(self, seconds, callback):
            self.seconds, self.callback, self.cancelled = seconds, callback, False
            timers.append(self)

        def start(self):
            pass

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(vision.threading, "Timer", Timer)

    def expire():
        clock.advance(1200)
        timers[0].callback()

    requests, responses = _fake_http(vision, monkeypatch, [[_chunk(tokens=6), expire]])
    with pytest.raises(vision.ToolError, match="deadline"):
        vision.run_tool(arguments)
    assert len(requests) == 1 and responses[0].shutdowns == [socket.SHUT_RDWR]
    assert timers[0].seconds == 1200 and timers[0].cancelled and responses[0].closed
    assert _vision_log(tmp_path)[0]["usage"]["prompt_tokens"] == 6


def test_vision_direct_ask_also_enforces_payload_cap(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    image = vision.load_source(arguments["images"][0])
    monkeypatch.setattr(vision, "MAX_CALL_BYTES", len(image["data"]) - 1)
    requests, _ = _fake_http(vision, monkeypatch, [])
    operation = vision.Operation()
    with vision.operation_scope(operation), pytest.raises(vision.ToolError, match="MiB per call"):
        vision.ask(vision.build_messages("q", [image]))
    assert not requests and operation.accounting()["usage"]["answerer_calls"] == 0
    assert operation.rejected_payload_bytes == len(image["data"])


def test_vision_deadline_includes_source_preparation(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    real_load = vision.load_source

    def slow_source(source):
        clock.advance(1200)
        return real_load(source)

    monkeypatch.setattr(vision, "load_source", slow_source)
    requests, _ = _fake_http(vision, monkeypatch, [])
    with pytest.raises(vision.ToolError, match="deadline"):
        vision.run_tool(arguments)
    assert not requests and _vision_log(tmp_path)[0]["usage"]["answerer_calls"] == 0


def test_vision_cancellation_targets_only_its_request(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    first, second = vision.Operation(), vision.Operation()
    vision._active.update({1: first, 2: second})
    vision.handle({"method": "notifications/cancelled", "params": {"requestId": "unknown"}})
    assert not first.cancelled.is_set() and not second.cancelled.is_set()
    vision.handle({"method": "notifications/cancelled", "params": {"requestId": 1}})
    assert first.cancelled.is_set() and not second.cancelled.is_set()
    _fake_http(vision, monkeypatch, [[_chunk()]])
    assert vision.run_tool(arguments, second).endswith("green") and second.completed
    vision.handle({"method": "notifications/cancelled", "params": {"requestId": 2}})
    assert not second.cancelled.is_set()  # cancellation after the completion boundary is ignored
    vision._active.clear()


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("phase", ["connect", "headers"])
def test_vision_interrupts_connection_before_response_headers(tmp_path, monkeypatch, stop, scheme, phase):
    """Exercise the actual urllib handlers with a fake connection awaiting headers."""
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    monkeypatch.setenv("RV_LLM_BASE_URL", scheme + "://offline.invalid/v1")
    monkeypatch.setattr(vision.urllib.request, "getproxies", lambda: {})
    operation = vision.Operation()
    connections = []

    def stop_now():
        if stop == "cancel":
            operation.cancel()
        else:
            clock.advance(1200)
            operation.interrupt()  # watchdog while awaiting connection or headers

    class Connection(vision.http.client.HTTPConnection):
        def __init__(self, host, timeout, **kwargs):
            self.sock = None
            self.posted = False
            self.interrupted = vision.threading.Event()
            self.closed = False
            self.timeout = timeout
            connections.append(self)

        def set_debuglevel(self, level):
            pass

        def connect(self):
            if phase == "connect":
                stop_now()
            self.sock = self

        def request(self, method, url, body, headers, **kwargs):
            assert method == "POST" and operation.connection is self
            self.connect()
            self.posted = True

        def shutdown(self, how):
            assert how == socket.SHUT_RDWR
            self.interrupted.set()

        def getresponse(self):
            assert operation.response is None  # still awaiting headers
            stop_now()
            assert self.interrupted.is_set()
            raise OSError("interrupted header read")

        def close(self):
            self.closed = True

    monkeypatch.setattr(vision.http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(vision.http.client, "HTTPSConnection", Connection)
    with pytest.raises(vision.ToolError, match="cancelled" if stop == "cancel" else "deadline"):
        vision.run_tool(arguments, operation)
    assert len(connections) == 1 and connections[0].closed and connections[0].timeout == 600
    assert connections[0].posted is (phase == "headers")
    entry, = _vision_log(tmp_path)
    assert entry["status"] == ("cancelled" if stop == "cancel" else "error")
    assert entry["requests"][0]["status"] == entry["status"] and entry["requests"][0]["usage"] is None


def test_vision_cancelled_socket_error_has_cancelled_request_status(tmp_path, monkeypatch):
    vision, clock, arguments = _offline_vision(tmp_path, monkeypatch)
    operation = vision.Operation()
    requests, responses = _fake_http(vision, monkeypatch, [
        [_chunk(tokens=8), operation.cancel, OSError("socket shutdown")]
    ])
    with pytest.raises(vision.Cancelled):
        vision.run_tool(arguments, operation)
    entry, = _vision_log(tmp_path)
    assert len(requests) == 1 and responses[0].shutdowns and responses[0].closed
    assert entry["status"] == entry["requests"][0]["status"] == "cancelled"
    assert entry["usage"]["prompt_tokens"] == 8


# ------------------------------------------------------------------ website DNS


dns = _load("website_dns")


def _query(name, typ=1):
    return b"\x12\x34" + struct.pack("!HHHHH", 0x0100, 1, 0, 0, 0) + b"".join(
        bytes([len(s)]) + s.encode() for s in name.split(".")) + b"\0" + struct.pack("!HH", typ, 1)


def test_website_dns_fixed_sites_and_only_gitlab_wildcard_loopback():
    config = {"hostnames": ["mailhub.osworld.test"], "gitlab_domain": "gitlab.osworld.test", "forwarders": []}
    for name in ["mailhub.osworld.test", "gitlab.osworld.test", "user-a.gitlab.osworld.test",
                 "pages-a.gitlab.osworld.test"]:
        reply = dns.response(_query(name), config)
        assert struct.unpack("!H", reply[6:8])[0] == 1 and reply[-4:] == b"\x7f\0\0\1"
    for name in ["secret.osworld.test", "evilgitlab.osworld.test", "gitlab.osworld.test.evil"]:
        reply = dns.response(_query(name), config)
        assert struct.unpack("!H", reply[6:8])[0] == 0
    assert struct.unpack("!H", dns.response(_query("mailhub.osworld.test", 28), config)[6:8])[0] == 0
