"""Offline tests of the benchmark runners: no VM, container, network or model call.

Arguments and task selection, protocol defaults and the user-supplied model errors, run settings,
the WeaveBench judge (profile, retry proxy, per-judge scores, trajectory conversion), ALE budget
and reference handling, the evaluator token floor and the OSWorld 2.0 launcher.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from mmcoding import harnesses, paths

if not (paths.OSWORLD / "hybrid_agents").is_dir():
    # Tests run inside the OSWorld-V2 venv; locate that checkout when $RV_THIRD_PARTY is not set.
    import hybrid_agents

    paths.OSWORLD = paths.UPSTREAM["osworld"] = Path(next(iter(hybrid_agents.__path__))).resolve().parent
paths.use_upstream("osworld", "weavebench")
from mmcoding.host import deploy  # noqa: E402  (before the runners: they set the guest password at import)
from mmcoding.benchmarks import common  # noqa: E402
from mmcoding.benchmarks import evaluation_transport as transport  # noqa: E402
from mmcoding.benchmarks import weavebench_judge as wj  # noqa: E402
from mmcoding.benchmarks import weavebench_judge_proxy as proxy  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BENCHES = ("osworld2", "weavebench", "ale")
TASK_COUNTS = {"osworld2": 108, "weavebench": 114, "ale": 85}


def runner(name: str):
    """Import a runner without leaking its guest password into the environment of other tests."""
    saved = os.environ.get("OSWORLD_CLIENT_PASSWORD")
    try:
        return importlib.import_module(f"mmcoding.benchmarks.{name}")
    finally:
        if saved is None:
            os.environ.pop("OSWORLD_CLIENT_PASSWORD", None)
        else:
            os.environ["OSWORLD_CLIENT_PASSWORD"] = saved


def agent_args(bench, *extra, harness="mm_coding"):
    parser = argparse.ArgumentParser()
    common.add_agent_args(parser, bench)
    return parser.parse_args(["--harness", harness, "--model", "policy-model", *extra])


# ------------------------------------------------------------------ protocol


def test_protocol_has_the_three_benchmarks_and_the_paper_defaults():
    data = json.loads((paths.CONFIGS / "protocol.json").read_text())
    assert set(data) - {"_comment"} == set(BENCHES)
    for bench in BENCHES:
        proto = common.protocol(bench)
        assert (proto["call_budget"], proto["reasoning_effort"], proto["max_tokens"]) == (500, "max", 65536)
        assert (proto["vision_model"], proto["vision_reasoning_effort"]) == ("qwen3.8-27b", "off")
        assert (proto["vision_thinking"], proto["vision_coords"]) == ("off", "norm1000")
        assert proto["timeout_s"] == 57600
    for bench in ("osworld2", "weavebench"):
        assert (common.protocol(bench)["vm_ram"], common.protocol(bench)["vm_cores"]) == ("8G", "4")
    assert common.protocol("osworld2")["client_password"] == "osworld-public-evaluation"
    assert common.protocol("weavebench")["client_password"] == common.protocol("ale")["client_password"] == "password"
    assert common.protocol("osworld2")["benchmark_release"] == "osworld-v2.1"
    run_keys = {"call_budget", "timeout_s", "reasoning_effort", "max_tokens", "vision_model", "vision_reasoning_effort",
                "vision_thinking", "vision_coords", "client_password"}
    assert set(data["osworld2"]) == run_keys | {"vm_ram", "vm_cores", "benchmark_release", "evaluator_provider",
                                                "evaluator_model", "evaluator_api_key_env", "user_simulator_model"}
    assert set(data["weavebench"]) == run_keys | {"vm_ram", "vm_cores", "judge_model", "judge_thinking",
                                                  "judge_timeout_s", "judge"}
    assert set(data["ale"]) == run_keys | {"eval_timeout_s", "judge_timeout_s", "pass_threshold"}


def test_judge_and_evaluator_models_are_left_to_the_user():
    assert common.protocol("osworld2")["evaluator_model"] is None
    assert common.protocol("osworld2")["user_simulator_model"] is None
    proto = common.protocol("weavebench")
    assert proto["judge_model"] is None
    assert proto["judge"]["base_url_env"] == "WEAVEBENCH_JUDGE_BASE_URL"
    assert proto["judge"]["api_key_env"] == "WEAVEBENCH_JUDGE_API_KEY"


def test_user_setting_errors_when_unset_and_prefers_the_environment(monkeypatch):
    monkeypatch.delenv("OSWORLD_EVAL_MODEL_NAME", raising=False)
    with pytest.raises(SystemExit, match="evaluator_model is not set; export OSWORLD_EVAL_MODEL_NAME"):
        common.user_setting("osworld2", "evaluator_model", "OSWORLD_EVAL_MODEL_NAME")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_NAME", "eval-model")
    assert common.user_setting("osworld2", "evaluator_model", "OSWORLD_EVAL_MODEL_NAME") == "eval-model"


# ------------------------------------------------------------------ arguments


@pytest.mark.parametrize("bench", BENCHES)
def test_agent_args_take_protocol_defaults(bench):
    args = agent_args(bench)
    assert (args.harness, args.model, args.provider) == ("mm_coding", "policy-model", "openai")
    assert (args.call_budget, args.reasoning_effort, args.max_tokens) == (500, "max", 65536)
    assert (args.vision_model, args.vision_reasoning_effort) == ("qwen3.8-27b", "off")
    assert (args.vision_thinking, args.vision_coords) == ("off", "norm1000")
    assert (args.api_key_env, args.tasks, args.result_dir, args.network_guard) == ("RV_LLM_API_KEY", "all", None, False)


@pytest.mark.parametrize("value,expected", [
    ("gui", "gui"), ("GUI", "gui"), ("OS-Helix", "os_helix"), ("os_helix", "os_helix"),
    ("MM-Coding", "mm_coding"), ("Vision-Tool", "vision_tool"), ("cli", "cli"), ("CLI", "cli"),
])
def test_harness_accepts_ids_and_paper_names(value, expected):
    assert agent_args("osworld2", harness=value).harness == expected


@pytest.mark.parametrize("value", ["mmcoding", "vision", "MM Coding", ""])
def test_unknown_harness_is_rejected(value, capsys):
    with pytest.raises(SystemExit):
        agent_args("osworld2", harness=value)
    assert "unknown harness" in capsys.readouterr().err


def test_every_harness_parses():
    assert {agent_args("ale", harness=h).harness for h in harnesses.HARNESSES} == set(harnesses.HARNESSES)


def test_model_is_required(capsys):
    parser = argparse.ArgumentParser()
    common.add_agent_args(parser, "osworld2")
    with pytest.raises(SystemExit):
        parser.parse_args(["--harness", "gui"])
    assert "--model" in capsys.readouterr().err


def test_explicit_values_override_the_protocol_and_network_guard_flag():
    args = agent_args("osworld2", "--reasoning_effort", "low", "--max_tokens", "1024", "--call_budget", "12",
                      "--vision_model", "other-vision", "--vision_reasoning_effort", "high",
                      "--vision_thinking", "on", "--vision_coords", "pixel", "--network-guard")
    assert (args.reasoning_effort, args.max_tokens, args.call_budget) == ("low", 1024, 12)
    assert (args.vision_model, args.vision_reasoning_effort, args.vision_thinking, args.vision_coords) == (
        "other-vision", "high", "on", "pixel")
    assert args.network_guard is True


def test_finish_agent_args_resolves_tasks_credentials_and_result_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RESULTS", tmp_path)
    parser = argparse.ArgumentParser()
    common.add_agent_args(parser, "weavebench")
    args = parser.parse_args(["--harness", "Vision-Tool", "--model", "m", "--base_url", "https://x.invalid/v1"])
    monkeypatch.delenv("RV_LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        common.finish_agent_args(parser, args, "weavebench")
    monkeypatch.setenv("RV_LLM_API_KEY", "synthetic")
    args = parser.parse_args(["--harness", "Vision-Tool", "--model", "m", "--base_url", "https://x.invalid/v1"])
    common.finish_agent_args(parser, args, "weavebench")
    assert len(args.tasks) == 114 and args.api_key == "synthetic"
    assert args.result_dir.parent.parent == tmp_path / "weavebench" / "m" and args.result_dir.name == "vision_tool"


# ------------------------------------------------------------------ tasks


@pytest.mark.parametrize("bench", BENCHES)
def test_all_is_the_complete_task_list(bench):
    tasks = common.load_tasks("all", bench)
    assert len(tasks) == len(set(tasks)) == TASK_COUNTS[bench]
    data = json.loads(paths.task_file(bench).read_text())
    assert set(data) == {"note", "tasks"}


def test_task_selection_by_ids_and_file(tmp_path):
    assert common.load_tasks(" 001, 002 ,", "osworld2") == ["001", "002"]
    (tmp_path / "list.json").write_text(json.dumps(["a", "b"]))
    (tmp_path / "obj.json").write_text(json.dumps({"tasks": ["c"]}))
    assert common.load_tasks(f"@{tmp_path / 'list.json'}", "weavebench") == ["a", "b"]
    assert common.load_tasks(f"@{tmp_path / 'obj.json'}", "weavebench") == ["c"]
    with pytest.raises(ValueError, match="duplicate"):
        common.load_tasks("a,b,a", "ale")
    with pytest.raises(ValueError, match="no tasks"):
        common.load_tasks(",", "ale")


def test_osworld_task_ids_are_padded_and_checked():
    ow = runner("osworld2")
    assert ow.parse_tasks(["1", "059", "108"]) == ["001", "059", "108"]
    for bad in (["109"], ["x"], ["1", "001"]):
        with pytest.raises(SystemExit):
            ow.parse_tasks(bad)
    assert ow.parse_tasks(common.load_tasks("all", "osworld2")) == [f"{i:03d}" for i in range(1, 109)]


def test_ale_list_excludes_the_in_guest_docker_task():
    tasks = common.load_tasks("all", "ale")
    assert "computing_math/k8s_migration_1" not in tasks and all(t.count("/") == 1 for t in tasks)


# ------------------------------------------------------------------ settings


def test_settings_carry_harness_route_budget_and_vision():
    args = agent_args("osworld2", harness="Vision-Tool")
    args.base_url = "https://policy.invalid/v1"
    settings = common.settings(args, 600)
    assert isinstance(settings, deploy.RvSettings)
    assert (settings.harness, settings.provider, settings.call_budget) == ("vision_tool", "openai", 500)
    assert (settings.model, settings.max_tokens, settings.reasoning_effort, settings.timeout) == (
        "policy-model", 65536, "max", 600)
    assert (settings.vision_model, settings.vision_reasoning_effort) == ("qwen3.8-27b", "off")
    assert (settings.vision_coords, settings.vision_thinking) == ("norm1000", "off")


@pytest.fixture
def osworld_env(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("private")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA test")
    image = tmp_path / "vm.qcow2"
    image.touch()
    monkeypatch.setenv("RV_LLM_API_KEY", "synthetic")
    for name in ("OSWORLD_EVAL_MODEL_NAME", "OSWORLD_USER_SIM_MODEL", "OSWORLD_EVAL_MODEL_PROVIDER",
                 "OSWORLD_USER_SIM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    return ["--harness", "MM-Coding", "--model", "policy-model", "--base_url", "https://policy.invalid/v1",
            "--ssh_private_key", str(key), "--path_to_vm", str(image), "--tasks", "1,2",
            "--result_dir", str(tmp_path / "run")]


def test_osworld_config_requires_evaluator_and_user_simulator(osworld_env, monkeypatch, capsys):
    ow = runner("osworld2")
    with pytest.raises(SystemExit):
        ow.config(osworld_env)
    assert "evaluator_model is not set" in capsys.readouterr().err
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_NAME", "eval-model")
    with pytest.raises(SystemExit):
        ow.config(osworld_env)
    assert "user_simulator_model is not set" in capsys.readouterr().err
    monkeypatch.setenv("OSWORLD_USER_SIM_MODEL", "sim-model")
    args = ow.config([*osworld_env, "--network-guard"])
    assert args.tasks == ["001", "002"] and args.harness == "mm_coding" and args.network_guard
    assert args.settings.harness == "mm_coding" and args.settings.call_budget == 500
    assert args.evaluator["OSWORLD_EVAL_MODEL_PROVIDER"] == "openai_compatible"
    assert args.evaluator["OSWORLD_USER_SIM_PROVIDER"] == "openai_compatible"
    assert (args.region, args.recovery_mode) == ("docker", True)
    assert len(args.snapshot_name.removeprefix("sha256:")) == 64


def test_osworld_resume_runs_only_tasks_without_a_directory(tmp_path):
    ow = runner("osworld2")
    args = SimpleNamespace(result_dir=tmp_path / "run", resume=False, tasks=["001", "002"])
    assert ow.prepare_result_dir(args) is False and args.tasks == ["001", "002"]
    (args.result_dir / "001").mkdir()
    with pytest.raises(SystemExit, match="--resume"):
        ow.prepare_result_dir(SimpleNamespace(result_dir=tmp_path / "run", resume=False, tasks=["001"]))
    again = SimpleNamespace(result_dir=tmp_path / "run", resume=True, tasks=["001", "002"])
    assert ow.prepare_result_dir(again) is True and again.tasks == ["002"]


def test_osworld_reaps_retained_vm_without_pausing(tmp_path, monkeypatch):
    ow = runner("osworld2")
    calls = []
    monkeypatch.setattr(ow.subprocess, "run",
                        lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    args = SimpleNamespace(result_dir=tmp_path)
    (tmp_path / "059").mkdir()
    (tmp_path / "059" / "container.json").write_text(json.dumps({"container": "vm_059"}))
    assert ow.reap_retained(args, "059", {"cleanup_error": "VM retained for evidence recovery"})["removed"]
    assert calls == [["docker", "rm", "-f", "vm_059"]]
    assert ow.reap_retained(args, "060", {"status": "completed"}) is None
    assert json.loads((tmp_path / "retained_vms.json").read_text())[0]["task"] == "059"
    assert "configuration_drift" in ow.ACCOUNT_FAILURES and "agent_runtime" not in ow.ACCOUNT_FAILURES


# ------------------------------------------------------------------ evaluation transport


def test_evaluator_floor_applies_only_to_listed_models(monkeypatch):
    from desktop_env.evaluators import model_client

    calls = []

    def original(model, max_tokens):
        calls.append((model, max_tokens))
        return SimpleNamespace(model=model, max_tokens=max_tokens, temperature=0, provider="test")

    monkeypatch.setattr(model_client, "_build_config", original)
    monkeypatch.delenv("RV_EVAL_MIN_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("RV_EVAL_MIN_OUTPUT_MODELS", raising=False)
    assert transport.install() == {"models": [], "min_output_tokens": 16384}
    assert model_client._build_config("reasoning-eval", 1500).max_tokens == 1500  # off by default
    monkeypatch.setenv("RV_EVAL_MIN_OUTPUT_MODELS", "reasoning-eval, other")
    assert transport.install()["models"] == ["other", "reasoning-eval"]
    first = model_client._build_config("reasoning-eval", 1500)
    assert first.max_tokens == 16384 and first.temperature == 0 and first.provider == "test"
    assert model_client._build_config("reasoning-eval", 32768).max_tokens == 32768
    assert model_client._build_config("unlisted", 1500).max_tokens == 1500
    assert len(calls) == 4  # repeated install does not stack wrappers


def test_evaluator_floor_validated_and_configurable(monkeypatch):
    monkeypatch.setenv("RV_EVAL_MIN_OUTPUT_TOKENS", "0")
    with pytest.raises(ValueError):
        transport.policy()
    monkeypatch.setenv("RV_EVAL_MIN_OUTPUT_TOKENS", "20000")
    assert transport.policy()["min_output_tokens"] == 20000


def test_evaluator_floor_applies_after_the_task_option(monkeypatch):
    from desktop_env.evaluators import model_client

    monkeypatch.setenv("OSWORLD_EVAL_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_NAME", "reasoning-eval")
    monkeypatch.setenv("OSWORLD_EVAL_MODEL_API_KEY", "unit-test-not-real")
    monkeypatch.setenv("RV_EVAL_MIN_OUTPUT_MODELS", "reasoning-eval")
    original = getattr(model_client._build_config, "_rv_eval_original", model_client._build_config)
    kwargs = dict(default_provider="openai", default_model="m", default_api_key_env="OPENAI_API_KEY",
                  default_max_tokens=512)
    assert original({"max_tokens": 1500}, **kwargs).max_tokens == 1500
    monkeypatch.setattr(model_client, "_build_config", original)
    transport.install()
    config = model_client._build_config({"max_tokens": 1500}, **kwargs)
    assert config.max_tokens == 16384 and config.model == "reasoning-eval"


# ------------------------------------------------------------------ trajectory


def test_dsh_trajectory_becomes_openclaw_chat(tmp_path):
    events = [
        {"type": "user/message", "data": {"content": [{"type": "text", "text": "do it"}]}},
        {"type": "assistant/message", "data": {"message": {"content": [
            {"type": "reasoning", "text": "think"}, {"type": "text", "text": "ok"},
            {"type": "tool-call", "id": "c1", "name": "bash", "arguments": "{\"command\": \"ls\"}"},
            {"type": "tool-call", "id": "c2", "name": "bash", "arguments": "{broken"}]}}},
        {"type": "tool/result", "data": {"message": {"content": [{"type": "tool-result", "toolCallId": "c1",
            "content": [{"type": "text", "text": "a.txt"},
                        {"type": "image", "attachment": {"width": 10, "height": 20}}], "isError": False}]}}},
    ]
    dsh = tmp_path / "task" / "dsh"
    dsh.mkdir(parents=True)
    (dsh / "notifications.jsonl.gz").write_bytes(gzip.compress(("".join(
        json.dumps({"payload": {"event": e}}) + "\n" for e in events) + "not json\n").encode()))
    n = common.dsh_to_openclaw_chat(dsh, tmp_path / "chat.jsonl", "policy-model")
    lines = [json.loads(x) for x in (tmp_path / "chat.jsonl").read_text().splitlines()]
    assert n == 5 and lines[0]["type"] == "session" and lines[0]["id"] == "task"
    assert (lines[1]["modelId"], lines[1]["provider"]) == ("policy-model", "openai")
    assert lines[2]["message"] == {"role": "user", "content": [{"type": "text", "text": "do it"}]}
    assistant = lines[3]["message"]["content"]
    assert [b["type"] for b in assistant] == ["thinking", "text", "tool_use", "tool_use"]
    assert assistant[2]["input"] == {"command": "ls"} and assistant[3]["input"] == {"raw": "{broken"}
    result = lines[4]["message"]["content"][0]
    assert result["tool_use_id"] == "c1" and result["content"] == "a.txt\n[image 10x20]" and not result["is_error"]


def test_missing_trajectory_gives_header_only(tmp_path):
    (tmp_path / "dsh").mkdir()
    assert common.dsh_to_openclaw_chat(tmp_path / "dsh", tmp_path / "chat.jsonl", "m", "p") == 2


# ------------------------------------------------------------------ WeaveBench judge profile


@pytest.fixture
def judge_env(monkeypatch):
    for name in ("WEAVEBENCH_JUDGE_MODEL", "WEAVEBENCH_JUDGE_API"):
        monkeypatch.delenv(name, raising=False)


def test_unset_judge_model_is_a_clear_error(judge_env):
    with pytest.raises(SystemExit, match="WeaveBench judge model is not set.*WEAVEBENCH_JUDGE_MODEL"):
        wj.judge()


def test_judge_model_from_argument_or_environment(judge_env, monkeypatch):
    assert wj.judge("judge-a")[0] == "judge-a"
    monkeypatch.setenv("WEAVEBENCH_JUDGE_MODEL", "judge-b")
    monkeypatch.setenv("WEAVEBENCH_JUDGE_API", "openai-completions")
    name, cfg = wj.judge()
    assert name == "judge-b" and cfg["api"] == "openai-completions"
    assert wj.judge("judge-a")[0] == "judge-a"


def test_template_is_placeholders_only():
    text = wj.TEMPLATE.read_text()
    template = json.loads(text)
    provider = template["models"]["providers"]["PLACEHOLDER_PROVIDER"]
    assert provider["apiKey"] == "PLACEHOLDER_API_KEY" and provider["baseUrl"] == "PLACEHOLDER_PROXY_URL"
    assert provider["models"][0]["id"] == "PLACEHOLDER_JUDGE_MODEL"
    assert template["agents"]["defaults"]["model"]["primary"] == "PLACEHOLDER_PROVIDER/PLACEHOLDER_JUDGE_MODEL"
    assert "http" not in text and "sk-" not in text


def test_render_substitutes_every_placeholder(judge_env, tmp_path):
    name, cfg = wj.judge("vendor/judge-model")
    profile = wj.render(name, cfg, key="sk-test-0123456789", token="tok", workspace=tmp_path / "ws")
    assert "PLACEHOLDER" not in json.dumps(profile)
    provider = profile["models"]["providers"][cfg["provider"]]
    assert provider["apiKey"] == "sk-test-0123456789"  # not clobbered by the shorter PLACEHOLDER_API
    assert provider["api"] == cfg["api"] and provider["baseUrl"] == f"http://127.0.0.1:{cfg['proxy_port']}/v1"
    model = provider["models"][0]
    assert model["id"] == name and (model["maxTokens"], model["contextWindow"]) == (16384, 262144)
    assert profile["agents"]["defaults"]["model"]["primary"] == f"{cfg['provider']}/{name}"
    assert wj.profile_dir(name).name == "vendor%2Fjudge-model"


def test_render_keeps_quotes_backslashes_and_placeholder_literals(judge_env, tmp_path):
    name, cfg = wj.judge("judge-a")
    key = 'secret"\\\nPLACEHOLDER_PROVIDER/PLACEHOLDER_API'
    token = 'token"\\\nPLACEHOLDER_JUDGE_MODEL'
    workspace = tmp_path / 'workspace"\\\nPLACEHOLDER_API_KEY'
    profile = wj.render(name, cfg, key, token, workspace)
    assert profile["models"]["providers"][cfg["provider"]]["apiKey"] == key
    assert profile["gateway"]["auth"]["token"] == token
    assert profile["agents"]["defaults"]["workspace"] == str(workspace)


def test_problem_reports_missing_and_mismatched_profiles(judge_env, tmp_path):
    name, cfg = wj.judge("judge-a")
    assert "missing" in wj.problem(name, cfg, profile=tmp_path)
    (tmp_path / "openclaw.json").write_text(json.dumps(wj.render("judge-b", cfg, "k", "t", tmp_path)))
    assert f"not {cfg['provider']}/judge-a" in wj.problem(name, cfg, profile=tmp_path)
    down = {**cfg, "proxy_port": 1}
    (tmp_path / "openclaw.json").write_text(json.dumps(wj.render(name, down, "k", "t", tmp_path)))
    assert "proxy not running" in wj.problem(name, down, profile=tmp_path)


def test_write_profile_needs_the_key_and_copies_the_workspace(judge_env, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "WEAVEBENCH_JUDGE", tmp_path)
    monkeypatch.delenv("WEAVEBENCH_JUDGE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="WEAVEBENCH_JUDGE_API_KEY"):
        wj.write_profile("judge-a")
    monkeypatch.setenv("WEAVEBENCH_JUDGE_API_KEY", "synthetic-judge-key")
    out = wj.write_profile("judge-a")
    profile = json.loads(out.read_text())
    assert oct(out.stat().st_mode & 0o777) == "0o600"
    assert (tmp_path / "template" / "workspace" / "AGENTS.md").is_file()
    token = profile["gateway"]["auth"]["token"]
    assert wj.write_profile("judge-a") == out and json.loads(out.read_text())["gateway"]["auth"]["token"] == token


def test_judge_cli_field(capsys):
    wj.main(["field", "proxy_port"])
    assert capsys.readouterr().out.strip() == str(common.protocol("weavebench")["judge"]["proxy_port"])
    with pytest.raises(SystemExit):
        wj.main(["nothing"])


# ------------------------------------------------------------------ WeaveBench judge retry proxy


def sse(*objs) -> bytes:
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs)


class Content:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    async def iter_any(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class Upstream:
    def __init__(self, chunks, status=200):
        self.content = Content(chunks)
        self.status = status
        self.headers = {"Content-Type": "text/event-stream"}
        self.closed = self.released = False

    def close(self):
        self.closed = True

    def release(self):
        self.released = True


class Response:
    def __init__(self, status, headers):
        self.status, self.headers = status, headers
        self.body = b""
        self.eof = False

    async def prepare(self, request):
        pass

    async def write(self, chunk):
        self.body += chunk

    async def write_eof(self):
        self.eof = True


def handle(monkeypatch, upstreams, *, max_wait=30):
    calls = []
    remaining = iter(upstreams)

    async def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        result = next(remaining)
        if isinstance(result, Exception):
            raise result
        return result

    async def read():
        return b'{"input":"same request"}'

    monkeypatch.setattr(proxy.web, "StreamResponse", Response)
    monkeypatch.setattr(proxy, "retry_delay", lambda attempt: 0)
    stats = dict(requests=0, retries=0, stream_errors=0, gave_up=0)
    req = SimpleNamespace(app={"session": SimpleNamespace(request=request), "upstream": "https://unused/v1",
                               "slots": asyncio.Semaphore(1), "max_wait": max_wait, "stats": stats},
                          headers={"Authorization": "Bearer fake", "Content-Length": "1"},
                          method="POST", path="/v1/responses", match_info={"tail": "responses"}, read=read)
    return asyncio.run(proxy.handle(req)), stats, calls


def test_proxy_event_classification():
    assert proxy.event_error(sse({"type": "response.created"})[:-2]) == (True, None)
    preamble, error = proxy.event_error(sse({"type": "error", "error": {"code": "billing_not_active"}})[:-2])
    assert not preamble and "billing_not_active" in error and proxy.RETRYABLE.search(error)
    _, error = proxy.event_error(sse({"type": "response.failed", "response": {"error": {
        "message": "Rate limit reached on tokens per min (TPM)"}}})[:-2])
    assert proxy.RETRYABLE.search(error)
    assert proxy.event_error(sse({"type": "response.output_text.delta", "delta": "hi"})[:-2]) == (False, None)
    assert proxy.event_error(sse({"choices": [{"delta": {"content": "x"}}]})[:-2]) == (False, None)
    _, error = proxy.event_error(sse({"error": {"message": "Invalid parameter"}})[:-2])
    assert error and not proxy.RETRYABLE.search(error)  # a request bug is passed through, not retried
    assert proxy.event_error(b"data: [DONE]") == (False, None)
    assert proxy.event_error(b": keepalive") == (True, None)


def test_proxy_retries_in_stream_errors_over_http(monkeypatch):
    from aiohttp import ClientSession, web
    from aiohttp.test_utils import TestServer

    monkeypatch.setattr(proxy, "retry_delay", lambda attempt: 0.01)
    calls = []
    bodies = [
        sse({"type": "response.created"}, {"type": "error", "error": {"code": "billing_not_active"}}),
        sse({"type": "response.created"}, {"type": "response.failed", "response": {"error": {"message": "Rate limit reached"}}}),
        sse({"type": "response.created"}, {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed"}),
    ]

    async def upstream_handler(request):
        calls.append(await request.read())
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(bodies[min(len(calls) - 1, len(bodies) - 1)])
        await resp.write_eof()
        return resp

    async def main():
        up_app = web.Application()
        up_app.router.add_post("/v1/responses", upstream_handler)
        async with TestServer(up_app) as up:
            args = SimpleNamespace(upstream=str(up.make_url("/v1")), max_wait=30, concurrency=2)
            async with TestServer(await proxy.make_app(args)) as px:
                async with ClientSession() as session:
                    async with session.post(px.make_url("/v1/responses"), data=b'{"x":1}') as r:
                        text = await r.text()
                    async with session.get(px.make_url("/_stats")) as r:
                        stats = await r.json()
        return text, stats

    text, stats = asyncio.run(main())
    assert len(calls) == 3 and all(c == b'{"x":1}' for c in calls)  # request bytes never altered
    assert '"delta": "ok"' in text and "billing_not_active" not in text
    assert stats["retries"] == 2 and stats["stream_errors"] == 2 and stats["gave_up"] == 0


def _transport_errors():
    from aiohttp import ClientPayloadError, ServerDisconnectedError

    return [TimeoutError, OSError, ClientPayloadError, ServerDisconnectedError]


@pytest.mark.parametrize("exception", _transport_errors())
@pytest.mark.parametrize("phase", ["connect", "head"])
def test_proxy_transport_failure_retries_without_forwarding(monkeypatch, exception, phase):
    failed = exception("not a rate-limit string")
    upstream = failed if phase == "connect" else Upstream([b": keepalive\n\n", failed])
    content = sse({"type": "response.output_text.delta", "delta": "ok"}) + b"data: [DONE]\n\n"
    success = Upstream([content])
    response, stats, calls = handle(monkeypatch, [upstream, success])
    assert response.status == 200 and response.body == content and response.eof
    assert stats["retries"] == 1 and stats["gave_up"] == 0
    assert stats["stream_errors"] == (phase == "head")
    assert len(calls) == 2 and calls[0] == calls[1]
    if phase == "head":
        assert upstream.closed and upstream.released
    assert success.released


@pytest.mark.parametrize("failure", [*(e("x") for e in _transport_errors()),
                                     sse({"type": "error", "error": {"code": "billing_not_active"}})])
def test_proxy_exhausted_stream_failure_returns_nonempty_502(monkeypatch, failure):
    upstream = Upstream([sse({"type": "response.created"}), failure])
    response, stats, calls = handle(monkeypatch, [upstream], max_wait=0)
    assert response.status == 502
    assert "retries exhausted" in json.loads(response.body)["error"]["message"]
    assert len(calls) == 1 and stats["gave_up"] == stats["stream_errors"] == 1
    assert upstream.closed and upstream.released


@pytest.mark.parametrize("separator", [b"\n", b"\r\n"])
def test_proxy_keepalives_do_not_hide_a_later_retryable_error(monkeypatch, separator):
    failed_bytes = (b": keepalive\n\n\n\ndata: \ndata: \n\n" + sse({"type": "response.created"})
                    + sse({"type": "error", "error": {"message": "Rate limit reached"}})).replace(b"\n", separator)
    failed = Upstream([bytes([byte]) for byte in failed_bytes])  # split at every byte
    success_bytes = b": keepalive\n\n" + sse({"choices": [{"delta": {"content": "ok"}}]}) + b"data: [DONE]\n\n"
    success = Upstream([bytes([byte]) for byte in success_bytes])
    response, stats, _ = handle(monkeypatch, [failed, success])
    assert response.body == success_bytes
    assert stats["retries"] == stats["stream_errors"] == 1


def test_proxy_forwards_nonretryable_error_unchanged(monkeypatch):
    content = b": ping\n\n" + sse({"type": "error", "error": {"message": "Invalid parameter"}})
    response, stats, _ = handle(monkeypatch, [Upstream([content])])
    assert response.status == 200 and response.body == content and not stats["retries"]


# ------------------------------------------------------------------ WeaveBench scores and runner


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def outcome(out, **extra):
    save(out / "outcome.json", {"status": "completed", "task_id": out.name, **extra})


def score(model, value=0.7, ok=True):
    return {"judge_model": model, "judge_ok": ok, "score": value if ok else None}


def wb_args(out, model):
    return SimpleNamespace(harness="cli", model="policy-model", judge_model=model, api_key="fake",
                           result_dir=out.parent, judge_workspace=out.parent / "workspace")


def test_rejudge_selection_is_per_judge_model(tmp_path):
    wb = runner("weavebench")
    out = tmp_path / "task"
    assert not wb.needs_rejudge(out, "judge-a")
    outcome(out)
    assert wb.needs_rejudge(out, "judge-a")
    save(out / "score.json", score("judge-b"))
    assert not wb.needs_rejudge(out, "judge-b") and wb.needs_rejudge(out, "judge-a")
    save(wb.judge_score_path(out, "judge-a"), score("judge-a", ok=False))
    assert wb.needs_rejudge(out, "judge-a")
    save(wb.judge_score_path(out, "judge-a"), score("judge-a"))
    assert not wb.needs_rejudge(out, "judge-a") and not wb.needs_rejudge(out, "judge-b")
    outcome(out, status="error")
    assert not wb.needs_rejudge(out, "judge-a")


def test_rejudge_keeps_the_scores_of_other_judges(monkeypatch, tmp_path):
    wb = runner("weavebench")
    out = tmp_path / "task"
    outcome(out, judge_model="judge-b", score=0.2, judge_ok=True)
    save(wb.judge_score_path(out, "judge-b"), score("judge-b", 0.2))
    save(out / "score.json", score("judge-b", 0.2))
    args = wb_args(out, "judge-a")
    args.tasks_info = {"task": {"task_id": "task", "category": "WEB"}}
    args.judge_slot = nullcontext()
    monkeypatch.setattr(wb, "judge", lambda *a: {"ok": True, "final_score": 0.9, "score_json": {"final_score": 0.9}})
    record = wb.rejudge(args, "task")
    assert record["judge_model"] == "judge-a" and record["score"] == 0.9 and record["rejudged"]
    assert wb.load_judge_score(out, "judge-b")["score"] == 0.2
    assert wb.load_judge_score(out, "judge-a")["scores"] == {"final_score": 0.9}
    assert json.loads((out / "score.json").read_text())["judge_model"] == "judge-a"
    assert json.loads((out / "score.json").read_text())["harness"] == "cli"


def test_summary_never_mixes_judges_or_untagged_scores(tmp_path):
    wb = runner("weavebench")
    for task in ("a", "b", "c", "d"):
        outcome(tmp_path / task, score=0.99, judge_ok=True)  # a stale outcome score must not leak in
    save(tmp_path / "a" / "score.json", score("judge-b", 0.1))
    save(wb.judge_score_path(tmp_path / "a", "judge-a"), score("judge-a", 0.8))
    save(tmp_path / "b" / "score.json", score("judge-b", 0.2))
    save(tmp_path / "c" / "score.json", {"score": 0.5, "judge_ok": True})
    save(tmp_path / "d" / "score.json", score("judge-a", ok=False))
    summary = wb.judge_summary(tmp_path, "judge-a")
    assert [r["score"] for r in summary] == [0.8, None, None, None]
    assert all(r["judge_model"] == "judge-a" for r in summary)
    assert [r["score"] for r in wb.judge_summary(tmp_path, "judge-b")] == [0.1, 0.2, None, None]


def test_concurrent_judges_keep_both_scores(monkeypatch, tmp_path):
    wb = runner("weavebench")
    out = tmp_path / "task"
    outcome(out)
    gate = Barrier(2)

    def fake_judge(args, task, out):
        gate.wait(timeout=5)
        return {"ok": True, "final_score": 0.2 if args.judge_model == "judge-b" else 0.8}

    monkeypatch.setattr(wb, "judge", fake_judge)
    task = {"task_id": "task", "category": "WEB"}
    models = ["judge-b", "judge-a"]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda model: wb.score_case(wb_args(out, model), task, out, {}), models))
    assert [wb.load_judge_score(out, model)["score"] for model in models] == [0.2, 0.8]
    assert json.loads((out / "score.json").read_text()) in results
    assert not [p for p in out.rglob(".*") if p.is_file()]


def test_failed_judge_records_scrubbed_stderr(monkeypatch, tmp_path):
    wb = runner("weavebench")
    out = tmp_path / "task"
    outcome(out)
    args = wb_args(out, "judge-a")
    args.judge_key = "judge-secret"
    monkeypatch.setattr(wb, "judge", lambda *a: {"ok": False, "error": "boom",
                                                  "openclaw_stderr_tail": "429 with fake and judge-secret"})
    record = {}
    result = wb.score_case(args, {"task_id": "task", "category": "WEB"}, out, record)
    assert result["score"] is None and not result["judge_ok"] and record["judge_ok"] is False
    assert "judge-secret" not in result["judge_stderr_tail"] and "fake" not in result["judge_stderr_tail"]


@pytest.mark.parametrize("other_run,other_judge", [(False, False), (False, True), (True, False)])
def test_concurrent_judge_staging_is_isolated(monkeypatch, tmp_path, other_run, other_judge):
    wb = runner("weavebench")
    from weavebench.eval.agent_judge import judge_runner

    tpl_profile, tpl_workspace = tmp_path / "template-profile", tmp_path / "template-workspace"
    tpl_workspace.mkdir()
    save(tpl_profile / "openclaw.json", {})
    iso_root = tmp_path / "shared-isolated-root"
    monkeypatch.setenv("AJ_ISOLATED_ROOT", str(iso_root))
    monkeypatch.setattr(wb, "TASKS_ROOT", tmp_path / "tasks")
    gate = Barrier(2)
    seen = []

    def fake_judge_one(stage, case_id, **kwargs):
        state, config, workspace = judge_runner._provision_isolated(
            case_id, Path(stage), tpl_profile, tpl_workspace, iso_root)
        marker = state / "owner"
        marker.write_text(case_id)
        seen.append((Path(stage), state, workspace))
        gate.wait(timeout=5)
        assert marker.read_text() == case_id
        assert (workspace / "_eval" / case_id / "chat.jsonl").read_text() == "rollout"
        assert json.loads(config.read_text())["agents"]["defaults"]["workspace"] == str(workspace)
        return {"ok": True, "final_score": 1}

    monkeypatch.setattr(judge_runner, "judge_one", fake_judge_one)
    out1 = tmp_path / "run1" / "task"
    out2 = tmp_path / ("run2" if other_run else "run1") / "task"
    for out in (out1, out2):
        out.mkdir(parents=True, exist_ok=True)
        (out / "chat.jsonl").write_text("rollout")
    args1, args2 = wb_args(out1, "judge-a"), wb_args(out2, "judge-b" if other_judge else "judge-a")
    args1.judge_workspace = args2.judge_workspace = tmp_path / "shared-staging"
    task = {"task_id": "task", "category": "WEB"}
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(wb.judge, args, task, out) for args, out in ((args1, out1), (args2, out2))]
        results = [future.result() for future in futures]
    assert len({result["case_id"] for result in results}) == 2
    assert all(seen[0][i] != seen[1][i] for i in range(3))
    assert os.environ["AJ_ISOLATED_ROOT"] == str(iso_root)


class _FakeGuest:
    def __init__(self, files):
        self.files, self.commands = files, []

    def command(self, command, timeout=60, check=True):
        self.commands.append(command)
        return "OK 12\n__RV_ARCHIVE_RC=0" if "tar czf" in command else "boom\n__RV_ARCHIVE_RC=3"

    def read(self, path):
        return self.files.get(path)


def test_archive_falls_back_to_ssh_when_the_rest_server_is_gone(tmp_path):
    import requests

    wb = runner("weavebench")
    rest_calls = []

    def rest_exec(env, cmd, shell=False, timeout=120):
        rest_calls.append(cmd)
        raise requests.ConnectionError("refused")

    def rest_fetch(env, remote, local):
        raise AssertionError("REST fetch must not be used for the SSH env")

    def archive_deliverables(env, out, client_password="x"):
        result = ds._vm_exec(env, ["bash", "-c", "tar czf /tmp/_results.tar.gz results/"], timeout=120)
        assert result["returncode"] == 0
        assert ds._vm_fetch(env, "/tmp/_results.tar.gz", out / "results.tar.gz")
        return True

    ds = SimpleNamespace(_vm_exec=rest_exec, _vm_fetch=rest_fetch, _archive_deliverables=archive_deliverables)
    guest = _FakeGuest({"/tmp/_results.tar.gz": b"tarbytes"})
    record = {}
    wb.archive(ds, SimpleNamespace(vm_ip="10.0.0.2", server_port=5000), guest, tmp_path, record)
    assert len(rest_calls) == 1 and record["archive_fallback"].startswith("ssh")
    assert (tmp_path / "results.tar.gz").read_bytes() == b"tarbytes"
    assert guest.commands[0].startswith("bash -c 'tar czf")
    assert ds._vm_exec(SimpleNamespace(ssh_guest=guest), ["false"])["returncode"] == 3  # exit status survives
    with pytest.raises(requests.ConnectionError):
        ds._vm_exec(SimpleNamespace(vm_ip="x", server_port=1), ["true"])


def test_ground_truth_scrub_fails_closed():
    wb = runner("weavebench")

    class Guest:
        def __init__(self, reply):
            self.reply, self.commands = reply, []

        def command(self, command, timeout=60, check=True):
            self.commands.append(command)
            return self.reply

    guest = Guest("/opt/task_gt\n/tmp/_warmup.sh\n")
    assert wb.scrub_vm_gt(guest) == ["/opt/task_gt", "/tmp/_warmup.sh"]
    assert "sudo -S" in guest.commands[0]
    with pytest.raises(RuntimeError, match="survived"):
        wb.scrub_vm_gt(Guest("LEFT:/opt/task_gt\n"))


def test_weavebench_config_requires_a_judge_unless_disabled(judge_env, monkeypatch, capsys, tmp_path):
    wb = runner("weavebench")
    monkeypatch.setenv("RV_LLM_API_KEY", "synthetic")
    base = ["--harness", "cli", "--model", "m", "--base_url", "https://x.invalid/v1", "--tasks", "a,b",
            "--result_dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        wb.config(base)
    assert "WeaveBench judge model is not set" in capsys.readouterr().err
    args = wb.config([*base, "--no_judge"])
    assert args.judge_config is None and args.tasks == ["a", "b"]
    with pytest.raises(SystemExit):
        wb.config([*base, "--no_judge", "--rejudge"])
    monkeypatch.setenv("WEAVEBENCH_JUDGE_API_KEY", "judge-key")
    args = wb.config([*base, "--judge_model", "judge-a"])
    assert (args.judge_model, args.judge_key) == ("judge-a", "judge-key")
    assert (args.judge_thinking, args.judge_timeout) == ("medium", 3600)


# ------------------------------------------------------------------ ALE


def test_ale_wall_timeout_keeps_the_task_card_cap():
    ale = runner("ale")
    args = SimpleNamespace(max_timeout=7200)
    assert ale.agent_wall_timeout(args, {"timeout_s": 1200}) == 1200
    assert ale.agent_wall_timeout(args, {"timeout_s": 10800}) == 7200
    assert ale.agent_wall_timeout(args, {}) == 7200


def test_ale_explicit_agent_cap_is_identical_across_tasks():
    ale = runner("ale")
    for budget in (100, 500):
        args = SimpleNamespace(max_timeout=7200, agent_timeout=21600, call_budget=budget)
        assert ale.agent_wall_timeout(args, {"timeout_s": 1200}) == 21600
        assert ale.agent_wall_timeout(args, {"timeout_s": 10800}) == 21600


def test_ale_config_defaults_and_reference_password(monkeypatch, capsys, tmp_path):
    ale = runner("ale")
    monkeypatch.setenv("RV_LLM_API_KEY", "synthetic")
    base = ["--harness", "OS-Helix", "--model", "m", "--base_url", "https://x.invalid/v1", "--result_dir", str(tmp_path)]
    monkeypatch.delenv("ALE_REFERENCE_ARCHIVE_PASSWORD", raising=False)
    with pytest.raises(SystemExit):
        ale.config(base)
    assert "ALE_REFERENCE_ARCHIVE_PASSWORD" in capsys.readouterr().err
    monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "public")
    args = ale.config(base)
    assert args.harness == "os_helix" and len(args.tasks) == 85
    assert (args.max_timeout, args.eval_timeout, args.call_budget) == (57600, 7200, 500)
    with pytest.raises(SystemExit):
        ale.config([*base, "--agent_timeout", "0"])


class _ShellGuest:
    """Runs guest commands in a local bash (sudo stripped), so the scrub script itself is exercised."""

    def __init__(self):
        self.commands = []

    def command(self, command, timeout=60, check=True):
        self.commands.append(command)
        local = command.replace("sudo -n ", "", 1)
        return subprocess.run(["bash", "-c", local], capture_output=True, text=True).stdout


def test_ale_scrub_references_removes_every_answer_and_fails_closed(tmp_path):
    ale = runner("ale")
    root = tmp_path / "agenthle"
    for task in ("demo/hello/base", "engineering/gcode/125162_319"):
        base = root / task
        (base / "input").mkdir(parents=True)
        (base / "input" / "data.csv").write_text("x")
        (base / "reference.7z").write_bytes(b"7z")
    (root / "demo/hello/base/reference").mkdir()
    (root / "demo/hello/base/reference/result.txt").write_text("secret")
    (root / "demo/hello/base/.reference_extract").mkdir()

    guest = _ShellGuest()
    removed = ale.scrub_references(guest, str(root))
    assert "sudo -n bash -c" in guest.commands[0]
    assert len(removed) == 4 and not any(line.startswith("LEFT:") for line in removed)
    assert not list(root.rglob("reference.7z")) and not list(root.rglob("reference"))
    assert (root / "demo/hello/base/input/data.csv").exists()  # inputs untouched
    assert ale.scrub_references(guest, str(root)) == []

    class Stubborn:
        def command(self, command, timeout=60, check=True):
            return "LEFT:/media/user/data/agenthle/x/y/base/reference.7z\n__RV_RC=0"

    with pytest.raises(RuntimeError, match="scrub failed"):
        ale.scrub_references(Stubborn(), "/media/user/data/agenthle")


def test_ale_needs_regrade_only_for_archived_grading_failures(tmp_path):
    ale = runner("ale")

    def case(name, status, archived=True):
        out = tmp_path / name
        out.mkdir()
        (out / "outcome.json").write_text(json.dumps({"status": status}))
        if archived:
            (out / "output.tar.gz").write_bytes(b"x")
        return out

    assert ale.needs_regrade(case("a", "eval_failed"))
    assert not ale.needs_regrade(case("b", "eval_failed", archived=False))
    assert not ale.needs_regrade(case("c", "completed"))
    assert not ale.needs_regrade(tmp_path / "missing")


def test_ale_judge_timeout_patch_is_idempotent(monkeypatch):
    import types

    ale = runner("ale")
    evaluation = types.ModuleType("tasks.utils.evaluation")
    evaluation._resolve_client_kwargs = lambda api_key=None, capability="vision": {"timeout": 120, "max_retries": 2}
    utils = types.ModuleType("tasks.utils")
    utils.evaluation = evaluation
    monkeypatch.setitem(sys.modules, "tasks", types.ModuleType("tasks"))
    monkeypatch.setitem(sys.modules, "tasks.utils", utils)
    monkeypatch.setitem(sys.modules, "tasks.utils.evaluation", evaluation)
    ale.install_judge_timeout()
    ale.install_judge_timeout()
    assert evaluation._resolve_client_kwargs(capability="vision") == {"timeout": ale.JUDGE_TIMEOUT_S, "max_retries": 2}
    assert ale.JUDGE_TIMEOUT_S == common.protocol("ale")["judge_timeout_s"] > 120


def test_ale_qemu_snapshot_is_pinned():
    snap = runner("ale").QEMU_SNAPSHOTS["cpu-free-ubuntu"]
    assert snap["image_revision"] == "40ba46b348a684b7d1b34d18e308aed54737f41a"
    assert snap["runner_image"] == "agentslastexam/ale-qemu:0.2.0" and snap["root"] == str(paths.ALE_QEMU_ROOT)


# ------------------------------------------------------------------ scripts


def test_rv_protocol_prints_null_as_empty():
    def protocol(bench, key):
        cmd = f'. "{REPO}/scripts/env.sh"; rv_protocol {bench} {key}'
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout.strip()

    assert protocol("osworld2", "evaluator_model") == ""
    assert protocol("osworld2", "call_budget") == "500"


ROUTE_VARS = [
    "OSWORLD_EVAL_MODEL_PROVIDER", "OSWORLD_EVAL_MODEL_NAME", "OSWORLD_EVAL_MODEL_BASE_URL",
    "OSWORLD_EVAL_MODEL_API_KEY_ENV", "OSWORLD_EVAL_MODEL_REASONING_EFFORT",
    "OSWORLD_USER_SIM_PROVIDER", "OSWORLD_USER_SIM_MODEL", "OSWORLD_USER_SIM_BASE_URL",
    "OSWORLD_USER_SIM_API_KEY_ENV", "OSWORLD_CLIENT_PASSWORD",
]


@pytest.fixture
def launcher(tmp_path):
    """Run scripts/run_osworld2.sh with synthetic credentials and fake git / uv (nothing is started)."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "run_osworld2.sh"
    script.write_text((REPO / "scripts/run_osworld2.sh").read_text())
    (scripts / "env.sh").write_text((REPO / "scripts/env.sh").read_text() + "\nrv_require_host_network() { :; }\n")
    credentials = tmp_path / "synthetic.env"
    credentials.write_text("OSWORLD_EVAL_API_KEY=synthetic-eval-key\nRV_LLM_API_KEY=synthetic-agent-key\n")
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "python3").symlink_to(sys.executable)
    for name, text in {
        "git": "#!/bin/sh\nprintf 'osworld-v2.1\\n'\n",
        "uv": f"#!{sys.executable}\nimport json, os, sys\n"
              f"print(json.dumps({{'env': {{k: os.environ[k] for k in {ROUTE_VARS!r} if k in os.environ}}, 'argv': sys.argv[1:]}}))\n",
    }.items():
        path = bins / name
        path.write_text(text)
        path.chmod(0o755)
    data = tmp_path / "data"
    for part in ("assets/osworld_v2/osworld_v2_assets_v2.1", "assets/sandbox/wheelhouse", "images/osworld_v2"):
        (data / part).mkdir(parents=True)
    (data / "images/osworld_v2/osworld-v2-ubuntu-x86.qcow2").touch()
    (tmp_path / "upstream").mkdir()
    env = {
        "PATH": str(bins) + os.pathsep + os.defpath, "HOME": str(tmp_path),
        "RV_REPO": str(REPO), "RV_OSWORLD": str(tmp_path / "upstream"), "RV_DATA": str(data),
        "RV_MODEL_CREDENTIALS": str(credentials),
        "OSWORLD_EVAL_MODEL_NAME": "eval-model", "OSWORLD_USER_SIM_MODEL": "sim-model",
        "OSWORLD_EVAL_MODEL_BASE_URL": "https://eval.invalid/v1",
    }

    def run(overrides=None):
        active = dict(env)
        for key, value in (overrides or {}).items():
            if value is None:
                active.pop(key, None)
            else:
                active[key] = value
        result = subprocess.run(["bash", str(script), "--harness", "mm_coding", "--model", "m"],
                                env=active, capture_output=True, text=True, timeout=20)
        assert "synthetic-eval-key" not in result.stdout + result.stderr
        assert "synthetic-agent-key" not in result.stdout + result.stderr
        return result

    return run


def test_launcher_routes_evaluator_and_simulator(launcher):
    result = launcher()
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["argv"] == ["run", "--no-sync", "python", "-m", "mmcoding.benchmarks.osworld2",
                              "--harness", "mm_coding", "--model", "m"]
    env = output["env"]
    for prefix in ("OSWORLD_EVAL_MODEL", "OSWORLD_USER_SIM"):
        assert env[prefix + "_PROVIDER"] == "openai_compatible"
        assert env[prefix + "_BASE_URL"] == "https://eval.invalid/v1"
        assert env[prefix + "_API_KEY_ENV"] == "OSWORLD_EVAL_API_KEY"
        assert prefix + "_REASONING_EFFORT" not in env
    assert (env["OSWORLD_EVAL_MODEL_NAME"], env["OSWORLD_USER_SIM_MODEL"]) == ("eval-model", "sim-model")
    assert env["OSWORLD_CLIENT_PASSWORD"] == "osworld-public-evaluation"


def test_launcher_separate_simulator_dialect(launcher):
    result = launcher({"OSWORLD_USER_SIM_PROVIDER": "anthropic"})
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)["env"]
    assert env["OSWORLD_EVAL_MODEL_BASE_URL"] == "https://eval.invalid/v1"
    assert "OSWORLD_USER_SIM_BASE_URL" not in env


@pytest.mark.parametrize("overrides,message", [
    ({"OSWORLD_EVAL_MODEL_NAME": None}, "Set the evaluator model"),
    ({"OSWORLD_USER_SIM_MODEL": None}, "Set the user simulator model"),
    ({"OSWORLD_EVAL_MODEL_BASE_URL": None}, "Set OSWORLD_EVAL_MODEL_BASE_URL"),
    ({"OSWORLD_EVAL_MODEL_PROVIDER": "other"}, "must be openai_compatible or anthropic"),
    ({"OSWORLD_EVAL_MODEL_API_KEY_ENV": "MISSING_TEST_KEY"}, "Credential referenced by"),
    ({"OSWORLD_USER_SIM_API_KEY_ENV": "bad-name"}, "must name an environment variable"),
    ({"OSWORLD_EVAL_MODEL_API_KEY": "synthetic-eval-key"}, "Unset OSWORLD_EVAL_MODEL_API_KEY"),
    ({"OSWORLD_USER_SIM_API_KEY": "synthetic-eval-key"}, "Unset OSWORLD_USER_SIM_API_KEY"),
    ({"OSWORLD_EVAL_MODEL_REASONING_EFFORT": "high"}, "upstream default thinking"),
])
def test_launcher_fails_closed_without_leaking_credentials(launcher, overrides, message):
    result = launcher(overrides)
    assert result.returncode == 2
    assert message in result.stderr
    assert result.stdout == ""

