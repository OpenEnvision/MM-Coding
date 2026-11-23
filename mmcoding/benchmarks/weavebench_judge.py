"""WeaveBench judge: the OpenClaw profile of the user's judge model and its local retry proxy.

WeaveBench scores a run with a trajectory-aware agent-as-judge (OpenClaw). The judge model is
user-supplied: ``--judge_model``, ``$WEAVEBENCH_JUDGE_MODEL`` or ``weavebench.judge_model`` in
``configs/protocol.json``. ``weavebench.judge`` holds the route: OpenClaw request adapter
(``api``, overridable with ``$WEAVEBENCH_JUDGE_API``), the environment variables with the
endpoint and the key, and the loopback port of the retry proxy (``weavebench_judge_proxy``) that
the profile talks to. The rendered profile holds the key, so it lives only under
``$RV_DATA/judge/weavebench/template/profiles/<judge>/`` and never in the repository.

CLI (stdlib only):
  python3 -m mmcoding.benchmarks.weavebench_judge field KEY          print one weavebench.judge setting
  python3 -m mmcoding.benchmarks.weavebench_judge profile [JUDGE]    print the profile directory
  python3 -m mmcoding.benchmarks.weavebench_judge render [JUDGE]     write the profile (needs the key env)
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

from mmcoding import paths

TEMPLATE_DIR = paths.REPO / "docker" / "weavebench_judge"
TEMPLATE = TEMPLATE_DIR / "openclaw.template.json"
PROFILE_DIRS = ("agents/main", "logs", "plugin-skills", "tasks")  # empty state dirs OpenClaw expects
MODEL_ENV = "WEAVEBENCH_JUDGE_MODEL"
API_ENV = "WEAVEBENCH_JUDGE_API"


def protocol() -> dict:
    return json.loads((paths.CONFIGS / "protocol.json").read_text())["weavebench"]


def judge(name: str | None = None) -> tuple[str, dict]:
    """(judge model, route settings); ``SystemExit`` with instructions when no judge model is set."""
    proto = protocol()
    name = name or os.getenv(MODEL_ENV) or proto.get("judge_model")
    if not name:
        raise SystemExit("WeaveBench judge model is not set: pass --judge_model, export "
                         f"{MODEL_ENV} or set \"judge_model\" in the \"weavebench\" block of configs/protocol.json")
    cfg = dict(proto["judge"])
    cfg["api"] = os.getenv(API_ENV) or cfg["api"]
    return name, cfg


def profile_dir(name: str) -> Path:
    return paths.WEAVEBENCH_JUDGE / "template" / "profiles" / quote(name, safe="")


def workspace_dir() -> Path:
    return paths.WEAVEBENCH_JUDGE / "template" / "workspace"


def proxy_url(cfg: dict) -> str:
    return f"http://127.0.0.1:{cfg['proxy_port']}/v1"


def render(name: str, cfg: dict, key: str, token: str, workspace: Path) -> dict:
    """The template with every placeholder substituted inside decoded JSON strings.

    Substituting decoded strings keeps keys, tokens and paths literal (quotes, backslashes and
    placeholder-like text in a value are never reinterpreted).
    """
    template = json.loads(TEMPLATE.read_text())
    values = {
        "PLACEHOLDER_WORKSPACE_DIR": str(workspace),
        "PLACEHOLDER_GATEWAY_TOKEN": token,
        "PLACEHOLDER_PROXY_URL": proxy_url(cfg),
        "PLACEHOLDER_JUDGE_MODEL": name,
        "PLACEHOLDER_PROVIDER": cfg["provider"],
        "PLACEHOLDER_API": cfg["api"],
        "PLACEHOLDER_API_KEY": key,
    }
    pattern = re.compile("|".join(sorted(values, key=len, reverse=True)))

    def replace(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: values[match.group()], value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {replace(k): replace(item) for k, item in value.items()}
        return value

    profile = replace(template)
    for model in profile["models"]["providers"][cfg["provider"]]["models"]:
        model["contextWindow"], model["maxTokens"] = cfg["context_window"], cfg["max_tokens"]
    return profile


def primary_model(profile: Path) -> str | None:
    """``<provider>/<model>`` the profile's judge agent runs (None without a profile)."""
    path = Path(profile) / "openclaw.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())["agents"]["defaults"]["model"]["primary"]


def problem(name: str, cfg: dict, profile: Path | None = None) -> str | None:
    """Why this judge cannot run now (missing or mismatched profile, proxy down), or None."""
    import urllib.request

    profile = profile or profile_dir(name)
    want = f"{cfg['provider']}/{name}"
    have = primary_model(profile)
    if have is None:
        return f"judge profile missing: scripts/render_judge_profile.py {name}"
    if have != want:
        return f"judge profile {profile} runs {have}, not {want}: scripts/render_judge_profile.py {name}"
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{cfg['proxy_port']}/_stats", timeout=5).read()
    except OSError:
        return f"judge proxy not running on :{cfg['proxy_port']}: scripts/judge_proxy.sh"
    return None


def write_profile(name: str | None = None) -> Path:
    """Render the profile of a judge model and copy the template workspace next to it."""
    name, cfg = judge(name)
    key = os.environ.get(cfg["api_key_env"])
    if not key:
        raise SystemExit(f"{cfg['api_key_env']} is not set (export it or add it to the credentials file)")
    workspace = workspace_dir()
    if not workspace.is_dir():
        shutil.copytree(TEMPLATE_DIR / "workspace", workspace)
    root = profile_dir(name)
    out = root / "openclaw.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    token = (old.get("gateway") or {}).get("auth", {}).get("token") or secrets.token_hex(24)
    profile = render(name, cfg, key, token, workspace)
    for sub in PROFILE_DIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
        (root / sub / ".gitkeep").touch()
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(profile, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(out)
    return out


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in ("field", "profile", "render") or (argv[0] == "field" and len(argv) != 2):
        raise SystemExit(__doc__)
    command, rest = argv[0], argv[1:]
    if command == "field":
        print(protocol()["judge"][rest[0]])
    elif command == "profile":
        print(profile_dir(judge(rest[0] if rest else None)[0]))
    else:
        name, cfg = judge(rest[0] if rest else None)
        out = write_profile(name)
        print(f"wrote {out} ({cfg['provider']}/{name}, {cfg['api']}, via proxy :{cfg['proxy_port']})")


if __name__ == "__main__":
    main(sys.argv[1:])
