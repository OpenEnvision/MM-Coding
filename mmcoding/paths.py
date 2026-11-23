"""Every location the project uses, in one place.

* ``REPO``         this repository (code and configs);
* ``THIRD_PARTY``  pinned, unmodified upstream checkouts (OSWorld-V2, WeaveBench, ALE);
* ``DATA``         every large file: VM images, sandbox assets, judge state and results.

Each location can be overridden with an environment variable (``RV_DATA``, ``RV_THIRD_PARTY``,
...). Nothing here touches the filesystem at import time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser() if value else default


REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
TASKS = CONFIGS / "tasks"

THIRD_PARTY = _env("RV_THIRD_PARTY", REPO / "third_party")
OSWORLD = _env("RV_OSWORLD", THIRD_PARTY / "OSWorld-V2")
WEAVEBENCH = _env("RV_WEAVEBENCH", THIRD_PARTY / "WeaveBench")
ALE = _env("RV_ALE", THIRD_PARTY / "agents-last-exam")

DATA = _env("RV_DATA", REPO / "data")
IMAGES = DATA / "images"
OSWORLD_V2_IMAGE = _env("RV_OSWORLD_V2_IMAGE", IMAGES / "osworld_v2" / "osworld-v2-ubuntu-x86.qcow2")
WEAVEBENCH_IMAGE = _env("RV_WEAVEBENCH_IMAGE", IMAGES / "weavebench" / "Ubuntu_rv.qcow2")
WEAVEBENCH_BASE_IMAGE = IMAGES / "weavebench" / "Ubuntu.qcow2"
ALE_QEMU_ROOT = _env("RV_ALE_QEMU_ROOT", IMAGES / "ale" / "qemu")

ASSETS = DATA / "assets"
SANDBOX_ASSETS = _env("RV_ASSET_DIR", ASSETS / "sandbox")  # wheelhouse/ + debs/ installed in every guest
OSWORLD_V2_ASSETS = _env("OSWORLD_FILE_BASE_URL", ASSETS / "osworld_v2" / "osworld_v2_assets_v2.1")
WEAVEBENCH_TASKS = _env("WB_TASKS_ROOT", ASSETS / "weavebench" / "tasks")

JUDGE = DATA / "judge"
WEAVEBENCH_JUDGE = JUDGE / "weavebench"

RESULTS = _env("RV_RESULTS", DATA / "results")
# Optional, user-supplied: evaluator-only paths of OSWorld 2.0 tasks that are hidden from the agent
# sandbox (the task classes are gated, so this file is not distributed). Missing file = no paths.
OSWORLD_V2_TASK_PRIVATE = _env("RV_TASK_PRIVATE", DATA / "private" / "osworld_v2_task_private.json")

# Machine-local secrets: never in the repository.
MODEL_CREDENTIALS = _env("RV_MODEL_CREDENTIALS", Path.home() / ".config" / "mm-coding" / "credentials.env")
SSH_KEY = _env("RV_SSH_KEY", Path.home() / ".ssh" / "mm_coding_vm_ed25519")

UPSTREAM = {"osworld": OSWORLD, "weavebench": WEAVEBENCH, "ale": ALE}


def use_upstream(*names: str) -> None:
    """Put the named upstream checkouts on ``sys.path`` (first name wins on conflicts)."""
    for name in reversed(names):
        root = str(UPSTREAM[name])
        if root not in sys.path:
            sys.path.insert(0, root)


def result_dir(bench: str, model: str, run: str, harness: str) -> Path:
    """Default result location: ``RESULTS/<bench>/<model>/<run>/<harness>``."""
    return RESULTS / bench / model / run / harness


def task_file(bench: str) -> Path:
    """The complete task list of a benchmark."""
    return TASKS / f"{bench}.json"
