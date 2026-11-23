"""The harness registry (mmcoding.harnesses) and the copies of its tables in the guest scripts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mmcoding import harnesses

GUEST = Path(harnesses.__file__).resolve().parent / "guest"
TABLES = ("HARNESSES", "X_HARNESSES", "BROKER", "BROKER_PROFILES")


def test_registry_ids_and_paper_names():
    assert harnesses.HARNESSES == ("gui", "os_helix", "mm_coding", "vision_tool", "cli")
    assert [harnesses.name(h) for h in harnesses.HARNESSES] == ["GUI", "OS-Helix", "MM-Coding", "Vision-Tool", "CLI"]
    for h in harnesses.HARNESSES:
        assert harnesses.get(h).id == h


@pytest.mark.parametrize("spelling,expected", [
    ("gui", "gui"), ("GUI", "gui"), ("OS-Helix", "os_helix"), ("os-helix", "os_helix"), (" MM-Coding ", "mm_coding"),
    ("Vision-Tool", "vision_tool"), ("vision_tool", "vision_tool"), ("CLI", "cli"),
])
def test_normalize_accepts_ids_and_paper_names(spelling, expected):
    assert harnesses.normalize(spelling) == expected


@pytest.mark.parametrize("unknown", ["desktop", "vision", "coding", "mm coding", "os_helix2", "G", ""])
def test_normalize_rejects_other_names(unknown):
    with pytest.raises(KeyError):
        harnesses.normalize(unknown)


def test_x_access_and_broker_per_harness():
    assert harnesses.X_HARNESSES == {"os_helix", "mm_coding", "vision_tool", "cli"}
    assert harnesses.BROKER == {"gui": "act", "os_helix": "act_guarded", "mm_coding": "observe"}
    assert set(harnesses.BROKER.values()) <= set(harnesses.BROKER_PROFILES)
    assert harnesses.BROKER_PROFILES == ("act", "act_guarded", "observe")
    assert harnesses.get("vision_tool").broker is None and harnesses.get("cli").broker is None


def test_os_helix_flag_and_image_bound():
    helix = harnesses.get("OS-Helix")
    assert helix.helix and helix.broker == "act_guarded" and helix.x_access
    assert helix.max_request_image_bytes == harnesses.IMAGE_BYTES_HELIX == 256 * 1024 * 1024
    for h in harnesses.HARNESSES:
        spec = harnesses.get(h)
        assert spec.helix == (h == "os_helix") and spec.vision_tool == (h == "vision_tool")
        if h != "os_helix":
            assert spec.max_request_image_bytes == harnesses.IMAGE_BYTES_DEFAULT == 8 * 1024 * 1024


def _guest_tables(path: Path) -> dict:
    """Top-level literal assignments of the registry tables in a guest script."""
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in TABLES:
                    try:
                        found[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return found


def test_guest_copies_of_the_tables_agree():
    expected = {"HARNESSES": harnesses.HARNESSES, "X_HARNESSES": set(harnesses.X_HARNESSES),
                "BROKER": harnesses.BROKER, "BROKER_PROFILES": harnesses.BROKER_PROFILES}
    copies = {path.name: _guest_tables(path) for path in sorted(GUEST.glob("*.py"))}
    assert {"HARNESSES", "X_HARNESSES", "BROKER"} <= set(copies["rv_run.py"])
    for name, tables in copies.items():
        for key, value in tables.items():
            value = set(value) if key == "X_HARNESSES" else (tuple(value) if key in ("HARNESSES", "BROKER_PROFILES") else value)
            assert value == expected[key], f"{name}: {key}"
