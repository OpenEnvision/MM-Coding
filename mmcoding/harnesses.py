"""The five harnesses of the paper: one registry for the host side, the guest and the runners.

All five share the policy model, the desktop, the task instruction, the call budget and the
evaluator. They differ only in the tools exposed to the policy and in the capability paragraph of
the system prompt that describes these tools (``mmcoding/prompts``).

=========== ============ ================================================================
id          paper name   what the policy receives and emits
=========== ============ ================================================================
gui         GUI          screenshots; one mouse/keyboard primitive per call. A terminal
                         window is one more application on the screen.
os_helix    OS-Helix     one GUI primitive or one block of Bash/Python per step; a screenshot
                         after every action; the ten most recent screenshots stay in context
mm_coding   MM-Coding    programs and file operations; screenshots and image files on request
vision_tool Vision-Tool  programs and file operations; a text-only ask_vision tool answered by
                         a separate, stateless, task-blind vision model
cli         CLI          programs and file operations; no perception tool
=========== ============ ================================================================

The guest-side scripts (``mmcoding/guest``) run inside the VM without this package and keep their
own copies of these tables; ``tests/test_harnesses.py`` checks that the copies agree.
"""

from __future__ import annotations

from dataclasses import dataclass

# Image history bound of the model route: beyond this much base64 image payload per request the
# oldest images become text placeholders. OS-Helix bounds its context by image count instead
# (guest/helix_shim.py keeps the ten most recent screenshots), so its byte bound is only a safety cap.
IMAGE_BYTES_DEFAULT = 8 * 1024 * 1024
IMAGE_BYTES_HELIX = 256 * 1024 * 1024


@dataclass(frozen=True)
class Harness:
    id: str
    name: str  # name used in the paper
    x_access: bool  # sandboxed code may use the X display (DISPLAY=:0)
    broker: str | None  # desktop broker tool profile (guest/rv_desktop.py --profile), None = no broker
    vision_tool: bool = False  # text-only ask_vision MCP tool (guest/vision_mcp.py)
    helix: bool = False  # one action per step, run_code tool and ten-screenshot context (guest/helix_*.py)
    max_request_image_bytes: int = IMAGE_BYTES_DEFAULT


HARNESS_LIST: tuple[Harness, ...] = (
    Harness("gui", "GUI", x_access=False, broker="act"),
    Harness("os_helix", "OS-Helix", x_access=True, broker="act_guarded", helix=True,
            max_request_image_bytes=IMAGE_BYTES_HELIX),
    Harness("mm_coding", "MM-Coding", x_access=True, broker="observe"),
    Harness("vision_tool", "Vision-Tool", x_access=True, broker=None, vision_tool=True),
    Harness("cli", "CLI", x_access=True, broker=None),
)

BY_ID = {h.id: h for h in HARNESS_LIST}
HARNESSES = tuple(BY_ID)
X_HARNESSES = frozenset(h.id for h in HARNESS_LIST if h.x_access)
BROKER = {h.id: h.broker for h in HARNESS_LIST if h.broker}

# Desktop broker tool profiles (guest/rv_desktop.py): every GUI action; every GUI action with code
# execution typed into terminals, run dialogs or developer consoles refused (OS-Helix runs code
# through run_code instead); screenshots only.
BROKER_PROFILES = ("act", "act_guarded", "observe")


def normalize(name: str) -> str:
    """Harness id for an id or a paper name in any case (``"MM-Coding"`` -> ``"mm_coding"``)."""
    key = name.strip().lower().replace("-", "_")
    if key in BY_ID:
        return key
    raise KeyError(f"unknown harness {name!r}; harnesses are {HARNESSES}")


def get(name: str) -> Harness:
    return BY_ID[normalize(name)]


def name(harness: str) -> str:
    """Paper name of a harness."""
    return get(harness).name
