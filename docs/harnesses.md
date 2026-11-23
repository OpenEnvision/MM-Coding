# Harnesses

All five harnesses run the same agent loop with the same policy model, desktop, task instruction,
budget and evaluator. They differ only in the tools exposed to the policy and in the capability
paragraph of the system prompt that describes these tools (`mmcoding/prompts/capability/`). Any
difference in score therefore comes from how the policy acts and perceives.

| Harness | Action per call | Code execution | Visual input to the policy | Images kept in context |
|---|---|---|---|---|
| GUI | one screen primitive | typed into a terminal window | screenshot after every action | up to 8 MB per request |
| OS-Helix | one primitive or one code block | one code block per step | screenshot after every action | ten most recent screenshots |
| MM-Coding | programs and file operations | direct | screenshot or image file on request | requested images, up to 8 MB |
| Vision-Tool | programs and file operations | direct | text answers of the vision model | none |
| CLI | programs and file operations | direct | none | none |

Shared settings (`configs/protocol.json`): a 1920x1080 desktop, the policy at its maximum reasoning
effort with a 256K context window and up to 64K output tokens per call, and a budget of 500 model
calls. A run ends when the policy finishes, reports the task as infeasible
(`report_infeasible`), or reaches the budget. On OSWorld 2.0 every harness also receives
`ask_user`, through which the benchmark's simulated user answers questions.

## GUI

The policy clicks, moves the pointer, drags, scrolls, types text, presses keys and waits, with
coordinates in pixels of the 1920x1080 screen, and every action returns a new screenshot. It has no
shell or file tools of its own. A terminal is one more application on the screen, so a program runs
only after the policy types it into a terminal window, and its output reaches the policy only as a
screenshot. Screenshots stay in the context up to 8 MB of images per request, and older screenshots
become short text placeholders.

Tools: `mcp__desktop__{screenshot,click,move,drag,scroll,type_text,key,wait}` served by
`guest/rv_desktop.py --profile act`.

## OS-Helix

The policy takes exactly one action per step, either one screen primitive of the GUI harness or one
block of Bash or Python that runs with access to the display (`mcp__helix__run_code`,
`guest/helix_code_mcp.py`). Every action returns a screenshot, and a code block also returns its
output and exit status. `guest/helix_shim.py` sits between the agent loop and the model endpoint: it
keeps the ten most recent screenshots (older ones become text placeholders while actions and outputs
remain), keeps the initial screenshot in the first request, and disables parallel tool calls. Typing
code into terminals, run dialogs and developer consoles is refused by the desktop broker
(`--profile act_guarded`), so code runs through `run_code` only.

## MM-Coding, Vision-Tool and CLI

The three share the action end. The policy acts through a shell and file tools (`bash`, `read`,
`write`, `edit`, `glob`, `grep`). Processes it starts reach the display (`DISPLAY=:0`), so its
programs can launch applications, send mouse and keyboard input with tools such as `xdotool` and
`pyautogui`, and capture the screen. They differ only in the perception tool.

* **MM-Coding.** A screenshot tool (`mcp__desktop__screenshot`, `--profile observe`) shows the
  current screen and `read_image` shows an image file. An image enters the context only through
  these calls.
* **Vision-Tool.** The policy cannot view images. `mcp__vision__ask_vision` (`guest/vision_mcp.py`)
  sends a question and a list of images (`"screen"` or paths of image files) to a separate vision
  model and returns text. Each call is stateless and contains only the system prompt
  (`mmcoding/prompts/vision_model.txt`), the images and the question, so the vision model knows
  neither the task nor any earlier call. It reports only what is visible and marks every location
  as a point or a box, and the tool converts these locations to pixels of the original image. The
  paper uses Qwen3.8-27B with thinking disabled; any OpenAI-compatible vision model can be set with
  `--vision_model`, optionally on its own endpoint (`RV_VISION_BASE_URL`, `RV_VISION_API_KEY`).
* **CLI.** No perception tool. Programs can still capture the screen and process the pixels, for
  example with OCR, and only their text output enters the context.

## Sandbox

Inside the VM, `guest/rv_run.py` (installed as `/usr/local/sbin/rv-run`) runs the agent as a
transient system unit in its own network namespace with a filtered session bus. The X display is
reachable only for the harnesses whose code may use it (all but GUI); the desktop broker runs
outside the sandbox and is mounted through a Unix socket. Evaluator-only files that a task's setup
places in the VM can be hidden from the sandbox (`RV_TASK_PRIVATE`). Services that must outlive the
run are started with `rv-serve`. On OSWorld 2.0, `mmcoding/host/audit.py` checks after each run
that no forbidden channel was used, for example that CLI never received an image.

## Network guard

With `--network-guard`, every outbound TCP connection of the task VM goes through a transparent
mitmproxy instance on the host (`mmcoding/host/guard.py`, `guard_addon.py`) that blocks public
sources of benchmark tasks, reference answers and evaluators and logs every request. It is active
from the start of the agent until the agent has stopped, so task setup and evaluation are unchanged.
The paper's runs used it to prevent reward hacking.
