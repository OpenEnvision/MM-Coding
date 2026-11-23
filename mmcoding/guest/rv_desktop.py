#!/usr/bin/env python3
"""Desktop MCP broker of the screenshot harnesses (guest side).

Runs OUTSIDE the agent sandbox as the desktop user, with X11 access, and serves
the MCP protocol (newline-delimited JSON-RPC, i.e. the MCP stdio framing) on a
unix socket. Inside the sandbox, DSH mounts it through a dumb byte relay
(``socat STDIO UNIX-CONNECT:<socket>``), so the sandbox itself never needs X.

Tool profiles (fixed per VM by ``--profile``, see mmcoding/harnesses.py BROKER):

* ``act`` (harness gui): ``screenshot`` plus the atomic actions ``click``,
  ``move``, ``drag``, ``scroll``, ``type_text``, ``key``, ``wait``. Every action
  returns a fresh screenshot. Pixel coordinates only: no accessibility tree, no
  DOM, no element ids, no clipboard, no window management. Terminals, run
  dialogs and consoles are ordinary apps.
* ``act_guarded`` (harness os_helix): the same tools with the guard below.
* ``observe`` (harness mm_coding): ``screenshot`` only (observation, no input).

Guard (``act_guarded``): code execution through the GUI is not allowed. Keyboard input is
refused while a terminal or developer-console window has focus, pointer clicks
into terminal windows are refused, and key combos that open a run dialog,
browser developer tools, an IDE terminal or a macro IDE are refused. Every
request (allowed or refused) is appended to ``--log`` as JSON lines.

Standard library + python-xlib + Pillow (both preinstalled in the guest image).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import socket
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any

PROTOCOL = "2025-06-18"
SCREEN_W, SCREEN_H = 1920, 1080

TERMINAL_CLASS_RE = re.compile(
    r"(term|tilix|konsole|kitty|alacritty|urxvt|guake|tilda|yakuake|^st(-256color)?$)", re.I
)
CONSOLE_TITLE_RE = re.compile(
    r"(^DevTools\b|Developer Tools|Web Console|Browser Console|Error Console|Python Console|"
    r"Python-Fu|Script-Fu Console|Basic IDE|Macro Organizer|Run a Command)",
    re.I,
)
BROWSER_CLASS_RE = re.compile(r"(chrome|chromium|firefox|code|electron|vivaldi|brave|opera)", re.I)
VSCODE_CLASS_RE = re.compile(r"(^|\s)code(\s|$)|visual studio code|vscodium", re.I)
OFFICE_CLASS_RE = re.compile(r"(libreoffice|soffice)", re.I)

KEY_ALIASES = {
    "enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "tab": "Tab",
    "space": "space", "spacebar": "space", "backspace": "BackSpace", "delete": "Delete", "del": "Delete",
    "insert": "Insert", "ins": "Insert", "home": "Home", "end": "End",
    "pageup": "Prior", "page_up": "Prior", "pgup": "Prior", "prior": "Prior",
    "pagedown": "Next", "page_down": "Next", "pgdn": "Next", "next": "Next",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "arrowup": "Up", "arrowdown": "Down", "arrowleft": "Left", "arrowright": "Right",
    "ctrl": "ctrl", "control": "ctrl", "ctl": "ctrl", "alt": "alt", "option": "alt", "shift": "shift",
    "super": "super", "win": "super", "windows": "super", "meta": "super", "cmd": "super", "command": "super",
    "capslock": "Caps_Lock", "caps_lock": "Caps_Lock", "printscreen": "Print", "print": "Print",
    "menu": "Menu", "`": "grave", "grave": "grave", "backtick": "grave", "minus": "minus", "-": "minus",
    "plus": "plus", "equal": "equal", "=": "equal", "comma": "comma", ",": "comma", "period": "period",
    ".": "period", "slash": "slash", "/": "slash", "backslash": "backslash", "\\": "backslash",
    "semicolon": "semicolon", ";": "semicolon", "apostrophe": "apostrophe", "'": "apostrophe",
    "bracketleft": "bracketleft", "[": "bracketleft", "bracketright": "bracketright", "]": "bracketright",
}
MODIFIERS = {"ctrl", "alt", "shift", "super"}

# (modifier set, key) pairs refused regardless of focus / for focus classes.
GLOBAL_BLOCKED = {(frozenset({"alt"}), "F2")}  # GNOME "Run a Command" dialog
BROWSER_BLOCKED = {
    (frozenset(), "F12"),
    (frozenset({"ctrl", "shift"}), "i"), (frozenset({"ctrl", "shift"}), "j"),
    (frozenset({"ctrl", "shift"}), "c"), (frozenset({"ctrl", "shift"}), "k"),
}
VSCODE_BLOCKED = {
    (frozenset({"ctrl"}), "grave"), (frozenset({"ctrl", "shift"}), "grave"),
    (frozenset({"ctrl", "shift"}), "c"),  # open external terminal
}
OFFICE_BLOCKED = {(frozenset({"alt"}), "F11")}  # macro organizer / Basic IDE


class Refused(Exception):
    """An action the guard does not allow; reported to the model as a tool error."""


class BadArgs(Exception):
    """Invalid tool arguments; reported to the model as a tool error."""


# ------------------------------------------------------------------- key parsing


def normalize_token(token: str) -> str:
    raw = token.strip()
    if not raw:
        raise BadArgs("empty key name")
    low = raw.lower()
    if low in KEY_ALIASES:
        return KEY_ALIASES[low]
    if re.fullmatch(r"f([1-9]|1[0-9]|2[0-4])", low):
        return low.upper()
    if len(raw) == 1:
        return raw
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", raw):
        return raw  # assume an X keysym name (e.g. KP_Enter, XF86AudioMute)
    raise BadArgs(f"unknown key name {token!r}")


def parse_combo(combo: str) -> tuple[frozenset[str], str, str]:
    """Return (modifiers, key, xdotool_combo) for one combo like 'ctrl+shift+t'."""
    combo = combo.strip()
    if combo.endswith("++"):
        parts = combo[:-2].split("+") + ["plus"]
    else:
        parts = combo.split("+")
    tokens = [normalize_token(p) for p in parts if p.strip()]
    if not tokens:
        raise BadArgs("empty key combo")
    mods = [t for t in tokens[:-1]]
    for m in mods:
        if m not in MODIFIERS:
            raise BadArgs(f"{m!r} is not a modifier in combo {combo!r}")
    key = tokens[-1]
    mod_set = frozenset(mods)
    key_cmp = key.lower() if len(key) == 1 else key
    return mod_set, key_cmp, "+".join(mods + [key])


def parse_keys(keys: str) -> list[tuple[frozenset[str], str, str]]:
    if not isinstance(keys, str) or not keys.strip():
        raise BadArgs("keys must be a non-empty string such as 'ctrl+s' or 'Return'")
    combos = keys.split()
    if len(combos) > 20:
        raise BadArgs("at most 20 key combos per call")
    return [parse_combo(c) for c in combos]


def combo_refusal(mods: frozenset[str], key: str, focus_class: str) -> str | None:
    pair = (mods, key)
    if pair in GLOBAL_BLOCKED:
        return "the run-command dialog (Alt+F2) is not allowed"
    if BROWSER_CLASS_RE.search(focus_class or "") and pair in BROWSER_BLOCKED:
        return "opening developer tools / consoles is not allowed"
    if VSCODE_CLASS_RE.search(focus_class or "") and pair in VSCODE_BLOCKED:
        return "opening an IDE terminal is not allowed"
    if OFFICE_CLASS_RE.search(focus_class or "") and pair in OFFICE_BLOCKED:
        return "opening the macro IDE is not allowed"
    return None


def focus_refusal(focus_class: str, focus_title: str) -> str | None:
    if TERMINAL_CLASS_RE.search(focus_class or ""):
        return "keyboard input into a terminal window is not allowed"
    if CONSOLE_TITLE_RE.search(focus_title or ""):
        return "keyboard input into a developer console / run dialog is not allowed"
    return None


# -------------------------------------------------------------------- X helpers


class Desktop:
    def __init__(self, settle: float) -> None:
        self.settle = settle
        self.lock = threading.Lock()
        self._display = None

    # -- X connection (python-xlib; reconnect on failure)
    def display(self):
        if self._display is None:
            from Xlib import display as xdisplay

            d = xdisplay.Display()
            try:
                d.xfixes_query_version()
            except Exception:
                pass
            self._display = d
        return self._display

    def _reset(self) -> None:
        try:
            if self._display is not None:
                self._display.close()
        except Exception:
            pass
        self._display = None

    def _x(self, fn):
        try:
            return fn(self.display())
        except Exception:
            self._reset()
            return fn(self.display())

    # -- window info
    @staticmethod
    def _wm_info(win) -> tuple[str, str]:
        cls = ""
        title = ""
        try:
            wc = win.get_wm_class()
            if wc:
                cls = " ".join(x for x in wc if x)
        except Exception:
            pass
        try:
            d = win.display
            prop = win.get_full_property(d.intern_atom("_NET_WM_NAME"), d.intern_atom("UTF8_STRING"))
            if prop and prop.value:
                title = prop.value.decode("utf-8", "replace") if isinstance(prop.value, bytes) else str(prop.value)
            else:
                name = win.get_wm_name()
                title = name if isinstance(name, str) else ""
        except Exception:
            pass
        return cls, title

    def active_window(self) -> dict[str, Any]:
        def go(d):
            root = d.screen().root
            prop = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), 0)
            if not prop or not prop.value or not prop.value[0]:
                return {"id": 0, "class": "", "title": ""}
            win = d.create_resource_object("window", int(prop.value[0]))
            cls, title = self._wm_info(win)
            return {"id": int(prop.value[0]), "class": cls, "title": title}

        try:
            return self._x(go)
        except Exception as exc:
            return {"id": 0, "class": "", "title": "", "error": str(exc)[:200]}

    def window_under_pointer(self) -> dict[str, Any]:
        def go(d):
            win = d.screen().root
            for _ in range(12):
                child = win.query_pointer().child
                if not child:
                    break
                win = child
                cls, title = self._wm_info(win)
                if cls:
                    return {"id": int(win.id), "class": cls, "title": title}
            return {"id": int(win.id), "class": "", "title": ""}

        try:
            return self._x(go)
        except Exception as exc:
            return {"id": 0, "class": "", "title": "", "error": str(exc)[:200]}

    def cursor_position(self) -> tuple[int, int]:
        def go(d):
            p = d.screen().root.query_pointer()
            return int(p.root_x), int(p.root_y)

        return self._x(go)

    # -- screenshot (root window + composited cursor)
    def screenshot_png(self) -> tuple[bytes, tuple[int, int]]:
        from PIL import Image
        from Xlib import X

        def go(d):
            root = d.screen().root
            geo = root.get_geometry()
            raw = root.get_image(0, 0, geo.width, geo.height, X.ZPixmap, 0xFFFFFFFF)
            img = Image.frombytes("RGB", (geo.width, geo.height), raw.data, "raw", "BGRX")
            cursor = None
            try:
                cursor = d.xfixes_get_cursor_image(root)
            except Exception:
                cursor = None
            return img, cursor

        img, cursor = self._x(go)
        pos = (0, 0)
        if cursor is not None:
            try:
                w, h = int(cursor.width), int(cursor.height)
                pixels = list(cursor.cursor_image)
                buf = bytearray()
                for p in pixels[: w * h]:
                    a = (p >> 24) & 0xFF
                    r, g, b = (p >> 16) & 0xFF, (p >> 8) & 0xFF, p & 0xFF
                    if 0 < a < 255:  # un-premultiply
                        r, g, b = min(255, r * 255 // a), min(255, g * 255 // a), min(255, b * 255 // a)
                    buf += bytes((r, g, b, a))
                cimg = Image.frombytes("RGBA", (w, h), bytes(buf))
                pos = (int(cursor.x), int(cursor.y))
                img.paste(cimg, (pos[0] - int(cursor.xhot), pos[1] - int(cursor.yhot)), cimg)
            except Exception:
                pass
        out = io.BytesIO()
        img.save(out, format="PNG", compress_level=3)
        return out.getvalue(), pos

    # -- input through xdotool (keysym remapping handles arbitrary text)
    @staticmethod
    def xdotool(*args: str, timeout: float = 60) -> None:
        proc = subprocess.run(["xdotool", *args], capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"xdotool {' '.join(args[:3])} failed: {proc.stderr.strip()[:200]}")


# ------------------------------------------------------------------ tool schemas


def _int(desc: str) -> dict:
    return {"type": "integer", "description": desc}


XY = {"x": _int("X coordinate in screen pixels (0-1919)."), "y": _int("Y coordinate in screen pixels (0-1079).")}

TOOLS: dict[str, dict] = {
    "screenshot": {
        "description": "Capture the full screen (1920x1080 pixels, mouse cursor drawn). Returns the image.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    "click": {
        "description": (
            "Move the mouse to (x, y) in screen pixels and click. button: left (default), right or middle; "
            "count: 1 (default), 2 for double-click, 3 for triple-click. Returns a screenshot taken after the click."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **XY,
                "button": {"type": "string", "enum": ["left", "right", "middle"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 3},
            },
            "required": ["x", "y"],
            "additionalProperties": False,
        },
    },
    "move": {
        "description": "Move the mouse to (x, y) in screen pixels without clicking (hover). Returns a screenshot.",
        "inputSchema": {"type": "object", "properties": XY, "required": ["x", "y"], "additionalProperties": False},
    },
    "drag": {
        "description": (
            "Press the left mouse button at (x1, y1), move to (x2, y2) and release. Returns a screenshot."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "x1": _int("Start X (screen pixels)."), "y1": _int("Start Y (screen pixels)."),
                "x2": _int("End X (screen pixels)."), "y2": _int("End Y (screen pixels)."),
            },
            "required": ["x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
    },
    "scroll": {
        "description": (
            "Move the mouse to (x, y) and scroll the wheel. direction: up, down, left or right; "
            "amount: number of wheel clicks (1-20, default 3). Returns a screenshot."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                **XY,
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"]},
                "amount": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["x", "y", "direction"],
            "additionalProperties": False,
        },
    },
    "type_text": {
        "description": (
            "Type text with the keyboard into the focused window (newline = Enter, tab = Tab). "
            "Returns a screenshot."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to type (max 4000 characters)."}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    "key": {
        "description": (
            "Press a key or key combination, e.g. 'Return', 'Escape', 'ctrl+s', 'alt+Tab', 'ctrl+shift+t'. "
            "Several combos separated by spaces are pressed in order. Returns a screenshot."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"keys": {"type": "string", "description": "Key combo(s), xdotool style."}},
            "required": ["keys"],
            "additionalProperties": False,
        },
    },
    "wait": {
        "description": "Wait for the given number of seconds (0.5-30, default 2), then return a screenshot.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number", "minimum": 0.5, "maximum": 30}},
            "additionalProperties": False,
        },
    },
}
ACTION_TOOLS = list(TOOLS)
# Tool profiles (mmcoding/harnesses.py BROKER_PROFILES): all actions; all actions with code
# execution through terminals / run dialogs / consoles refused; screenshots only.
PROFILE_TOOLS = {"act": ACTION_TOOLS, "act_guarded": ACTION_TOOLS, "observe": ["screenshot"]}
GUARDED_PROFILES = frozenset({"act_guarded"})


# ----------------------------------------------------------------------- server


class Broker:
    def __init__(self, profile: str, log_path: str, settle: float) -> None:
        if profile not in PROFILE_TOOLS:
            raise ValueError(f"profile must be one of {sorted(PROFILE_TOOLS)}")
        self.profile = profile
        self.guarded = profile in GUARDED_PROFILES
        self.tools = PROFILE_TOOLS[profile]
        self.desk = Desktop(settle)
        self.log_path = log_path
        self.log_lock = threading.Lock()
        self.seq = 0

    def log(self, record: dict) -> None:
        with self.log_lock:
            self.seq += 1
            record = {"seq": self.seq, "t": round(time.time(), 3), **record}
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- argument validation
    @staticmethod
    def _coord(args: dict, key: str, limit: int) -> int:
        value = args.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadArgs(f"{key} must be an integer")
        value = int(round(value))
        if not 0 <= value < limit:
            raise BadArgs(f"{key}={value} is outside the screen (0..{limit - 1})")
        return value

    def _shot_content(self, note: str) -> list[dict]:
        png, pos = self.desk.screenshot_png()
        text = f"{note} Screen 1920x1080; cursor at ({pos[0]}, {pos[1]})."
        return [
            {"type": "text", "text": text.strip()},
            {"type": "image", "data": base64.b64encode(png).decode("ascii"), "mimeType": "image/png"},
        ]

    def _settle(self, extra: float = 0.0) -> None:
        time.sleep(self.desk.settle + extra)

    # -- tools
    def call(self, name: str, args: dict) -> tuple[list[dict], bool, dict]:
        if name not in self.tools:
            raise BadArgs(f"unknown tool {name!r}")
        info: dict[str, Any] = {}
        d = self.desk
        if name == "screenshot":
            return self._shot_content("Screenshot."), False, info
        if name == "wait":
            seconds = args.get("seconds", 2)
            if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0.5 <= seconds <= 30:
                raise BadArgs("seconds must be a number between 0.5 and 30")
            time.sleep(float(seconds))
            return self._shot_content(f"Waited {seconds} s."), False, info
        if name in {"click", "move", "scroll"}:
            x, y = self._coord(args, "x", SCREEN_W), self._coord(args, "y", SCREEN_H)
            d.xdotool("mousemove", "--sync", str(x), str(y))
            target = d.window_under_pointer()
            info["target"] = target
            if name == "move":
                self._settle()
                return self._shot_content(f"Moved the mouse to ({x}, {y})."), False, info
            if self.guarded and TERMINAL_CLASS_RE.search(target.get("class", "")):
                raise Refused("interacting with a terminal window is not allowed")
            if name == "click":
                button = args.get("button", "left")
                count = args.get("count", 1)
                if button not in {"left", "right", "middle"}:
                    raise BadArgs("button must be left, right or middle")
                if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 3:
                    raise BadArgs("count must be 1, 2 or 3")
                code = {"left": "1", "middle": "2", "right": "3"}[button]
                d.xdotool("click", "--repeat", str(count), "--delay", "90", code)
                self._settle()
                label = {1: "Clicked", 2: "Double-clicked", 3: "Triple-clicked"}[count]
                return self._shot_content(f"{label} {button} at ({x}, {y})."), False, info
            direction = args.get("direction")
            amount = args.get("amount", 3)
            if direction not in {"up", "down", "left", "right"}:
                raise BadArgs("direction must be up, down, left or right")
            if isinstance(amount, bool) or not isinstance(amount, int) or not 1 <= amount <= 20:
                raise BadArgs("amount must be an integer between 1 and 20")
            code = {"up": "4", "down": "5", "left": "6", "right": "7"}[direction]
            d.xdotool("click", "--repeat", str(amount), "--delay", "40", code)
            self._settle()
            return self._shot_content(f"Scrolled {direction} {amount} at ({x}, {y})."), False, info
        if name == "drag":
            x1, y1 = self._coord(args, "x1", SCREEN_W), self._coord(args, "y1", SCREEN_H)
            x2, y2 = self._coord(args, "x2", SCREEN_W), self._coord(args, "y2", SCREEN_H)
            d.xdotool("mousemove", "--sync", str(x1), str(y1))
            target = d.window_under_pointer()
            info["target"] = target
            if self.guarded and TERMINAL_CLASS_RE.search(target.get("class", "")):
                raise Refused("interacting with a terminal window is not allowed")
            d.xdotool("mousedown", "1")
            try:
                steps = 12
                for i in range(1, steps + 1):
                    xi = round(x1 + (x2 - x1) * i / steps)
                    yi = round(y1 + (y2 - y1) * i / steps)
                    d.xdotool("mousemove", str(xi), str(yi))
                    time.sleep(0.02)
            finally:
                d.xdotool("mouseup", "1")
            self._settle()
            return self._shot_content(f"Dragged from ({x1}, {y1}) to ({x2}, {y2})."), False, info
        # keyboard tools: focus guard first
        focus = d.active_window()
        info["focus"] = focus
        reason = focus_refusal(focus.get("class", ""), focus.get("title", "")) if self.guarded else None
        if reason:
            raise Refused(reason)
        if name == "type_text":
            text = args.get("text")
            if not isinstance(text, str) or not text:
                raise BadArgs("text must be a non-empty string")
            if len(text) > 4000:
                raise BadArgs("text is limited to 4000 characters per call")
            d.xdotool("type", "--delay", "12", "--", text, timeout=max(60, len(text) * 0.05 + 30))
            self._settle()
            return self._shot_content(f"Typed {len(text)} characters."), False, info
        if name == "key":
            combos = parse_keys(args.get("keys"))
            for mods, key, _x in combos:
                why = combo_refusal(mods, key, focus.get("class", "")) if self.guarded else None
                if why:
                    raise Refused(why)
            d.xdotool("key", "--delay", "60", *[c[2] for c in combos])
            self._settle()
            return self._shot_content(f"Pressed {' '.join(c[2] for c in combos)}."), False, info
        raise BadArgs(f"unknown tool {name!r}")

    # -- MCP dispatch
    def handle(self, msg: dict) -> dict | None:
        method, msg_id = msg.get("method"), msg.get("id")
        if msg_id is None:
            return None
        if method == "initialize":
            version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
            return _result(msg_id, {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": f"rv-desktop-{self.profile}", "version": "1.0.0"},
            })
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": [{"name": n, **TOOLS[n]} for n in self.tools]})
        if method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            started = time.time()
            record: dict[str, Any] = {"tool": name, "args": _loggable(args)}
            try:
                if not isinstance(args, dict):
                    raise BadArgs("arguments must be an object")
                with self.desk.lock:
                    content, is_error, info = self.call(name, args)
                record.update(info, status="ok")
                result = {"content": content, "isError": is_error}
            except Refused as exc:
                record.update(status="refused", reason=str(exc))
                result = {"content": [{"type": "text", "text": f"Refused: {exc}."}], "isError": True}
            except BadArgs as exc:
                record.update(status="bad_args", reason=str(exc))
                result = {"content": [{"type": "text", "text": f"Invalid arguments: {exc}."}], "isError": True}
            except Exception as exc:  # noqa: BLE001 - surfaced as a tool error
                record.update(status="error", reason=str(exc)[:300])
                result = {"content": [{"type": "text", "text": f"Action failed: {str(exc)[:300]}"}], "isError": True}
            record["elapsed_s"] = round(time.time() - started, 3)
            self.log(record)
            return _result(msg_id, result)
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def _result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _loggable(args: Any) -> Any:
    if isinstance(args, dict):
        return {k: (v if not isinstance(v, str) or len(v) <= 4000 else v[:4000] + "...") for k, v in args.items()}
    return args


def serve(broker: Broker, path: str) -> None:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:  # one MCP session per connection
            broker.log({"event": "connect"})
            for line in self.rfile:
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                reply = broker.handle(msg) if isinstance(msg, dict) else None
                if reply is not None:
                    self.wfile.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
            broker.log({"event": "disconnect"})

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    if os.path.exists(path):
        os.unlink(path)
    old = os.umask(0o077)
    try:
        server = Server(path, Handler)
    finally:
        os.umask(old)
    broker.log({"event": "listening", "profile": broker.profile, "tools": broker.tools, "pid": os.getpid()})
    server.serve_forever()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_TOOLS), required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--settle", type=float, default=0.6, help="seconds to wait after an action")
    parser.add_argument("--selftest", action="store_true", help="take one screenshot and exit")
    args = parser.parse_args(argv)
    broker = Broker(args.profile, args.log, args.settle)
    if args.selftest:
        png, pos = broker.desk.screenshot_png()
        print(json.dumps({"png_bytes": len(png), "cursor": pos, "active": broker.desk.active_window()}))
        return
    serve(broker, args.socket)


if __name__ == "__main__":
    main()
