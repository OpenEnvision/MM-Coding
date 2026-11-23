#!/usr/bin/env python3
"""MCP stdio server of the Vision-Tool harness (vision_tool): one tool, ``ask_vision``.

The agent model never receives pixels. It sends a question plus image sources
("screen" for the current 1920x1080 screen, or absolute paths of image files);
this server forwards them to a separate, stateless call of a vision model (the
answerer) and returns the answer as text only. The answerer sees no task
instruction and no history: whatever context it needs must be in the question.

Prompt version v2: the answerer only reports what is visible (no instructions,
advice or general knowledge; out-of-scope questions are declined), never trusts a
premise of the question, and marks locations as <pt>x,y</pt> / <box>x1,y1,x2,y2</box>
in its native coordinate convention (per answerer profile: pixels of the image as
shown, or a 0-1000 scale). The server converts every marked location to pixels of
the original image before the agent sees the answer. Image labels carry no paths.

Image admission and normalization mirror the DSH 0.1.5rc1 attachment store used
by ``read_image`` in the screenshot harnesses (sources up to 8192 px per side, 64 MP and
20 MiB; copies downscaled proportionally to at most 2048x2048 pixels), so the
answerer sees the resolution the policy model of those harnesses would have seen.
Encodings above 1 MiB are re-encoded as JPEG (quality 90 down to 70) because
multi-MiB uploads can time out on slow links; at most 6 MiB per call.

Every call is appended to ``$OSWORLD_DSH_ATTEMPT/vision_calls.jsonl`` and the
images it was shown are saved under ``$OSWORLD_DSH_ATTEMPT/vision/`` (collected
with the attempt). Runs inside the agent sandbox with X access (DISPLAY=:0).
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

PROTOCOL = "2025-06-18"
HERE = Path(__file__).resolve().parent
SCREEN = "screen"
MAX_IMAGES = 8
MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_DIMENSION = 8192
MAX_SOURCE_PIXELS = 64_000_000
NORMALIZED_MAX_PIXELS = 2048 * 2048
UPLOAD_TARGET_BYTES = 1024 * 1024  # larger encodings are re-encoded as JPEG (see normalize)
JPEG_QUALITIES = (90, 85, 80, 70)
MAX_CALL_BYTES = 6 * 1024 * 1024  # total encoded image bytes per call
MAX_QUESTION_CHARS = 8000
MAX_ANSWER_CHARS = 20000
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
DEADLINE_S = 1200
HTTP_TIMEOUT_S = 600
RETRY_WAITS = (5, 20, 60)

PROMPT_VERSION = "v2"
OUT_OF_SCOPE = "OUT OF SCOPE: I only report what is visible."
# Answerer profiles: coordinate convention and thinking switch per model family (Qwen models
# answer in a 0-1000 space and localize best with thinking off; Kimi models answer in pixels).
# Overridable with RV_VISION_COORDS / RV_VISION_THINKING.
PROFILES = (
    ("kimi", {"coords": "pixel", "thinking": None}),
    ("qwen", {"coords": "norm1000", "thinking": "off"}),
)
COORD_RULES = {
    "pixel": "in pixels of the image as you see it (0,0 is the top-left corner)",
    "norm1000": ("as integers on a 0-1000 scale relative to the image width and height (0,0 is the "
                 "top-left corner, 1000,1000 the bottom-right corner)"),
}

TOOL = {
    "name": "ask_vision",
    "description": (
        "Ask a vision assistant about the current screen or about image files. It looks at the images you "
        "pass and answers your question in text. It sees only these images and your question: it does not "
        "know the task and remembers nothing between calls, so include any context it needs. It only reports "
        "what is visible: it gives no instructions or advice and does not answer from general knowledge. "
        "Image sources: \"screen\" (the current 1920x1080 screen, mouse cursor drawn) or absolute paths of "
        "PNG/JPEG/WebP/GIF files. Locations in its answers are pixel coordinates of the original image "
        "(screen pixels for the screen)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What you want to know about the images."},
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Image sources, in order: \"screen\" or absolute file paths (default [\"screen\"]).",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

SYSTEM_TEMPLATE = (
    "You are the eyes of a computer-use agent that cannot see. The agent sends you one or more images "
    "(screenshots of its Ubuntu desktop, or image files) and a question. You only report what is visible in "
    "these images: describe, transcribe text exactly, locate, count, compare or verify. Be accurate and "
    "concrete, and answer only what is asked.\n"
    "Rules:\n"
    "- Base every statement on the images. If something is not visible, unreadable or uncertain, say so. "
    "Never assume that a premise of the question is true; if it is false, say what is actually visible.\n"
    "- Do not give instructions, advice, plans or next steps, and do not answer from general knowledge. If "
    "the question asks what to do, how to do something, or for facts that are not visible in the images, "
    "start your reply with \"" + OUT_OF_SCOPE + "\" and then report only the relevant visible content. "
    "Questions about what an image shows are never out of scope, even when their premise is wrong.\n"
    "- Mark every location as <pt>x,y</pt> (a point) or <box>x1,y1,x2,y2</box> (a region), {coords}. "
    "When several images are attached, add the image number, for example <pt i=2>x,y</pt>."
)

def tool_spec() -> dict:
    return TOOL


def answerer_profile(model: str | None = None) -> dict:
    model = (model if model is not None else os.environ.get("RV_VISION_MODEL", "")).lower()
    profile = {"coords": "pixel", "thinking": None}
    for prefix, values in PROFILES:
        if model.startswith(prefix):
            profile = dict(values)
            break
    coords = os.environ.get("RV_VISION_COORDS", "")
    if coords in COORD_RULES:
        profile["coords"] = coords
    thinking = os.environ.get("RV_VISION_THINKING", "")
    if thinking in ("on", "off"):
        profile["thinking"] = thinking
    profile["model"] = model
    return profile


def system_prompt(profile: dict) -> str:
    return SYSTEM_TEMPLATE.format(coords=COORD_RULES[profile["coords"]])


class ToolError(Exception):
    """Reported to the agent as a tool error."""


class Cancelled(ToolError):
    """The MCP caller cancelled this question."""


class Operation:
    """One question's deadline, cancellation and partial accounting (including retries)."""

    def __init__(self, timeout: float = DEADLINE_S):
        self.deadline = time.monotonic() + min(timeout, DEADLINE_S)
        self.cancelled = threading.Event()
        self.completed = False
        self.lock = threading.RLock()
        self.response = None
        self.connection = None
        self.requests: list[dict] = []
        self.rejected_payload_bytes = None

    def interrupt(self) -> None:
        # Closing a buffered response from another thread can block on its read lock.
        # Track the connection *before* requesting headers, then shutdown its socket.
        # DNS/connect before a socket exists remains subject to the platform resolver
        # and socket timeout; once connected, headers and SSE are both interruptible.
        with self.lock:
            response, connection = self.response, self.connection
        sock = getattr(connection, "sock", None)
        if sock is None:
            sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def cancel(self) -> None:
        with self.lock:
            if self.completed:
                return
            self.cancelled.set()
        self.interrupt()

    def remaining(self) -> float:
        if self.cancelled.is_set():
            raise Cancelled("the vision question was cancelled")
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise ToolError("the vision question deadline was exceeded")
        return left

    def accounting(self) -> dict:
        total: dict = {"answerer_calls": len(self.requests)}
        for request in self.requests:
            for key, value in (request.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    total[key] = total.get(key, 0) + value
        # payload_bytes is the largest actual request, not just the initial images.
        # Per-request records retain usage details and unknown usage remains None.
        return {"requests": self.requests, "usage": total,
                "payload_bytes": max((r["payload_bytes"] for r in self.requests), default=0),
                "total_payload_bytes": sum(r["payload_bytes"] for r in self.requests),
                "rejected_payload_bytes": self.rejected_payload_bytes}


_operation_local = threading.local()
_active: dict[object, Operation] = {}
_active_lock = threading.Lock()


@contextmanager
def operation_scope(operation: Operation | None = None, timeout: float | None = None):
    previous = getattr(_operation_local, "current", None)
    current = operation or previous or Operation(DEADLINE_S if timeout is None else timeout)
    if timeout is not None:
        current.deadline = min(current.deadline, time.monotonic() + timeout)
    _operation_local.current = current
    try:
        yield current
    finally:
        _operation_local.current = previous


_out_lock = threading.Lock()
_log_lock = threading.Lock()
_counter = {"n": 0}


def reply(msg_id, result=None, error=None) -> None:
    payload = {"jsonrpc": "2.0", "id": msg_id}
    payload["error" if error else "result"] = error or result
    with _out_lock:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def attempt_dir() -> Path:
    return Path(os.environ.get("OSWORLD_DSH_ATTEMPT", "."))


# ------------------------------------------------------------------ images


def capture_screen() -> tuple[bytes, tuple[int, int]]:
    """Same capture as the broker's screenshot tool (root window + cursor)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from rv_desktop import Desktop  # noqa: PLC0415 - needs DISPLAY; imported lazily

    return Desktop(0.0).screenshot_png()


def normalize(data: bytes) -> tuple[bytes, str, tuple[int, int], tuple[int, int]]:
    """Admit one source image and return (bytes, mime, original size, shown size).

    Resolution follows the DSH attachment store; the encoding does not: an image
    whose encoding exceeds UPLOAD_TARGET_BYTES is re-encoded as JPEG (same pixels),
    because multi-MiB uploads to the model endpoint can take minutes on a slow link
    and time out.
    """
    from PIL import Image  # noqa: PLC0415

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"not a readable image ({exc})") from exc
    w, h = img.size
    if max(w, h) > MAX_SOURCE_DIMENSION:
        raise ToolError(f"at least one image side exceeds the {MAX_SOURCE_DIMENSION}px limit; downscale the "
                        "image and pass the smaller copy")
    if w * h > MAX_SOURCE_PIXELS:
        raise ToolError(f"the image exceeds the {MAX_SOURCE_PIXELS}-pixel limit; downscale the image and pass "
                        "the smaller copy")
    if getattr(img, "is_animated", False):
        img.seek(0)
    scale = min(1.0, (NORMALIZED_MAX_PIXELS / float(w * h)) ** 0.5)
    shown = (max(1, int(w * scale)), max(1, int(h * scale)))
    fmt = (img.format or "").upper()
    if shown == (w, h) and fmt in {"PNG", "JPEG"} and len(data) <= UPLOAD_TARGET_BYTES:
        return data, "image/png" if fmt == "PNG" else "image/jpeg", (w, h), shown
    if shown != (w, h):
        img = img.resize(shown, Image.LANCZOS)
    has_alpha = img.mode in {"RGBA", "LA"} or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    out = io.BytesIO()
    img.save(out, format="PNG", compress_level=6)
    if out.tell() <= UPLOAD_TARGET_BYTES:
        return out.getvalue(), "image/png", (w, h), shown
    rgb = img.convert("RGB") if has_alpha else img
    for quality in JPEG_QUALITIES:
        out = io.BytesIO()
        rgb.save(out, format="JPEG", quality=quality)
        if out.tell() <= UPLOAD_TARGET_BYTES:
            break
    return out.getvalue(), "image/jpeg", (w, h), shown


def load_source(source: str) -> dict:
    """Read one image source; returns a dict with the image and its label."""
    if source == SCREEN:
        try:
            png, cursor = capture_screen()
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"screen capture failed: {exc}") from exc
        data, mime, size, shown = normalize(png)
        label = f"the current screen ({shown[0]}x{shown[1]}, the mouse cursor is drawn)"
        return {"source": SCREEN, "data": data, "mime": mime, "size": size, "shown": shown, "label": label,
                "note": f"screen {size[0]}x{size[1]}; cursor at ({cursor[0]}, {cursor[1]})"}
    path = Path(source)
    if not path.is_absolute():
        raise ToolError(f"{source!r}: use \"screen\" or an absolute file path")
    if path.suffix.lower() not in EXTENSIONS:
        raise ToolError(f"{source}: only PNG, JPEG, WebP and GIF files are supported")
    try:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ToolError(f"{source}: the file exceeds the 20 MiB limit; downscale it and pass the smaller copy")
        raw = path.read_bytes()
    except ToolError:
        raise
    except OSError as exc:
        raise ToolError(f"cannot read {source}: {exc.strerror or exc}") from exc
    try:
        data, mime, size, shown = normalize(raw)
    except ToolError as exc:
        raise ToolError(f"{source}: {exc}") from exc
    label = f"an image ({shown[0]}x{shown[1]})"
    note = f"{source} {size[0]}x{size[1]}"
    if shown != size:
        note += f" (shown downscaled to {shown[0]}x{shown[1]})"
    return {"source": source, "data": data, "mime": mime, "size": size, "shown": shown, "label": label,
            "note": note}


# -------------------------------------------------------------------- model


def build_messages(question: str, images: list[dict], profile: dict | None = None) -> list[dict]:
    profile = profile or answerer_profile()
    content: list[dict] = []
    for i, image in enumerate(images, 1):
        content.append({"type": "text", "text": f"Image {i}: {image['label']}."})
        url = f"data:{image['mime']};base64," + base64.b64encode(image["data"]).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": "Question: " + question})
    return [{"role": "system", "content": system_prompt(profile)}, {"role": "user", "content": content}]


# Tolerant of the variants answerers produce: <pt>x,y</pt>, <pt i=2>x, y</pt>, <pt>(x, y)</pt>,
# <pt x=497,y=558</pt>, <pt x="497" y="558"></pt>.
_TAG = re.compile(r"<(pt|box)\b([^<]{0,80}?)</(?:pt|box)>")
_IDX = re.compile(r"\bi\s*=\s*[\"']?(\d+)")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def to_source_px(value: float, axis: int, image: dict, coords: str) -> int:
    """A coordinate on an image as shown -> pixels of its source image."""
    full, shown = image["size"][axis], image["shown"][axis]
    px = value / 1000.0 * full if coords == "norm1000" else value * full / shown
    return int(round(min(max(px, 0), full - 1)))


def convert_locations(text: str, images: list[dict], coords: str) -> tuple[str, int]:
    """Replace <pt>/<box> tags with pixel coordinates of the original images."""
    count = {"n": 0}
    sources = len(images)

    def to_px(value: float, axis: int, image: dict) -> int:
        return to_source_px(value, axis, image, coords)

    def repl(match: re.Match) -> str:
        kind, inner = match.group(1), match.group(2)
        idx = _IDX.search(inner)
        if idx:
            inner = inner[:idx.start()] + inner[idx.end():]
        nums = [float(x) for x in _NUM.findall(inner)]
        i = int(idx.group(1)) if idx else 1
        if not 1 <= i <= len(images) or len(nums) != (2 if kind == "pt" else 4):
            return match.group(0)
        image = images[i - 1]
        pts = [(to_px(nums[k], 0, image), to_px(nums[k + 1], 1, image)) for k in range(0, len(nums), 2)]
        count["n"] += 1
        where = f" in image {i}" if sources > 1 else ""
        if kind == "pt":
            return f"({pts[0][0]}, {pts[0][1]}){where}"
        return f"({pts[0][0]}, {pts[0][1]})-({pts[1][0]}, {pts[1][1]}){where}"

    return _TAG.sub(repl, text), count["n"]


def request_body(messages: list[dict], profile: dict | None = None) -> dict:
    profile = profile or answerer_profile()
    body = {
        "model": os.environ.get("RV_VISION_MODEL", ""),
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": int(os.environ.get("RV_VISION_MAX_TOKENS", "16384")),
    }
    if profile.get("thinking") == "off":
        body["enable_thinking"] = False  # no reasoning effort either: perception only
    else:
        if profile.get("thinking") == "on":
            body["enable_thinking"] = True
        effort = os.environ.get("RV_VISION_EFFORT", "")
        if effort:
            body["reasoning_effort"] = effort
    return body


def open_request(request, timeout: float, operation: Operation):
    """urllib with a cancellable connection even before response headers arrive."""
    def factory(connection_type):
        class Connection(connection_type):
            def connect(self):
                operation.remaining()
                super().connect()
                try:
                    # DNS/connect may finish after cancellation; never send a late POST.
                    operation.remaining()
                except ToolError:
                    self.close()
                    raise

        def connect(*args, **kwargs):
            connection = Connection(*args, **kwargs)
            with operation.lock:
                operation.connection = connection
            operation.remaining()
            return connection
        return connect

    class HTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(factory(http.client.HTTPConnection), req)

    class HTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(factory(http.client.HTTPSConnection), req, context=self._context)

    return urllib.request.build_opener(HTTPHandler(), HTTPSHandler()).open(request, timeout=timeout)


@contextmanager
def model_response(request):
    operation = getattr(_operation_local, "current", None) or Operation()
    remaining = operation.remaining()
    # A socket inactivity timeout alone does not bound a slowly trickling SSE stream.
    watchdog = threading.Timer(remaining, operation.interrupt)
    watchdog.daemon = True
    watchdog.start()
    try:
        with open_request(request, timeout=min(HTTP_TIMEOUT_S, remaining), operation=operation) as response:
            with operation.lock:
                operation.response = response
            operation.remaining()
            yield response
            operation.remaining()
    finally:
        with operation.lock:
            operation.response = None
            operation.connection = None
        watchdog.cancel()


def call_model(messages: list[dict], profile: dict | None = None) -> dict:
    """One streamed chat completion; returns answer text, usage and finish reason."""
    # RV_VISION_BASE_URL / RV_VISION_API_KEY: optional separate answerer endpoint (worker_rv exports them
    # only when the host configured one); otherwise the agent's own route.
    base = (os.environ.get("RV_VISION_BASE_URL") or os.environ["RV_LLM_BASE_URL"]).rstrip("/")
    key = os.environ.get("RV_VISION_API_KEY") or os.environ["RV_LLM_API_KEY"]
    body = request_body(messages, profile)
    request = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json"},
    )
    content, reasoning, usage, finish = [], [], None, None
    operation = getattr(_operation_local, "current", None)
    with model_response(request) as response:
        for raw in response:
            # Capture reported usage even on the final chunk of a cancelled stream.
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                if operation:
                    operation.remaining()
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
                if operation and operation.requests:
                    operation.requests[-1]["usage"] = usage
            if operation:
                operation.remaining()
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    text = "".join(reasoning)
    return {"answer": "".join(content).strip(), "usage": usage, "finish_reason": finish,
            "reasoning_chars": len(text), "reasoning": text}


def image_payload_bytes(messages: list[dict]) -> int:
    """Encoded image bytes (before base64), counting every image actually resent."""
    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("data:") or ";base64," not in url:
                    raise ToolError("vision images must be inline base64 data")
                encoded = url.split(";base64,", 1)[1]
                total += len(encoded) // 4 * 3 - (len(encoded) - len(encoded.rstrip("=")))
    return total


def check_payload(payload: int, operation: Operation) -> None:
    if payload > MAX_CALL_BYTES:
        operation.rejected_payload_bytes = payload
        raise ToolError(f"the images total {payload / 2**20:.1f} MiB after compression, more than "
                        f"{MAX_CALL_BYTES / 2**20:g} MiB per call; pass fewer or smaller images")


def ask(messages: list[dict], profile: dict | None = None, *, timeout: float | None = None) -> dict:
    """Retry within the remaining question budget."""
    with operation_scope(timeout=timeout) as operation:
        errors = []
        for wait in (0, *RETRY_WAITS):
            remaining = operation.remaining()
            if wait:
                if wait >= remaining:
                    break
                operation.cancelled.wait(wait)  # wake immediately on MCP cancellation
            operation.remaining()
            payload = image_payload_bytes(messages)
            check_payload(payload, operation)
            request = {"payload_bytes": payload, "usage": None, "status": "running"}
            operation.requests.append(request)
            try:
                result = call_model(messages, profile) if profile else call_model(messages)
                request.update(usage=result.get("usage"), finish_reason=result.get("finish_reason"))
                operation.remaining()  # no late answer or retry
                if result["answer"]:
                    request["status"] = "ok"
                    result["errors"] = errors
                    return result
                errors.append(f"empty answer (finish_reason={result['finish_reason']})")
                request.update(status="error", error=errors[-1])
            except ToolError as exc:
                request.update(status="cancelled" if isinstance(exc, Cancelled) else "error", error=str(exc))
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors are retried
                if isinstance(exc, urllib.error.HTTPError):
                    # Do not read an unbounded error body after a failed request.
                    exc.close()
                    errors.append(f"HTTP {exc.code}: {exc.reason}")
                else:
                    errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
                request.update(status="error", error=errors[-1])
                try:
                    operation.remaining()
                except ToolError as stopped:
                    request.update(status="cancelled" if isinstance(stopped, Cancelled) else "error",
                                   error=str(stopped))
                    raise
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
                    break
        raise ToolError("the vision assistant is unavailable right now (" + "; ".join(errors[-2:]) + ")")


# ------------------------------------------------------------------- tool


def record(entry: dict, images: list[dict]) -> None:
    root = attempt_dir()
    with _log_lock:
        _counter["n"] += 1
        n = _counter["n"]
    saved = []
    if images:
        folder = root / "vision"
        folder.mkdir(exist_ok=True)
        for i, image in enumerate(images, 1):
            ext = ".png" if image["mime"] == "image/png" else ".jpg"
            target = folder / f"{n:04d}_{i}{ext}"
            target.write_bytes(image["data"])
            saved.append(str(target.relative_to(root)))
    entry = {"n": n, **entry, "saved": saved}
    with _log_lock, (root / "vision_calls.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_tool(arguments: dict, operation: Operation | None = None) -> str:
    with operation_scope(operation) as current:
        return _run_tool(arguments, current)


def _run_tool(arguments: dict, operation: Operation) -> str:
    started = time.time()
    question = arguments.get("question")
    sources = arguments.get("images", [SCREEN])
    images: list[dict] = []
    entry: dict = {"time": round(started, 3), "question": str(question)[:MAX_QUESTION_CHARS],
                   "images": sources if isinstance(sources, list) else str(sources)}
    try:
        operation.remaining()
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question is required")
        if len(question) > MAX_QUESTION_CHARS:
            raise ToolError(f"question is longer than {MAX_QUESTION_CHARS} characters")
        if not isinstance(sources, list) or not sources or not all(isinstance(s, str) for s in sources):
            raise ToolError("images must be a non-empty list of strings (\"screen\" or absolute file paths)")
        if len(sources) > MAX_IMAGES:
            raise ToolError(f"at most {MAX_IMAGES} images per call")
        for source in sources:
            operation.remaining()
            images.append(load_source(source.strip()))
        check_payload(sum(len(im["data"]) for im in images), operation)
        profile = answerer_profile()
        result = ask(build_messages(question, images, profile), profile)
        raw = result["answer"][:MAX_ANSWER_CHARS]
        answer, converted = convert_locations(raw, images, profile["coords"])
        operation.remaining()
        entry.update(status="ok", answer=answer, raw_answer=raw if converted else None, converted=converted,
                     prompt_version=PROMPT_VERSION, profile=profile, usage=result["usage"],
                     finish_reason=result["finish_reason"], reasoning_chars=result["reasoning_chars"],
                     retries=result["errors"], out_of_scope=raw.lstrip().startswith(OUT_OF_SCOPE),
                     image_info=[{"source": im["source"], "size": im["size"], "shown": im["shown"],
                                  "sha256": hashlib.sha256(im["data"]).hexdigest()} for im in images],
                     latency_s=round(time.time() - started, 1))
        # Linearize completion against cancellation: a cancelled question is never answered.
        with operation.lock:
            operation.remaining()
            operation.completed = True
        entry.update(operation.accounting())
        record(entry, images)
        header = "Images: " + "; ".join(f"{i}: {im['note']}" for i, im in enumerate(images, 1))
        header += ". Locations below are pixel coordinates of the original images."
        return header + "\n\n" + answer
    except Exception as exc:
        entry.update(operation.accounting())
        entry.update(status="cancelled" if isinstance(exc, Cancelled) else "error",
                     error=str(exc), latency_s=round(time.time() - started, 1))
        record(entry, images)
        raise


def handle(msg: dict, operation: Operation | None = None) -> None:
    method, msg_id = msg.get("method"), msg.get("id")
    if method == "notifications/cancelled":
        request_id = (msg.get("params") or {}).get("requestId")
        with _active_lock:
            current = _active.get(request_id)
        if current is not None:
            current.cancel()
        return
    if msg_id is None:
        return
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
        reply(msg_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "rv-vision", "version": "2.0.0"}})
    elif method == "ping":
        reply(msg_id, {})
    elif method == "tools/list":
        reply(msg_id, {"tools": [tool_spec()]})
    elif method == "tools/call":
        params = msg.get("params") or {}
        arguments = params.get("arguments") or {}
        if params.get("name") != "ask_vision" or not isinstance(arguments, dict):
            with _active_lock:
                if _active.get(msg_id) is operation:
                    _active.pop(msg_id, None)
            reply(msg_id, {"content": [{"type": "text", "text": "Unknown tool."}], "isError": True})
            return
        operation = operation or Operation()
        with _active_lock:
            _active[msg_id] = operation
        try:
            text = run_tool(arguments, operation)
            reply(msg_id, {"content": [{"type": "text", "text": text}]})
        except Cancelled:
            pass  # MCP cancellation notifications do not need a response
        except ToolError as exc:
            reply(msg_id, {"content": [{"type": "text", "text": f"ask_vision failed: {exc}"}], "isError": True})
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            reply(msg_id, {"content": [{"type": "text", "text": f"ask_vision failed: {type(exc).__name__}: "
                                        f"{str(exc)[:300]}"}], "isError": True})
        finally:
            with _active_lock:
                if _active.get(msg_id) is operation:
                    del _active[msg_id]
    else:
        reply(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("method") == "tools/call" and msg.get("id") is not None:
            # Register before starting the worker so a following cancellation cannot
            # race ahead of registration. The worker always removes its own entry.
            operation = Operation()
            with _active_lock:
                _active[msg["id"]] = operation
            threading.Thread(target=handle, args=(msg, operation), daemon=True).start()
        else:
            handle(msg)


if __name__ == "__main__":
    main()
