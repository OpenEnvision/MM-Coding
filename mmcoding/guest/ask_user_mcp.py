#!/usr/bin/env python3
"""Minimal MCP stdio server exposing one tool, ``ask_user``.

It forwards the question through the official OSWorld-V2 DSH bridge
(``hybrid_agents/deepseek/helpers.py: ask``), i.e. the same question/answer
files the unmodified host-side ``DeepSeekAgent`` already serves with the task's
user simulator. Mounted in every harness, only for tasks that define a simulator,
so asking the user never depends on shell access.

Environment: OSWORLD_DSH_ATTEMPT (attempt directory), OSWORLD_DSH_CODE
(directory holding the official helpers.py). Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("OSWORLD_DSH_CODE", "/home/user/.local/share/osworld-dsh/code"))
from helpers import ask  # noqa: E402  (official bridge, uploaded unchanged)

PROTOCOL = "2025-06-18"
TOOL = {
    "name": "ask_user",
    "description": (
        "Ask the user a question when information required by the task is missing. "
        "Blocks until the user answers and returns the answer text."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question for the user."}},
        "required": ["question"],
        "additionalProperties": False,
    },
}


def reply(msg_id, result=None, error=None) -> None:
    payload = {"jsonrpc": "2.0", "id": msg_id}
    payload["error" if error else "result"] = error or result
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def handle(msg: dict) -> None:
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:
        return  # notifications (e.g. notifications/initialized)
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
        reply(msg_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "osworld-ask-user", "version": "1.0.0"}})
    elif method == "ping":
        reply(msg_id, {})
    elif method == "tools/list":
        reply(msg_id, {"tools": [TOOL]})
    elif method == "tools/call":
        params = msg.get("params") or {}
        question = (params.get("arguments") or {}).get("question", "")
        if params.get("name") != "ask_user":
            reply(msg_id, {"content": [{"type": "text", "text": "Unknown tool."}], "isError": True})
            return
        try:
            answer = ask(Path(os.environ["OSWORLD_DSH_ATTEMPT"]), str(question))
            reply(msg_id, {"content": [{"type": "text", "text": answer}]})
        except Exception as exc:  # surfaced to the model as a tool error
            reply(msg_id, {"content": [{"type": "text", "text": f"ask_user failed: {exc}"}], "isError": True})
    else:
        reply(msg_id, error={"code": -32601, "message": f"Method not found: {method}"})


def main() -> None:
    for line in sys.stdin:
        if line.strip():
            try:
                handle(json.loads(line))
            except json.JSONDecodeError:
                continue


if __name__ == "__main__":
    main()
