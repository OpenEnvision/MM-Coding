#!/usr/bin/env python3
"""Minimal MCP stdio server exposing one tool, ``report_infeasible``.

Mounted in every harness so each agent has the same honest exit
when it concludes that the task cannot be completed with its capabilities. The
report is appended to ``$OSWORLD_DSH_ATTEMPT/infeasible.jsonl`` (collected with
the attempt) and does not change the task's evaluation. Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROTOCOL = "2025-06-18"
TOOL = {
    "name": "report_infeasible",
    "description": (
        "Report that the task cannot be completed with your available capabilities. Give the concrete "
        "reason. After calling it, stop working and end your turn with a short message to the user."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why the task cannot be completed."},
            "completed_parts": {"type": "string", "description": "What, if anything, was completed."},
        },
        "required": ["reason"],
        "additionalProperties": False,
    },
}


def reply(msg_id, result=None, error=None) -> None:
    payload = {"jsonrpc": "2.0", "id": msg_id}
    payload["error" if error else "result"] = error or result
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def record(arguments: dict) -> None:
    attempt = Path(os.environ.get("OSWORLD_DSH_ATTEMPT", "."))
    entry = {
        "time": round(time.time(), 3),
        "reason": str(arguments.get("reason", ""))[:4000],
        "completed_parts": str(arguments.get("completed_parts", ""))[:4000],
    }
    with (attempt / "infeasible.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def handle(msg: dict) -> None:
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:
        return
    if method == "initialize":
        version = (msg.get("params") or {}).get("protocolVersion") or PROTOCOL
        reply(msg_id, {"protocolVersion": version, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "rv-report", "version": "1.0.0"}})
    elif method == "ping":
        reply(msg_id, {})
    elif method == "tools/list":
        reply(msg_id, {"tools": [TOOL]})
    elif method == "tools/call":
        params = msg.get("params") or {}
        arguments = params.get("arguments") or {}
        if params.get("name") != "report_infeasible" or not isinstance(arguments, dict):
            reply(msg_id, {"content": [{"type": "text", "text": "Unknown tool."}], "isError": True})
            return
        if not str(arguments.get("reason", "")).strip():
            reply(msg_id, {"content": [{"type": "text", "text": "reason is required."}], "isError": True})
            return
        try:
            record(arguments)
            reply(msg_id, {"content": [{"type": "text", "text": "Recorded. Stop working now and end your "
                                        "turn with a short message to the user."}]})
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            reply(msg_id, {"content": [{"type": "text", "text": f"report failed: {exc}"}], "isError": True})
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
