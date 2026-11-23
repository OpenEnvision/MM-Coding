#!/usr/bin/env python3
"""rv-serve: keep a service running after the agent session ends.

Usage: rv-serve --name NAME [--port PORT] [--cwd DIR] -- COMMAND [ARGS...]

The command is started as a separate persistent process in the same sandbox and
keeps running after the agent session ends. With --port it becomes reachable on
the desktop at http://localhost:PORT and must listen on 0.0.0.0:PORT (not only
127.0.0.1). Re-running with the same NAME replaces the previous one.
"""

import argparse
import json
import os
import socket
import sys


def main() -> None:
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing COMMAND after --")
    request = {"op": "serve", "name": args.name, "port": args.port, "cwd": os.path.abspath(args.cwd),
               "argv": command}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(90)
        s.connect("/run/rvd/ctl.sock")
        s.sendall((json.dumps(request) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    reply = json.loads(data or b"{}")
    print(json.dumps(reply, indent=1))
    sys.exit(0 if reply.get("ok") else 1)


if __name__ == "__main__":
    main()
