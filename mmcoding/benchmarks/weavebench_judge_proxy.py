#!/usr/bin/env python3
"""Local retry proxy between the WeaveBench judge (OpenClaw) and the judge model's endpoint.

OpenClaw treats an HTTP 429 quota/rate-limit answer as a billing failure, disables the key for
several minutes and the whole judge session fails, discarding a long judgement. This proxy retries
429s, 5xx answers and connection errors with backoff *before* any response bytes are sent, then
streams the upstream response through unchanged. Requests are never altered.

Some endpoints report transient failures (rate limits, a spurious ``billing_not_active``) as an
error event inside an HTTP 200 event stream. An event stream is therefore held back until its first
event that is not a preamble (``response.created`` / ``response.in_progress`` / keepalives); a
retryable error there is retried like a 429. Once a content event arrives everything is streamed.

usage: scripts/judge_proxy.sh   (python -m mmcoding.benchmarks.weavebench_judge_proxy --upstream URL --port N)
The judge profile's provider baseUrl is http://127.0.0.1:<port>/v1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time

from aiohttp import ClientConnectionError, ClientPayloadError, ClientSession, ClientTimeout, web

HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}
PREAMBLE = {"response.created", "response.in_progress", "response.queued"}
RETRYABLE = re.compile(r"rate.?limit|\b429\b|quota|billing_not_active|account is not active|overloaded|"
                       r"server_error|server error|temporarily|try again", re.I)
TRANSPORT_ERRORS = (OSError, asyncio.TimeoutError, ClientConnectionError, ClientPayloadError)
HEAD_LIMIT = 1 << 20  # never hold back more than this many bytes
log = logging.getLogger("wb_judge_proxy")


def retry_delay(attempt: int) -> float:
    return min(120.0, 10.0 * 2**attempt) * (0.75 + random.random() / 2)


def event_error(event: bytes) -> tuple[bool, str | None]:
    """(is_preamble, error text) of one SSE event (``data:`` lines joined)."""
    data = b"\n".join(line[5:].strip() for line in event.splitlines() if line.startswith(b"data:"))
    if not data.strip():  # keepalive comments / empty events must not commit the response
        return True, None
    if data == b"[DONE]":
        return False, None
    try:
        obj = json.loads(data)
    except ValueError:
        return False, None
    if not isinstance(obj, dict):
        return False, None
    kind = obj.get("type")
    if kind in PREAMBLE:
        return True, None
    if kind in ("error", "response.failed") or (kind is None and obj.get("error")):
        error = obj.get("error") or (obj.get("response") or {}).get("error") or obj
        text = error if isinstance(error, str) else json.dumps(error)
        return False, text[:500]
    return False, None


async def stream_head(upstream) -> tuple[bytes, str | None]:
    """Bytes up to and including the first non-preamble SSE event, and that event's error text."""
    head = b""
    async for chunk in upstream.content.iter_any():
        head += chunk
        events = head.replace(b"\r\n", b"\n").split(b"\n\n")
        for event in events[:-1]:  # complete events
            preamble, error = event_error(event)
            if not preamble:
                return head, error
        if len(head) > HEAD_LIMIT:
            break
    return head, None


async def handle(request: web.Request) -> web.StreamResponse:
    app = request.app
    body = await request.read()
    url = app["upstream"] + "/" + request.match_info["tail"]
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    deadline = time.monotonic() + app["max_wait"]
    attempt = 0
    async with app["slots"]:
        while True:
            try:
                upstream = await app["session"].request(request.method, url, data=body, headers=headers)
            except TRANSPORT_ERRORS as exc:
                upstream, error = None, f"{type(exc).__name__}"
            else:
                error = None
            head = b""
            stream = False
            retryable = upstream is None or upstream.status == 429 or upstream.status >= 500
            if (upstream is not None and upstream.status == 200
                    and "text/event-stream" in upstream.headers.get("Content-Type", "")):
                stream = True
                try:
                    head, stream_error = await stream_head(upstream)
                except TRANSPORT_ERRORS as exc:
                    # Transport failures are retryable regardless of their exception text.
                    retryable, error = True, f"{type(exc).__name__} while reading the stream head"
                    app["stats"]["stream_errors"] += 1
                else:
                    if stream_error and RETRYABLE.search(stream_error):
                        retryable, error = True, f"stream error {stream_error[:200]}"
                        app["stats"]["stream_errors"] += 1
            wait = retry_delay(attempt) if retryable else 0
            if retryable and time.monotonic() + wait < deadline:
                status = error or (upstream.status if upstream is not None else "?")
                if upstream is not None:
                    upstream.close()  # do not drain a failed/possibly disconnected response
                    upstream.release()
                app["stats"]["retries"] += 1
                log.warning("upstream %s on %s; retry %d in %.0fs", status, request.path, attempt + 1, wait)
                await asyncio.sleep(wait)
                attempt += 1
                continue
            if upstream is None:
                app["stats"]["gave_up"] += 1
                return web.json_response({"error": {"message": f"proxy: upstream unreachable ({error})"}}, status=502)
            if retryable:
                app["stats"]["gave_up"] += 1
                if stream:
                    upstream.close()
                    upstream.release()
                    return web.json_response({"error": {"message": f"proxy: upstream retries exhausted ({error})"}},
                                             status=502)
            app["stats"]["requests"] += 1
            response = web.StreamResponse(status=upstream.status, headers={
                k: v for k, v in upstream.headers.items() if k.lower() not in HOP})
            await response.prepare(request)
            if head:
                await response.write(head)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            upstream.release()
            await response.write_eof()
            return response


async def stats(request: web.Request) -> web.Response:
    return web.json_response(request.app["stats"])


async def make_app(args) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["upstream"] = args.upstream.rstrip("/")
    app["max_wait"] = args.max_wait
    app["slots"] = asyncio.Semaphore(args.concurrency)
    app["stats"] = {"requests": 0, "retries": 0, "stream_errors": 0, "gave_up": 0, "started": time.time()}

    async def start(app):
        app["session"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30, sock_read=900),
                                       auto_decompress=False)

    async def stop(app):
        await app["session"].close()

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_get("/_stats", stats)
    app.router.add_route("*", "/v1/{tail:.*}", handle)
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", required=True, help="base URL of the judge model's endpoint")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--max_wait", type=float, default=900, help="give up retrying after this many seconds")
    p.add_argument("--concurrency", type=int, default=6, help="max in-flight upstream requests")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    web.run_app(make_app(args), host="127.0.0.1", port=args.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
