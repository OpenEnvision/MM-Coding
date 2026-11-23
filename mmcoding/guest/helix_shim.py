"""OS-Helix chat-completions shim. Keeps the latest ten images and at most one action.
No ambient proxies/redirects. Tool-call SSE chunks are buffered until completion;
if a provider ignores parallel_tool_calls=false only its first tool call is delivered.
Other content streams immediately. Original action/stdout messages are preserved.

max_tokens: pi-ai clamps max_tokens to contextWindow - estimate(full DSH transcript, every
screenshot) - 4096 (floor 1), but this shim sends only the latest ten screenshots, so long
episodes would be cut to very few output tokens far below the real context size. For agent
requests (with tools) max_tokens is therefore recomputed from the request actually sent: never
lowered, capped at the configured model maxTokens (65536), bounded by window - estimate(trimmed
request) - safety. The estimate uses the provider-reported prompt_tokens of the previous agent
request (+completion) plus a conservative static estimate of the messages added since; otherwise
a conservative static estimate of the whole trimmed request.
"""
import base64
import hashlib
import os
import json
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None

def rewrite(body, initial=None):
    messages = body['messages']
    if initial:
        # Reconstruct ctx=[task,screenshot0] on every request. s0 is oldest and is
        # itself replaced once ten later screenshots have accumulated.
        user = next(m for m in messages if m.get('role') == 'user')
        content = user.get('content', '')
        if isinstance(content, str): content = [{'type': 'text', 'text': content}]
        user['content'] = [*content, {'type': 'text', 'text': 'Initial screenshot (1920x1080 screen pixels):'}, initial]
    images = [(m, j) for m in messages if isinstance(m.get('content'), list) for j, p in enumerate(m['content']) if isinstance(p, dict) and p.get('type') in ('image_url', 'image', 'input_image')]
    for m, j in images[:-10]:
        m['content'][j] = {'type': 'text', 'text': '[earlier screenshot omitted]'}
    body['parallel_tool_calls'] = False
    return len(images), min(10, len(images))

WINDOW = int(os.environ.get('HELIX_CONTEXT_WINDOW', '262144'))      # contextWindow of the policy models (configs/models.yaml)
CAP = int(os.environ.get('HELIX_MAX_TOKENS', '65536'))              # configured maxTokens of the policy models
SAFETY = 8192
IMAGE_TOKENS = 3500   # conservative (1920x1080 screenshot)


def est_tokens(obj):
    """Conservative static token estimate: images 3500, non-ASCII 1/char, ASCII 1 per 3 chars (no base64 counted)."""
    n = 0.0
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x.get('type') in ('image_url', 'image', 'input_image'):
                n += IMAGE_TOKENS
                continue
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
        elif isinstance(x, str):
            a = len(x.encode('ascii', 'ignore'))
            n += a / 3 + (len(x) - a)
    return int(n) + 1


class HelixShim:
    def __init__(self, base, key, attempt, initial_capture):
        self.base, self.key = base.rstrip('/'), key
        self.log = Path(attempt) / 'helix_requests.jsonl'
        self.initial_capture = initial_capture
        self.initial = None
        self.lock = threading.Lock()
        self.state = None   # {'sig','n','prompt','completion'} of the last completed agent request
        owner = self
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args):
                pass
            def do_POST(self):
                if self.path not in ('/v1/chat/completions', '/chat/completions'):
                    self.send_error(404); return
                rec = {}
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    with owner.lock:
                        if owner.initial is None:
                            data, _ = owner.initial_capture()
                            owner.initial = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}
                        # Initial image is reinserted on requests while no observations exist, so the initial observation
                        # persists in subsequent contexts without changing SDK's event transcript.
                        ins, kept = rewrite(body, owner.initial)
                        sig = n_msgs = None
                        if body.get('tools'):
                            msgs = body['messages']
                            sig = hashlib.sha256(json.dumps(msgs[0], sort_keys=True).encode()).hexdigest()
                            n_msgs = len(msgs)
                            field = 'max_completion_tokens' if 'max_completion_tokens' in body and 'max_tokens' not in body else 'max_tokens'
                            pm = body.get(field)
                            st = owner.state
                            if st and st['sig'] == sig and n_msgs > st['n']:
                                est = st['prompt'] + st['completion'] + est_tokens(msgs[st['n']:]); src = 'usage'
                            else:
                                est = est_tokens(msgs) + est_tokens(body.get('tools')) + est_tokens(body.get('system')); src = 'static'
                            if isinstance(pm, int):
                                new = max(pm, min(CAP, WINDOW - est - SAFETY))
                                body[field] = new
                                rec.update(max_tokens_in=pm, max_tokens_out=new, est_tokens=est, est_source=src)
                    rec.update(n_images_in=ins, n_images_kept=kept, parallel_tool_calls=False, tools=[t.get('function', {}).get('name') for t in body.get('tools', [])])
                    usage = None
                    req = urllib.request.Request(owner.base + '/chat/completions', data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + owner.key, 'Content-Type': 'application/json'})
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
                    with opener.open(req, timeout=600) as up:
                        self.send_response(up.status)
                        self.send_header('Content-Type', up.headers.get('Content-Type', 'application/json'))
                        self.send_header('Connection', 'close'); self.end_headers()
                        if body.get('stream'):
                            calls, held, first = set(), [], None
                            for line in up:
                                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                                    try:
                                        obj = json.loads(line[6:])
                                        if isinstance(obj.get('usage'), dict) and obj['usage'].get('prompt_tokens'):
                                            usage = obj['usage']
                                        for ch in obj.get('choices', []):
                                            tc = ch.get('delta', {}).get('tool_calls', [])
                                            for c in tc:
                                                idx = c.get('index', 0); calls.add(idx)
                                                if first is None: first = idx
                                            if tc:
                                                ch['delta']['tool_calls'] = [c for c in tc if c.get('index', 0) == first]
                                        if any(ch.get('delta', {}).get('tool_calls') for ch in obj.get('choices', [])):
                                            held.append(b'data: ' + json.dumps(obj).encode() + b'\n\n'); continue
                                        if any(ch.get('finish_reason') for ch in obj.get('choices', [])):
                                            for chunk in held: self.wfile.write(chunk)
                                            held = []
                                        # Omit chunks that contain ONLY the filtered second call.
                                        if any(ch.get('delta', {}).get('tool_calls') == [] and not ch.get('finish_reason') for ch in obj.get('choices', [])):
                                            continue
                                    except (ValueError, TypeError):
                                        pass
                                self.wfile.write(line); self.wfile.flush()
                            for chunk in held: self.wfile.write(chunk)
                            rec.update(n_tool_calls_provider=len(calls), n_tool_calls_allowed=min(1, len(calls)))
                        else:
                            obj = json.load(up); counts = []
                            if isinstance(obj.get('usage'), dict) and obj['usage'].get('prompt_tokens'):
                                usage = obj['usage']
                            for ch in obj.get('choices', []):
                                m = ch.get('message', {}); tc = m.get('tool_calls', []); counts.append(len(tc))
                                if tc: m['tool_calls'] = tc[:1]
                            self.wfile.write(json.dumps(obj).encode())
                            rec.update(n_tool_calls_provider=max(counts or [0]), n_tool_calls_allowed=min(1, max(counts or [0])))
                    if sig is not None:
                        with owner.lock:
                            owner.state = ({'sig': sig, 'n': n_msgs, 'prompt': int(usage['prompt_tokens']),
                                            'completion': int(usage.get('completion_tokens') or 0)} if usage else None)
                        if usage:
                            rec.update(prompt_tokens=int(usage['prompt_tokens']))
                except urllib.error.HTTPError as e:
                    self.send_response(e.code); self.send_header('Connection', 'close'); self.end_headers(); self.wfile.write(e.read())
                    rec['http_error'] = e.code
                except Exception as e:
                    rec['error_type'] = type(e).__name__
                    # Do not expose exception strings (which may include upstream URLs/credentials).
                    try: self.send_error(502, 'Helix upstream failure')
                    except Exception: pass
                finally:
                    self.close_connection = True
                    with owner.lock:
                        with owner.log.open('a') as f: f.write(json.dumps(rec) + '\n')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = 'http://127.0.0.1:%d/v1' % self.server.server_address[1]
