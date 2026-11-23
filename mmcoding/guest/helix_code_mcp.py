#!/usr/bin/env python3
"""One block of bash/python, then a fresh desktop screenshot. Fail closed on confinement failure."""
import base64
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from rv_desktop import Desktop

LOCK = threading.Lock()
TOOL = {'name': 'run_code', 'description': 'Execute exactly one block of bash or Python with DISPLAY=:0; returns exit code, stdout/stderr and a fresh screenshot. Filesystem writes confined to the workspace and temporary files. timeout_s <= 600.',
        'inputSchema': {'type': 'object', 'properties': {'language': {'type': 'string', 'enum': ['bash', 'python']}, 'code': {'type': 'string'}, 'timeout_s': {'type': 'number', 'minimum': 1, 'maximum': 600}}, 'required': ['language', 'code'], 'additionalProperties': False}}

def execute(args):
    if not LOCK.acquire(blocking=False):
        raise ValueError('A code action is already running; take exactly one action per round')
    try:
        lang, code = args['language'], args['code']
        if lang not in ('bash', 'python') or not isinstance(code, str): raise ValueError('Invalid code block')
        timeout = float(args.get('timeout_s', 120))
        if not 1 <= timeout <= 600: raise ValueError('timeout_s must be in [1,600]')
        cwd = str(Path.cwd().resolve())
        command = ['bash', '-c', code] if lang == 'bash' else ['/usr/bin/python3', '-c', code]
        argv = ['bwrap', '--ro-bind', '/', '/', '--dev', '/dev', '--unshare-pid', '--proc', '/proc', '--die-with-parent', '--tmpfs', '/tmp', '--bind', cwd, cwd, '--', *command]
        env = os.environ.copy(); env['DISPLAY'] = ':0'
        # tmpfs must not hide the workspace (OS2 commonly uses /home/user or /tmp_workspace, never /tmp).
        if cwd == '/tmp' or cwd.startswith('/tmp/'): raise RuntimeError('Workspace must not be below private /tmp')
        import tempfile
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            timed = False
            try: rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed = True; os.killpg(p.pid, signal.SIGKILL); rc = p.wait()
            stdout.seek(0); stderr.seek(0)
            so, se = stdout.read(30000).decode(errors='replace'), stderr.read(10000).decode(errors='replace')
        if rc and 'bwrap:' in se: raise RuntimeError('Confinement failed; code not executed: ' + se[:1500])
        png, _ = Desktop(0.0).screenshot_png()
        content = [{'type': 'text', 'text': f'Exit code: {rc}; timeout: {timed}\nSTDOUT (first 30000 bytes):\n{so}\nSTDERR (first 10000 bytes):\n{se}'}, {'type': 'image', 'mimeType': 'image/png', 'data': base64.b64encode(png).decode()}]
        return content
    finally:
        LOCK.release()

def handle(msg):
    mid = msg.get('id'); method = msg.get('method')
    if mid is None: return
    try:
        if method == 'initialize': out = {'protocolVersion': (msg.get('params') or {}).get('protocolVersion', '2025-06-18'), 'capabilities': {'tools': {}}, 'serverInfo': {'name': 'helix-code', 'version': '1'}}
        elif method == 'tools/list': out = {'tools': [TOOL]}
        elif method == 'ping': out = {}
        elif method == 'tools/call':
            params = msg['params']
            if params['name'] != 'run_code': raise ValueError('Unknown tool')
            out = {'content': execute(params.get('arguments') or {}), 'isError': False}
        else: raise ValueError('Unknown method')
    except Exception as e:
        out = {'content': [{'type': 'text', 'text': f'{type(e).__name__}: {e}'}], 'isError': True}
    print(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': out}), flush=True)

if __name__ == '__main__':
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            # Receive overlapping JSON-RPC calls rather than silently queueing
            # them: execute()'s nonblocking lock refuses the second action.
            if msg.get('method') == 'tools/call':
                threading.Thread(target=handle, args=(msg,), daemon=True).start()
            else:
                handle(msg)
        except ValueError: pass
