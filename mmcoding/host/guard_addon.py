"""mitmproxy addon of the optional network guard: block benchmark answer sources, log every flow.

The guard keeps an agent from fetching the benchmark's own task definitions, evaluators and
expected answers from the internet (source repositories, mirrors, dataset hubs, code search,
page-fetch services, archives). ``mmcoding.host.guard`` redirects the TCP traffic of one case's
container to this proxy while the agent runs; this addon decides per request.

Run it on the host in transparent mode, with the configuration used by ``mmcoding.host.guard``
(``RV_GUARD_CONFIG``, default ``configs/guard.json``, else ``configs/guard.example.json``)::

    RV_GUARD_CONFIG=configs/guard.json mitmdump --mode transparent --listen-port 8893 \\
        --set confdir=~/.mitmproxy --set block_global=false -s mmcoding/host/guard_addon.py

``--listen-port`` must equal ``port`` and ``confdir`` must hold the CA named by ``ca_cert``
(mitmproxy creates ``mitmproxy-ca-cert.pem`` on first start). Run it as the systemd unit named by
``service`` so ``guard.harden`` can check that it is active. The addon needs only the standard
library and mitmproxy; the rule functions can be imported without mitmproxy.

Per source IP the host writes a registry entry ``<registry>/<ip>.json`` = {"log": <jsonl path>,
"terms": [task identifiers], "task": <task id>, "attempt": ..., "container": ...}. Requests from
unregistered IPs pass unchanged and are logged to ``<log_dir>/global.jsonl``.

Rules:

* TLS is intercepted only for hosts that can serve benchmark sources or copies (``INTERCEPT_SNI``);
  every other TLS connection passes through with its original certificate. Plain HTTP is
  inspected for every host. Hosts in ``passthrough_hosts`` are never intercepted.
* URL rules (``BLOCK``) match the lower-cased, repeatedly percent-decoded URL, so wrapped URLs
  (``ghproxy``, ``jsdelivr /gh/``, ``r.jina.ai/<url>``) are covered; task identifiers of at least
  8 characters are blocked too. Local benchmark sites (``*.osworld.test``) are never blocked.
* Bodies of non-GET requests to intercepted hosts are matched by the same rules, also after
  ``\\uXXXX`` unescaping; a streamed (uninspectable) body sent to a page-fetch service is blocked.
* Responses are checked independently of the URL: a body that carries this task's configuration
  (a task identifier of at least 32 characters next to evaluator keys, also inside base64
  ``content`` of API blobs) or a listing of the benchmark's task tree is replaced by 404.
* Compressed transfers are opened: git smart-HTTP pack responses (side-band demultiplexed, every
  object inflated), zip entry names and gzip / tar.gz content are searched for the task's
  identifiers; ordinary clones and downloads are unaffected.
* ``task_allow`` (optional JSON ``{"tasks": {task_id: [url fragment, ...]}}``): lower-cased URL
  fragments without scheme that are task input material, e.g. a repository the instruction itself
  asks to clone or an input file the task's own setup references. They are removed from the URL
  before the rules run, so exactly that resource is allowed. Build it from the task instructions
  and the tasks' input files only, never from evaluator or expected-output files. Without it every
  matching URL is blocked.

A blocked request or response gets ``404`` with the header ``X-RV-Blocked: <rule>``.
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
import zlib
from pathlib import Path
from urllib.parse import unquote

try:
    from mitmproxy import http, tls  # noqa: F401  (tls: annotations only)
except ImportError:  # rule functions stay importable for the host side and the tests
    http = tls = None

REPO = Path(__file__).resolve().parents[2]
PATH_KEYS = ("registry", "log_dir", "ca_cert", "nss_deb_dir", "task_allow")


def config_path() -> Path:
    explicit = os.environ.get("RV_GUARD_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    own = REPO / "configs" / "guard.json"
    return own if own.is_file() else REPO / "configs" / "guard.example.json"


def resolve(value: str | None) -> Path | None:
    """A configured path: ``$VARS`` and ``~`` expanded, relative paths against the repository."""
    if not value:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(value)))
    return path if path.is_absolute() else REPO / path


def load_config(path: Path | None = None) -> dict:
    """The guard configuration with path-valued keys resolved to absolute ``Path`` objects."""
    cfg = {key: value for key, value in json.loads((path or config_path()).read_text()).items()
           if not key.startswith("_")}
    for key in PATH_KEYS:
        cfg[key] = resolve(cfg.get(key))
    for key in ("registry", "log_dir"):
        if cfg[key] is None:
            raise ValueError(f"guard configuration needs {key!r}")
    cfg.setdefault("passthrough_hosts", [])
    cfg.setdefault("direct_cidrs", [])
    cfg.setdefault("nss_debs", {})
    cfg["rules_version"] = str(cfg.get("rules_version", "1"))
    return cfg


CONFIG = load_config()
REGISTRY = CONFIG["registry"]
GLOBAL_LOG = CONFIG["log_dir"] / "global.jsonl"
TASK_ALLOW_PATH = CONFIG["task_allow"]
RULES_VERSION = CONFIG["rules_version"]

# TLS is never intercepted for these SNI hosts (model and control endpoints): bytes pass unchanged.
PASSTHROUGH_SNI = tuple(re.compile(r"(^|\.)" + re.escape(host.lower().strip(".")) + r"$")
                        for host in CONFIG["passthrough_hosts"])
# TLS is intercepted only for hosts that can serve benchmark sources or copies (code hosting, raw/CDN
# mirrors, dataset hubs, code search, fetch proxies, archives).
INTERCEPT_SNI = re.compile(
    r"(^|\.)(github\.com|githubusercontent\.com|github\.io|githubassets\.com|ghcr\.io"
    r"|huggingface\.co|hf\.co|hf-mirror\.com|modelscope\.cn|kaggle\.com|kaggleusercontent\.com"
    r"|jsdelivr\.net|statically\.io|githack\.com|gitcdn\.link|gitee\.com|gitcode\.com|gitcode\.net"
    r"|gitlab\.com|codeberg\.org|bitbucket\.org|sourceforge\.net|grep\.app|sourcegraph\.com|searchcode\.com"
    r"|publicwww\.com|jina\.ai|archive\.org|archive\.ph|archive\.today|12ft\.io|translate\.goog"
    r"|googleusercontent\.com|xlang\.ai|docker\.com|docker\.io)$"
    r"|github|ghproxy|gh-proxy|ghfast|osworld|xlang|agenthle|lastexam"
    r"|jsd|jsdelivr|onmicrosoft\.cn|statically|githack|fastgit|kgithub|kkgithub|githubfast|gitclone|gitmirror|fgit|bgithub"
    r"|ghps|moeyy|99988866|gitee|gitcode|gitlab|sourceforge|hf-mirror|huggingface|hf\.co|modelscope|archive")


def intercept_host(sni: str) -> bool:
    return bool(sni) and bool(INTERCEPT_SNI.search(sni))


# Local benchmark sites (OSWorld 2.0 *.osworld.test etc.) are never blocked by the 'osworld' rule.
ALLOW_HOST = (re.compile(r"(^|\.)osworld\.test$"), re.compile(r"(^|\.)osworld\.local$"))

BLOCK = [
    # OSWorld / OSWorld-V2 source, tasks, mirrors and proxied forms (xlang-ai GitHub org, xlangai HF org)
    ("osworld_source", re.compile(r"xlang-ai|xlangai")),
    ("osworld_name", re.compile(r"osworld")),
    ("osworld_paths", re.compile(r"evaluation_examples(_v2)?/|desktop_env/evaluators|examples_clarified|ubuntu_osworld_file_cache")),
    # Third-party repositories that hold copies of the OSWorld task files
    ("osworld_known_copies", re.compile(r"simular-ai/cua_reliability|cua_reliability|sai-osworld|exp-osworld|evocua")),
    # Numeric GitHub API id of the OSWorld repository
    ("osworld_repo_id", re.compile(r"api\.github\.com/repositories/705433049(/|$|\?)")),
    # Agents' Last Exam source / images / data
    ("ale_source", re.compile(r"agents-last-exam|agentslastexam|agents_last_exam|agenthle|rdi-berkeley")),
    # Code search and code-search frontends
    ("code_search", re.compile(r"^https?://(api\.)?github\.com/search/code|^https?://github\.com/search\?[^#]*type=code"
                               r"|^https?://([a-z0-9-]+\.)*(grep\.app|sourcegraph\.com|searchcode\.com|publicwww\.com)(/|$)")),
]

_lock = threading.Lock()
_cache: dict[str, tuple[float, dict | None]] = {}


def registry(ip: str) -> dict | None:
    path = REGISTRY / f"{ip}.json"
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return None
    hit = _cache.get(ip)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        value = json.loads(path.read_text())
    except Exception:
        value = None
    _cache[ip] = (mtime, value)
    return value


def normalise(url: str) -> str:
    text = url
    for _ in range(3):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    return text.lower()


_allow_cache: list = [None, {}]


def task_allow(task: str) -> list[str]:
    """URL fragments that are input material of this task (``task_allow`` file; none without it)."""
    if TASK_ALLOW_PATH is None:
        return []
    try:
        mtime = TASK_ALLOW_PATH.stat().st_mtime
    except FileNotFoundError:
        return []
    if _allow_cache[0] != mtime:
        _allow_cache[1] = json.loads(TASK_ALLOW_PATH.read_text()).get("tasks", {})
        _allow_cache[0] = mtime
    return _allow_cache[1].get(task, [])


def decide(url: str, host: str, terms: list[str], allow: list[str] = ()) -> str | None:
    """Name of the first matching block rule for a URL (or a request body), else None."""
    host = (host or "").lower().rstrip(".")
    if any(p.search(host) for p in ALLOW_HOST):
        return None
    norm = normalise(url)
    for token in allow:
        norm = norm.replace(token.lower(), "")
    for name, pattern in BLOCK:
        if pattern.search(norm):
            return name
    for term in terms:
        t = term.lower().strip()
        if len(t) >= 8 and t in norm:
            return "task_identifier"
    return None


FETCH_PROXY = re.compile(r"(^|\.)(jina\.ai|12ft\.io|archive\.org|archive\.ph|archive\.today|translate\.goog)$")


def body_text(flow) -> str | None:
    """Decoded request body (<= 1 MiB) or None when it is streamed/unavailable."""
    if flow.request.stream:
        return None
    try:
        raw = flow.request.get_content(strict=False) or b""
    except Exception:
        return None
    return raw[:1048576].decode("utf-8", "replace")


UESC = re.compile(r"\\u([0-9a-fA-F]{4})")
ANSWER_KEYS = re.compile(r'(\\*")(evaluator|expected|func)\\*"\s*:')
LISTING = re.compile(r"evaluation_examples(_v2)?/examples/|xlang-ai/osworld|xlangai/ubuntu_osworld|examples_clarified")
B64 = re.compile(r'"content"\s*:\s*"([A-Za-z0-9+/=\\n\s]{40,})"')


def unescape(text: str) -> str:
    """Undo (possibly repeated) \\uXXXX / JSON string escaping so wrapped URLs and nested JSON match the rules."""
    for _ in range(3):
        if "\\u" not in text and '\\"' not in text:
            break
        new = UESC.sub(lambda m: chr(int(m.group(1), 16)), text.replace("\\\\u", "\\u")).replace('\\"', '"').replace("\\/", "/")
        if new == text:
            break
        text = new
    return text


INFLATE_CAP = 768 * 1048576


def _needles(terms):
    return [t.lower().strip().encode() for t in terms if len(t.strip()) >= 32]


def git_pack_hit(data: bytes, needles) -> bool:
    """True when an inflated object of a git smart-HTTP pack response contains one of the needles."""
    out = bytearray(); i = 0; n = len(data)
    while i + 4 <= n:  # pkt-line framing; side band 1 carries the pack
        try:
            ln = int(data[i:i + 4], 16)
        except ValueError:
            break
        if ln < 4:
            i += 4; continue
        pkt = data[i + 4:i + ln]; i += ln
        if pkt[:1] == b"\x01":
            out += pkt[1:]
        elif pkt[:4] == b"PACK":
            out += pkt
    pack = bytes(out) if out else data
    p = pack.find(b"PACK")
    if p < 0:
        return False
    count = int.from_bytes(pack[p + 8:p + 12], "big"); pos = p + 12; total = 0
    mv = memoryview(pack)
    for _ in range(count):
        if pos >= len(pack):
            break
        c = pack[pos]; pos += 1; typ = (c >> 4) & 7
        while c & 0x80:
            c = pack[pos]; pos += 1
        if typ == 6:  # ofs-delta: variable-length base offset
            c = pack[pos]; pos += 1
            while c & 0x80:
                c = pack[pos]; pos += 1
        elif typ == 7:  # ref-delta: 20-byte base id
            pos += 20
        d = zlib.decompressobj(); buf = bytearray()
        while not d.eof and pos < len(pack):
            chunk = mv[pos:pos + 65536]
            buf += d.decompress(chunk)
            used = len(chunk) - len(d.unused_data)
            pos += used
            if d.eof:
                break
            if used == 0:
                return False
        total += len(buf)
        low = bytes(buf).lower()
        if any(nd in low for nd in needles):
            return True
        if total > INFLATE_CAP:
            return False
    return False


def archive_hit(data: bytes, needles) -> bool:
    low = data[:INFLATE_CAP].lower()
    if any(nd in low for nd in needles):  # zip local/central headers carry entry names in plain text
        return True
    if data[:2] == b"\x1f\x8b":  # gzip / tar.gz
        try:
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            raw = d.decompress(data, INFLATE_CAP).lower()
            return any(nd in raw for nd in needles)
        except Exception:
            return False
    return False


def response_rule(text: str, terms: list[str]) -> str | None:
    """URL-independent check of a fetched body (already decompressed)."""
    views = [text]
    if '"encoding"' in text and "base64" in text:
        for m in B64.finditer(text[:8388608]):
            try:
                views.append(base64.b64decode(m.group(1).replace("\\n", "").replace("\n", ""), validate=False)
                             .decode("utf-8", "replace"))
            except Exception:
                pass
    for v in views:
        low = unescape(v).lower()
        if LISTING.search(low):
            return "response_osworld_listing"
        for term in terms:
            t = term.lower().strip()
            if len(t) >= 32 and t in low and ANSWER_KEYS.search(low):
                return "response_task_config"
    return None


def write(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as stream:
            stream.write(line)


def client_ip(flow) -> str:
    peer = flow.client_conn.peername
    ip = peer[0] if peer else ""
    return ip[7:] if ip.startswith("::ffff:") else ip


def _blocked(rule: str):
    return http.Response.make(404, b"Not Found\n", {"Content-Type": "text/plain", "X-RV-Blocked": rule})


class Guard:
    def tls_clienthello(self, data: tls.ClientHelloData) -> None:
        sni = (data.client_hello.sni or "").lower()
        if any(p.search(sni) for p in PASSTHROUGH_SNI) or not intercept_host(sni):
            data.ignore_connection = True
            ip = data.context.client.peername[0] if data.context.client.peername else ""
            ip = ip[7:] if ip.startswith("::ffff:") else ip
            reg = registry(ip)
            write(Path(reg["log"]) if reg else GLOBAL_LOG,
                  {"t": time.time(), "ip": ip, "event": "tls_passthrough", "sni": sni, "rules": RULES_VERSION})

    def request(self, flow: http.HTTPFlow) -> None:
        ip = client_ip(flow)
        reg = registry(ip)
        url = flow.request.pretty_url
        host = flow.request.pretty_host
        allow = task_allow(reg.get("task", "")) if reg else []
        rule = decide(url, host, (reg or {}).get("terms", []), allow) if reg else None
        body_rule = None
        if reg and not rule and flow.request.method not in ("GET", "HEAD", "OPTIONS"):
            body = body_text(flow)
            if body is None:
                if FETCH_PROXY.search(host.lower().rstrip(".")):
                    body_rule = "fetch_proxy_body_uninspectable"
            elif body:
                hit = decide(body, host, reg.get("terms", []), allow) or decide(unescape(body), host, reg.get("terms", []), allow)
                body_rule = ("body_" + hit) if hit else None
            rule = body_rule
        allowed_by_task = bool(reg and allow and not rule and decide(url, host, reg.get("terms", [])))
        record = {"t": time.time(), "ip": ip, "event": "request", "method": flow.request.method,
                  "url": url[:4000], "host": host, "blocked": bool(rule), "rule": rule, "rules": RULES_VERSION,
                  "registered": bool(reg), "allowed_by_task": allowed_by_task, "body_rule": body_rule}
        if reg:
            record.update(attempt=reg.get("attempt"), unit=reg.get("unit"), task=reg.get("task"))
        write(Path(reg["log"]) if reg else GLOBAL_LOG, record)
        if rule:
            flow.response = _blocked(rule)

    def response(self, flow: http.HTTPFlow) -> None:
        ip = client_ip(flow)
        reg = registry(ip)
        if not reg or flow.response is None or flow.response.headers.get("X-RV-Blocked"):
            return
        url = flow.request.pretty_url
        allow = task_allow(reg.get("task", ""))
        if allow and any(tok.lower() in normalise(url) for tok in allow):
            return  # task input material (task_allow)
        if flow.response.stream:
            rule = None  # too large to inspect (> 16 MiB); the URL rules still applied
        else:
            try:
                text = (flow.response.get_content(strict=False) or b"")[:16777216].decode("utf-8", "replace")
            except Exception:
                text = ""
            rule = response_rule(text, reg.get("terms", [])) if text else None
            needles = _needles(reg.get("terms", []))
            if not rule and needles:
                raw = flow.response.get_content(strict=False) or b""
                ctype = (flow.response.headers.get("content-type") or "").lower()
                if "git-upload-pack" in ctype or url.split("?")[0].endswith("/git-upload-pack"):
                    rule = "response_git_pack_task" if git_pack_hit(raw, needles) else None
                elif raw[:2] in (b"PK", b"\x1f\x8b") or any(x in ctype for x in ("zip", "gzip", "x-tar", "octet-stream")):
                    rule = "response_archive_task" if archive_hit(raw, needles) else None
        if rule:
            write(Path(reg["log"]), {"t": time.time(), "ip": ip, "event": "response_blocked", "method": flow.request.method,
                                     "url": url[:4000], "host": flow.request.pretty_host, "rule": rule,
                                     "status": flow.response.status_code, "rules": RULES_VERSION,
                                     "attempt": reg.get("attempt"), "task": reg.get("task")})
            flow.response = _blocked(rule)

    def tcp_start(self, flow) -> None:
        ip = client_ip(flow)
        reg = registry(ip)
        server = flow.server_conn.address
        write(Path(reg["log"]) if reg else GLOBAL_LOG,
              {"t": time.time(), "ip": ip, "event": "raw_tcp", "server": list(server) if server else None,
               "rules": RULES_VERSION})


addons = [Guard()]
