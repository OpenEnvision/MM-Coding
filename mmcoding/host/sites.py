"""Case-owned, loopback-only access to the OSWorld 2.0 websites.

The OSWorld 2.0 task websites (``*.osworld.test``) run on the host behind a Caddy proxy (see the
upstream OSWorld-web deployment). The host owns Caddy, a 127.0.0.1:80 -> Caddy proxy and the host
``/etc/hosts`` entries (``install_hosts`` is an explicit helper, never called by ``attach``).

Call ``attach(env, sshkey, password, config)`` after authorizing SSH and setting ``env.vm_ip``,
before task reset / sandbox relay setup. Keep its return value alive until after ``env.close()``,
then call ``close()`` in a finally block. The separate SSH connection is never reused for the
agent transport. The guest's root ``/etc/hosts`` is also visible in the sandbox; the sandbox's
fixed port-80 Unix relay provides namespace access. No firewall, namespace, model or evaluator
hooks are installed.

Configuration: a JSON object with ``suffix``, ``host_port``, ``guest_forward_port``,
``hostnames``, optional ``gitlab_domain`` (= ``gitlab.<suffix>``) and optional
``repo_commit`` / ``submodules`` provenance, read from ``RV_WEBSITE_CONFIG`` or
``DEFAULT_CONFIG``. With GitLab enabled, ``host_port`` points to the Host-header router, not
directly to Caddy. The launcher calls ``install_process_resolution`` before importing task classes.
Guest GitLab name resolution uses the root-owned responder ``DNS_SCRIPT`` (``guest/website_dns.py``,
installed by ``deploy.docker_new_env``) and systemd-resolved. No host-global resolver is changed.
A missing default configuration disables the bridge; an explicitly selected bad or missing
configuration fails closed. This module neither resets app state nor shares cookies: the runner
must isolate website state per case.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import select
import shlex
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from mmcoding import paths

DEFAULT_CONFIG = paths.DATA / "private" / "osworld_web.json"
LOOPBACK = "127.0.0.1"
UNIT = "rv-web-loopback"
DNS_UNIT = "rv-web-gitlab-dns"
DNS_SCRIPT = "/usr/local/lib/rv-web-dns.py"
MARKER = "# rv-web-loopback"
# App labels from the pinned OSWorld-web */web-compose.yml, not a suffix wildcard.
APP_LABELS = frozenset("""
awsconsole basesite budgetwise calendar careerlink cloudcrm dinogame eventix expenseflow
formcraft glbviewer interfacinglinux mailhub overleaf overleaf-collab reviewsphere
slidepuzzle streamview studio.streamview teamchat travelhubpro trippza vaultbank
visaapplication wandb insurance-claim
""".split())
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


@dataclass(frozen=True)
class WebsiteConfig:
    suffix: str
    host_port: int
    guest_forward_port: int
    hostnames: tuple[str, ...]
    gitlab_domain: str | None = None

    def __post_init__(self):
        if (not isinstance(self.suffix, str) or len(self.suffix) > 200
                or not all(_DNS_LABEL.fullmatch(p) for p in self.suffix.split("."))):
            raise ValueError("website suffix must be a lower-case DNS domain")
        for name, port in (("host_port", self.host_port),
                           ("guest_forward_port", self.guest_forward_port)):
            if type(port) is not int or not 1024 <= port <= 65535:
                raise ValueError(f"{name} must be an integer unprivileged TCP port")
        if not isinstance(self.hostnames, (tuple, list)) or not self.hostnames:
            raise ValueError("hostnames must be a nonempty explicit list")
        if self.gitlab_domain is not None and self.gitlab_domain != f"gitlab.{self.suffix}":
            raise ValueError("gitlab_domain must be exactly gitlab.<website suffix>")
        allowed = {f"{app}.{self.suffix}" for app in APP_LABELS}
        if self.gitlab_domain:
            allowed.add(self.gitlab_domain)
        if any(not isinstance(h, str) or len(h) > 253 or h not in allowed
               for h in self.hostnames):
            raise ValueError("hostnames must be fixed pinned Caddy app names under suffix")
        if len(set(self.hostnames)) != len(self.hostnames):
            raise ValueError("duplicate website hostname")
        names = tuple(self.hostnames)
        if self.gitlab_domain and self.gitlab_domain not in names:
            names += (self.gitlab_domain,)
        object.__setattr__(self, "hostnames", names)


def load_config(config=None) -> WebsiteConfig | None:
    """Accept a config object, mapping or JSON path; absent default is a no-op."""
    if isinstance(config, WebsiteConfig):
        return config
    if config is None:
        explicit = os.environ.get("RV_WEBSITE_CONFIG")
        config = Path(explicit) if explicit else DEFAULT_CONFIG
        if not explicit and not config.exists():
            return None
    if isinstance(config, (str, os.PathLike)):
        config = json.loads(Path(config).read_text())
    if not isinstance(config, Mapping):
        raise ValueError("website config must be a JSON object")
    return WebsiteConfig(
        suffix=config.get("suffix", "osworld.test"),
        host_port=config.get("host_port", 18090),
        guest_forward_port=config.get("guest_forward_port", 18090),
        hostnames=config.get("hostnames"),
        gitlab_domain=config.get("gitlab_domain"),
    )


def _gitlab_name(host, domain: str) -> bool:
    """Match DNS labels, never textual lookalikes, URLs, or unrelated names."""
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    name = host.lower().removesuffix(".")
    return (len(name) <= 253 and all(_DNS_LABEL.fullmatch(p) for p in name.split("."))
            and (name == domain or name.endswith("." + domain)))


def install_process_resolution(config=None):
    """Scope Python DNS to exact GitLab/Pages names; return a conditional undo.

    Call explicitly at launcher startup BEFORE importing task classes. ``attach``
    also installs it, but task import can precede attachment. Loading config alone
    is deliberately side-effect free. No host resolv.conf or global DNS changes.
    One process owns one deployment; switching GitLab domains replaces our old
    wrapper instead of widening its scope. Socket arguments and nonmatches pass
    unchanged to the resolver that was present when installation occurred.
    """
    config = load_config(config)
    if config is None or config.gitlab_domain is None:
        return lambda: None
    current = socket.getaddrinfo
    if getattr(current, "_rv_gitlab_domain", None) == config.gitlab_domain:
        return lambda: None
    previous = getattr(current, "_rv_gitlab_original", current)
    domain = config.gitlab_domain

    def resolve(host, port, *args, **kwargs):
        target = LOOPBACK if _gitlab_name(host, domain) else host
        return previous(target, port, *args, **kwargs)

    resolve._rv_gitlab_domain = domain
    resolve._rv_gitlab_original = previous
    socket.getaddrinfo = resolve

    def restore():
        if socket.getaddrinfo is resolve:
            socket.getaddrinfo = previous
    return restore


def hosts_text(text: str, config: WebsiteConfig) -> str:
    """Idempotently replace only the explicit app aliases; preserve other aliases."""
    names = set(config.hostnames)
    lines = []
    for line in text.splitlines():
        if line.endswith(MARKER):
            # Replace this module's own entries, including an earlier suffix.
            continue
        address, separator, comment = line.partition("#")
        fields = address.split()
        if len(fields) > 1 and any(alias.lower() in names for alias in fields[1:]):
            aliases = [alias for alias in fields[1:] if alias.lower() not in names]
            if aliases:
                lines.append("\t".join([fields[0], *aliases]) +
                             (" #" + comment if separator else ""))
            elif separator:
                lines.append("#" + comment)
        else:
            lines.append(line)
    lines.extend(f"{LOOPBACK}\t{host} {MARKER}" for host in config.hostnames)
    return "\n".join(lines) + "\n"


def install_hosts(config, path: str | Path = "/etc/hosts") -> bool:
    """Explicit host deployment helper; requires caller's existing write permission.

    Write in place because /etc/hosts may be a bind mount. Entries intentionally
    persist after tunnel close. This is not safe for simultaneous deployments
    modifying the same hosts file; the deployment must serialize those operations.
    """
    config = load_config(config)
    if config is None:
        return False
    path = Path(path)
    before = path.read_text()
    after = hosts_text(before, config)
    if before != after:
        path.write_text(after)
    return True


def SSHGuest(host, key, user="user"):
    """Lazy upstream import: offline disabled runs do not need OSWorld imports."""
    from hybrid_agents.deepseek.transport import Guest
    return Guest(host, key, user=user)


def _probe_host(config):
    for host in config.hostnames:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, 80, type=socket.SOCK_STREAM)}
        if addresses != {LOOPBACK}:
            raise RuntimeError(f"host website name is not exclusively loopback: {host}")
        for port in (config.host_port, 80):
            conn = http.client.HTTPConnection(LOOPBACK, port, timeout=5)
            try:
                conn.request("GET", "/", headers={"Host": host, "Connection": "close"})
                status = conn.getresponse().status
                if not (200 <= status < 400 or status in (401, 403)):
                    raise RuntimeError(f"host website readiness failed: {host}:{port} HTTP {status}")
            finally:
                conn.close()


# Sent as trusted source, with validated hostnames inserted as data. No API state
# or cookies are returned to the host. No redirects are followed off loopback.
_GUEST_PROBE = '''import http.client, socket, time
hosts = HOSTS
ports = {80, FORWARD_PORT}
def check():
    listeners = set()
    for filename in ('/proc/net/tcp', '/proc/net/tcp6'):
        with open(filename) as f:
            for line in list(f)[1:]:
                row = line.split()
                address, port = row[1].split(':')
                if row[3] == '0A' and int(port, 16) in ports:
                    assert address == '0100007F', 'website listener is not IPv4 loopback'
                    listeners.add(int(port, 16))
    assert listeners == ports, 'website listeners missing'
    for host in hosts:
        assert {x[4][0] for x in socket.getaddrinfo(host, 80, type=socket.SOCK_STREAM)} == {'127.0.0.1'}
        c = http.client.HTTPConnection('127.0.0.1', 80, timeout=2)
        try:
            c.request('GET', '/', headers={'Host': host, 'Connection': 'close'})
            status = c.getresponse().status
            assert 200 <= status < 400 or status in (401, 403), 'website HTTP readiness failed'
        finally:
            c.close()
deadline = time.monotonic() + 30
while True:
    try:
        check()
        break
    except (OSError, AssertionError, http.client.HTTPException):
        if time.monotonic() >= deadline:
            raise
        time.sleep(.2)
'''


# Guest-only resolver ownership. Never rewrite the systemd-resolved target file
# behind /etc/resolv.conf, and never change its upstream/link configuration.
_GUEST_DNS_INSTALL = '''import json, os
from pathlib import Path
p=Path('/etc/resolv.conf')
b=Path('/run/rv-web-gitlab-resolver.json')
original=p.read_text()
assert any(line.split() == ['nameserver', '127.0.0.53'] for line in original.splitlines()), 'expected systemd-resolved stub'
value={'symlink': os.readlink(p) if p.is_symlink() else None, 'text': original}
replacement='nameserver 127.0.0.1\\noptions timeout:2 attempts:2\\n# rv-web-gitlab-dns\\n'
fd=os.open(b,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f: json.dump(value,f)
try:
 if p.is_symlink(): p.unlink()
 p.write_text(replacement)
except BaseException:
 if p.exists() or p.is_symlink(): p.unlink()
 if value['symlink'] is not None: p.symlink_to(value['symlink'])
 else: p.write_text(original)
 b.unlink()
 raise
'''

_GUEST_DNS_RESTORE = '''import json
from pathlib import Path
p=Path('/etc/resolv.conf')
b=Path('/run/rv-web-gitlab-resolver.json')
if b.exists():
 value=json.loads(b.read_text())
 replacement='nameserver 127.0.0.1\\noptions timeout:2 attempts:2\\n# rv-web-gitlab-dns\\n'
 assert not p.is_symlink() and p.read_text()==replacement, 'resolver ownership changed; preserve newer configuration'
 p.unlink()
 if value['symlink'] is not None: p.symlink_to(value['symlink'])
 else: p.write_text(value['text'])
 b.unlink()
'''


class WebsiteTunnel:
    """Own one transport, one guest service, and bounded daemon stream workers."""

    MAX_STREAMS = 128

    def __init__(self, guest, config, password):
        self.guest, self.config, self._password = guest, config, password
        self.transport = None
        self._forwarded = self._service = False
        self._dns_service = self._dns_route = False
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._streams = set()
        self._threads = set()

    def _command(self, argv, *, sudo=False, timeout=45):
        # Do not use Guest.command(): its reconnect-on-close behavior could create
        # an orphan SSH connection after env.close() has terminated the VM.
        command = shlex.join(argv)
        if sudo:
            command = "sudo -S -p '' -- " + command
        stdin, stdout, stderr = self.guest.ssh.exec_command(command, timeout=timeout)
        channel = stdout.channel
        try:
            if sudo:
                stdin.write(self._password + "\n")
                stdin.flush()
            stdin.close()
            deadline = time.monotonic() + timeout
            while True:
                # Discard all response bodies, including any privileged errors.
                if channel.recv_ready():
                    channel.recv(65536)
                if channel.recv_stderr_ready():
                    channel.recv_stderr(65536)
                if (channel.exit_status_ready() and not channel.recv_ready()
                        and not channel.recv_stderr_ready()):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("website guest setup timed out")
                time.sleep(.02)
            if channel.recv_exit_status() != 0:
                raise RuntimeError("website guest command failed")
        finally:
            channel.close()
            stdout.close()
            stderr.close()

    def _start_gitlab_dns(self):
        """Start the root-owned responder ``DNS_SCRIPT``; it forwards nonmatches to resolved.

        Responder CLI: --domain DOMAIN --listen 127.0.0.1 --port 53
        --upstream 127.0.0.53. Scoped A returns loopback and AAAA returns NODATA;
        every nonmatching query forwards to the original systemd-resolved stub.
        Both UDP/TCP DNS are required; namespace DNS installation is separate.
        """
        domain = self.config.gitlab_domain
        if domain is None:
            return
        # Fail closed without the trusted responder; no apt/network
        # installs, replacement DNS implementation, or host-global DNS changes.
        check = ("import os, stat\n"
                 f"p={DNS_SCRIPT!r}\ns=os.lstat(p)\n"
                 "assert stat.S_ISREG(s.st_mode) and s.st_uid == 0 and not s.st_mode & 0o022\n")
        self._command(["/usr/bin/python3", "-c", check], sudo=True)
        self._command([
            "/usr/bin/systemd-run", "--quiet", "--collect", f"--unit={DNS_UNIT}",
            "--property=Type=exec", "--property=Restart=no",
            "/usr/bin/python3", DNS_SCRIPT, "--domain", domain,
            "--listen", LOOPBACK, "--port", "53", "--upstream", "127.0.0.53",
        ], sudo=True)
        self._dns_service = True
        # systemd-resolved rejects per-link DNS on loopback on some releases.
        # Replace ONLY this dedicated VM's stub link, preserving its original
        # target unchanged. Non-GitLab queries still use the original stub.
        self._dns_route = True
        self._command(["/usr/bin/python3", "-c", _GUEST_DNS_INSTALL], sudo=True)
        source = ("import socket, time\n"
                  f"name='rv-wildcard-readiness.' + {domain!r}\n"
                  "deadline=time.monotonic()+15\n"
                  "while True:\n"
                  " try:\n"
                  "  assert {r[4][0] for r in socket.getaddrinfo(name,80,type=socket.SOCK_STREAM)} == {'127.0.0.1'}\n"
                  "  break\n"
                  " except (OSError, AssertionError):\n"
                  "  if time.monotonic() >= deadline: raise\n"
                  "  time.sleep(.2)\n")
        self._command(["/usr/bin/python3", "-c", source], timeout=25)

    def start(self):
        try:
            install_process_resolution(self.config)
            _probe_host(self.config)
            self.guest.connect()
            self.transport = self.guest.ssh.get_transport()
            self.transport.set_keepalive(20)
            self.transport.request_port_forward(
                LOOPBACK, self.config.guest_forward_port, self._handle)
            self._forwarded = True
            # Run the same pure hosts transformation without uploading a script or
            # copying deployment provenance/passwords into the VM filesystem.
            import inspect
            source = ("from pathlib import Path\nfrom types import SimpleNamespace\n"
                      f"LOOPBACK={LOOPBACK!r}\nMARKER={MARKER!r}\nWebsiteConfig=object\n"
                      + inspect.getsource(hosts_text) +
                      f"\np=Path('/etc/hosts')\nc=SimpleNamespace(hostnames={self.config.hostnames!r})\n"
                      "p.write_text(hosts_text(p.read_text(), c))\n")
            self._command(["/usr/bin/python3", "-c", source], sudo=True)
            self._start_gitlab_dns()
            # Do not stop an existing service: concurrent attachment must fail,
            # never steal another case's website forwarding.
            self._command([
                "/usr/bin/systemd-run", "--quiet", "--collect", f"--unit={UNIT}",
                "--property=Type=exec", "--property=Restart=no",
                "/usr/bin/socat", "TCP-LISTEN:80,bind=127.0.0.1,reuseaddr,fork",
                f"TCP:127.0.0.1:{self.config.guest_forward_port}",
            ], sudo=True)
            self._service = True
            probe = _GUEST_PROBE.replace("HOSTS", repr(self.config.hostnames)).replace(
                "FORWARD_PORT", str(self.config.guest_forward_port))
            self._command(["/usr/bin/python3", "-c", probe], timeout=90)
            return self
        except BaseException:
            self.close()
            raise

    def _handle(self, channel, origin, server):
        # Called on Paramiko's transport thread: never splice or connect here.
        with self._lock:
            if (self._closed.is_set() or len(self._threads) >= self.MAX_STREAMS
                    or server != (LOOPBACK, self.config.guest_forward_port)):
                channel.close()
                return
            self._streams.add(channel)
            worker = threading.Thread(target=self._relay, args=(channel,),
                                      name="rv-web-stream", daemon=True)
            self._threads.add(worker)
            worker.start()

    def _relay(self, channel):
        local = None
        try:
            local = socket.create_connection((LOOPBACK, self.config.host_port), timeout=5)
            with self._lock:
                if self._closed.is_set():
                    return
                self._streams.add(local)
            channel.settimeout(5)
            # Constant memory per stream, bounded sends, preserve half-close and
            # raw bytes (Host headers, upgrades, cookie boundaries all untouched).
            readers = [channel, local]
            while readers and not self._closed.is_set():
                ready, _, _ = select.select(readers, [], [], .5)
                for source in ready:
                    target = local if source is channel else channel
                    data = source.recv(65536)
                    if data:
                        target.sendall(data)
                    else:
                        readers.remove(source)
                        target.shutdown(socket.SHUT_WR)
        except (OSError, EOFError, ValueError):
            # select can see a descriptor concurrently closed by close().
            pass
        finally:
            channel.close()
            if local is not None:
                local.close()
            with self._lock:
                self._streams.discard(channel)
                self._streams.discard(local)
                self._threads.discard(threading.current_thread())

    def close(self):
        """Idempotent best-effort cleanup, also safe after the VM has disappeared."""
        with self._lock:
            if self._closed.is_set():
                return
            self._closed.set()
            streams, threads = tuple(self._streams), tuple(self._threads)
        cancellation = None
        if self._forwarded:
            # Paramiko's global-request wait has no timeout. A dead VM must not
            # hang runner cleanup; closing our transport below releases the wait.
            def cancel():
                try:
                    self.transport.cancel_port_forward(LOOPBACK, self.config.guest_forward_port)
                except Exception:
                    pass
            cancellation = threading.Thread(target=cancel, name="rv-web-cancel", daemon=True)
            cancellation.start()
            cancellation.join(timeout=1)
        for stream in streams:
            try:
                stream.close()
            except Exception:
                pass
        if self._service:
            try:
                self._command(["/usr/bin/systemctl", "stop", UNIT], sudo=True, timeout=3)
            except Exception:
                pass
        if self._dns_route:
            try:
                self._command(["/usr/bin/python3", "-c", _GUEST_DNS_RESTORE], sudo=True, timeout=3)
            except Exception:
                pass
        if self._dns_service:
            try:
                self._command(["/usr/bin/systemctl", "stop", DNS_UNIT], sudo=True, timeout=3)
            except Exception:
                pass
        try:
            self.guest.close()
        finally:
            self._password = ""
            deadline = time.monotonic() + 1
            if cancellation is not None:
                cancellation.join(timeout=max(0, deadline - time.monotonic()))
            for worker in threads:
                worker.join(timeout=max(0, deadline - time.monotonic()))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def attach(env, sshkey, password, config=None) -> WebsiteTunnel | None:
    """Connect before env.reset(); return None only for an absent default config."""
    config = load_config(config)
    if config is None:
        return None
    if not isinstance(password, str) or "\n" in password or "\r" in password or "\0" in password:
        raise ValueError("invalid guest sudo password")
    guest = SSHGuest(env.vm_ip, str(sshkey), user="user")
    return WebsiteTunnel(guest, config, password).start()
