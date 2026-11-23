"""Optional network guard, host side (off unless ``deploy.install(..., network_guard=True)``).

While the agent runs, every external TCP connection of the case's container (matched by IP and
MAC) is redirected to the host's transparent TLS proxy (mitmproxy with ``guard_addon.py``), which
answers 404 for benchmark answer sources and logs every URL; UDP 443 is dropped so QUIC falls back
to TCP. Private ranges and ``direct_cidrs`` are never redirected. The proxy CA certificate (public
part only; the key stays on the host) is installed in the guest: system store, Python certifi
bundles, NSS databases (Chrome), Chrome ``CACertificates`` policy and a login-shell profile; the
agent worker gets ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` / ``PIP_CERT`` / ``NODE_EXTRA_CA_CERTS``
from ``rv_run``. The redirect is removed once the agent is quiescent, so task setup and evaluation
are unchanged. Every step fails closed.

Configuration: ``configs/guard.example.json`` (copy to ``configs/guard.json`` or point
``RV_GUARD_CONFIG`` at a file; see ``guard_addon.load_config``). Per case the host environment may
set ``RV_GUARD_LOG`` (request log, default ``<log_dir>/<task>-<ip>-<time>.jsonl``) and
``RV_GUARD_TERMS`` (JSON list of task identifiers that are blocked in URLs and searched for in
responses; default ``task_terms(task)``). The host needs root for ``iptables`` / ``conntrack`` and access to the
Docker CLI.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path

from mmcoding.host import guard_addon

GUEST_CA = "/usr/local/share/ca-certificates/rv-answer-proxy.crt"  # path checked by guest/rv_run.py
PRIVATE = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10",
           "127.0.0.0/8", "224.0.0.0/4")
IPT = ["iptables", "-w", "10"]

INSTALL = r"""set -e
CA=/usr/local/share/ca-certificates/rv-answer-proxy.crt
install -m 0644 /tmp/rv-answer-proxy.crt "$CA"
rm -f /tmp/rv-answer-proxy.crt
update-ca-certificates >/dev/null 2>&1 || true
grep -q "$(sed -n 2p "$CA")" /etc/ssl/certs/ca-certificates.crt && echo system_store=ok || echo system_store=missing
n=0
for f in $(find / -xdev \( -path /proc -o -path /sys -o -path /dev -o -path /run \) -prune -o -name cacert.pem -type f -print 2>/dev/null); do
  grep -q RVAP-MARK "$f" || { printf '\n# RVAP-MARK rv-answer-proxy\n' >> "$f"; cat "$CA" >> "$f"; n=$((n+1)); }
done
echo certifi_patched=$n
CU=$(command -v certutil || true)
NSSLIB=
if [ -z "$CU" ]; then
  # No certutil in the image: unpack the host-shipped, hash-pinned NSS debs privately
  # (dpkg -x: nothing is installed, package database untouched), own libs via LD_LIBRARY_PATH.
  for d in /tmp/rv-nsstools/*.deb; do [ -f "$d" ] && dpkg -x "$d" /tmp/rv-nsstools/x; done
  [ -x /tmp/rv-nsstools/x/usr/bin/certutil ] && CU=/tmp/rv-nsstools/x/usr/bin/certutil && NSSLIB=/tmp/rv-nsstools/x/usr/lib/x86_64-linux-gnu
fi
if [ -n "$CU" ] && [ -x "$CU" ]; then
  certutil() { LD_LIBRARY_PATH="$NSSLIB" "$CU" "$@"; }
  for h in /home/user /root; do
    db="$h/.pki/nssdb"
    [ -d "$db" ] || { mkdir -p "$db"; certutil -N -d "sql:$db" --empty-password; }
    certutil -d "sql:$db" -D -n rv-answer-proxy >/dev/null 2>&1 || true
    certutil -d "sql:$db" -A -t "C,," -n rv-answer-proxy -i "$CA"
    chown -R "$(stat -c %u:%g "$h")" "$h/.pki"
    certutil -d "sql:$db" -L -n rv-answer-proxy >/dev/null 2>&1 && echo "nss=$db"
  done
else
  echo nss=no_certutil
fi
rm -rf /tmp/rv-nsstools
if command -v openssl >/dev/null 2>&1; then
  der=$(openssl x509 -in "$CA" -outform der | base64 -w0)
  for d in /etc/opt/chrome/policies/managed /etc/chromium/policies/managed /etc/chromium-browser/policies/managed; do
    mkdir -p "$d"; printf '{"CACertificates": ["%s"]}\n' "$der" > "$d/rv-answer-proxy.json"
  done
  echo chrome_policy=ok
fi
printf 'export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt PIP_CERT=/etc/ssl/certs/ca-certificates.crt NODE_EXTRA_CA_CERTS=%s\n' "$CA" > /etc/profile.d/rv-answer-proxy.sh
echo done
"""

# Self-check from inside the guest: a benchmark source is blocked, ordinary GitHub is reachable.
CHECK = r"""blk=$(curl -s -o /dev/null -D - --max-time 30 https://raw.githubusercontent.com/xlang-ai/OSWorld/main/README.md | tr -d '\r' | awk 'tolower($1)=="x-rv-blocked:"{print $2}')
ok=$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 https://github.com/)
echo "blocked_rule=${blk:-none} github=$ok"
"""

# Browser HTTP / code caches of the image may hold pages fetched when the image was built (for
# example benchmark task files); they are cleared before the agent starts. Profiles, history,
# bookmarks, cookies and settings (possible task state) are untouched.
PURGE = r"""n=0
for d in /home/*/.cache/google-chrome /home/*/.cache/chromium /home/*/.cache/google-chrome-beta /root/.cache/google-chrome /root/.cache/chromium \
         /home/*/.config/google-chrome/*/"Service Worker"/CacheStorage /home/*/.config/chromium/*/"Service Worker"/CacheStorage \
         /home/*/snap/chromium/common/.cache/chromium /home/*/.cache/mozilla/firefox; do
  [ -d "$d" ] || continue
  c=$(find "$d" -type f 2>/dev/null | wc -l); find "$d" -mindepth 1 -delete 2>/dev/null || true; n=$((n+c))
done
echo browser_cache_purged=$n
"""


class GuardError(RuntimeError):
    pass


def config() -> dict:
    return guard_addon.load_config()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ipt(args, check=True):
    r = subprocess.run(IPT + args, capture_output=True, text=True)
    if check and r.returncode:
        raise GuardError("iptables " + " ".join(args) + ": " + r.stderr.strip())
    return r


def chains(ip: str) -> tuple[str, str]:
    """Per-container iptables chains: nat redirect and DOCKER-USER filter."""
    n = int(ipaddress.IPv4Address(ip))
    return f"RVG_{n:08x}", f"RVGF_{n:08x}"


def container_for_ip(ip: str) -> tuple[str, str]:
    ids = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, check=True).stdout.split()
    data = json.loads(subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True, check=True).stdout) if ids else []
    hits = [(c["Name"].lstrip("/"), net["MacAddress"]) for c in data
            for net in c["NetworkSettings"]["Networks"].values() if net.get("IPAddress") == ip]
    if len(hits) != 1 or not re.fullmatch(r"[0-9a-f]{2}(:[0-9a-f]{2}){5}", hits[0][1].lower()):
        raise GuardError(f"cannot identify exactly one container for {ip}")
    return hits[0]


def _drop_jumps(table: list[str], chain: str, target: str) -> None:
    for line in _ipt(table + ["-S", chain], check=False).stdout.splitlines():
        if line.startswith("-A ") and line.endswith("-j " + target):
            _ipt(table + ["-D"] + line.split()[1:], check=False)


def disable(ip: str) -> None:
    cfg = config()
    nat, flt = chains(ip)
    _drop_jumps(["-t", "nat"], "PREROUTING", nat)
    _ipt(["-t", "nat", "-F", nat], check=False)
    _ipt(["-t", "nat", "-X", nat], check=False)
    _drop_jumps([], "DOCKER-USER", flt)
    _ipt(["-F", flt], check=False)
    _ipt(["-X", flt], check=False)
    reg = cfg["registry"] / f"{ip}.json"
    if reg.exists():
        done = cfg["registry"].parent / "done"
        done.mkdir(parents=True, exist_ok=True)
        reg.rename(done / f"{ip}-{int(time.time())}.json")


def enable(ip: str, mac: str, entry: dict) -> None:
    cfg = config()
    disable(ip)
    registry = cfg["registry"]
    registry.mkdir(parents=True, exist_ok=True)
    tmp = registry / f".{ip}.json.tmp"
    tmp.write_text(json.dumps({**entry, "mac": mac}, sort_keys=True))
    tmp.rename(registry / f"{ip}.json")
    direct = PRIVATE + tuple(cfg["direct_cidrs"])
    nat, flt = chains(ip)
    _ipt(["-t", "nat", "-N", nat])
    for cidr in direct:
        _ipt(["-t", "nat", "-A", nat, "-d", cidr, "-j", "RETURN"])
    _ipt(["-t", "nat", "-A", nat, "-p", "tcp", "-j", "REDIRECT", "--to-ports", str(cfg["port"])])
    _ipt(["-t", "nat", "-I", "PREROUTING", "1", "-s", ip, "-m", "mac", "--mac-source", mac, "-j", nat])
    _ipt(["-N", flt])
    for cidr in direct:
        _ipt(["-A", flt, "-d", cidr, "-j", "RETURN"])
    _ipt(["-A", flt, "-p", "udp", "--dport", "443", "-j", "DROP"])
    _ipt(["-I", "DOCKER-USER", "1", "-s", ip, "-m", "mac", "--mac-source", mac, "-j", flt])
    subprocess.run(["conntrack", "-D", "-s", ip], capture_output=True)


def proxy_healthy(cfg: dict) -> bool:
    """The configured systemd unit is active, or (without ``service``) the proxy port accepts."""
    if cfg.get("service"):
        r = subprocess.run(["systemctl", "is-active", cfg["service"]], capture_output=True, text=True)
        return r.stdout.strip() == "active"
    try:
        socket.create_connection(("127.0.0.1", int(cfg["port"])), timeout=5).close()
        return True
    except OSError:
        return False


def _ca(cfg: dict) -> bytes:
    if cfg["ca_cert"] is None:
        raise GuardError("guard configuration needs 'ca_cert'")
    ca = cfg["ca_cert"].read_bytes()
    if cfg.get("ca_sha256") and hashlib.sha256(ca).hexdigest() != cfg["ca_sha256"]:
        raise GuardError("proxy CA certificate differs from ca_sha256")
    return ca


def install_ca(guest, sudo) -> list[str]:
    """Trust the proxy CA in the guest (idempotent). Called before task setup, so browsers started
    by the setup already see it, and again at deploy."""
    cfg = config()
    guest.put("/tmp/rv-answer-proxy.crt", _ca(cfg), mode=0o644)
    guest.command("rm -rf /tmp/rv-nsstools && mkdir -p /tmp/rv-nsstools")
    for name, digest in sorted(cfg["nss_debs"].items()):
        if cfg["nss_deb_dir"] is None:
            raise GuardError("guard configuration lists nss_debs but no nss_deb_dir")
        data = (cfg["nss_deb_dir"] / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise GuardError("NSS deb differs from its pinned hash: " + name)
        guest.put("/tmp/rv-nsstools/" + name, data, mode=0o644)
    out = guest.command(sudo("bash -c " + shlex.quote(INSTALL)), timeout=900)
    lines = out.strip().splitlines()[-12:]
    if "system_store=ok" not in lines or not any(l.startswith("nss=/") for l in lines):
        raise GuardError("proxy CA not trusted in guest: " + " ".join(lines)[-300:])
    return lines


def task_terms(task: str) -> list[str]:
    """Identifiers of a task that must not appear in fetched URLs or responses.

    ALE task ids (``domain/task``) also occur as ``domain__task`` in public copies of the task
    data. OSWorld 2.0 and WeaveBench ids are short or generic, so they are not used as terms and
    the URL rules alone apply.
    """
    if "/" not in task:
        return []
    return [task, task.replace("/", "__")]


def harden(guest, sudo, *, task: str = "") -> dict:
    """Enable the guard for one case (after rv setup, before the agent). Raises on any failure."""
    cfg = config()
    raw = os.environ.get("RV_GUARD_TERMS")
    terms = json.loads(raw) if raw else task_terms(task)
    if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
        raise GuardError("RV_GUARD_TERMS must be a JSON list of strings")
    log = os.environ.get("RV_GUARD_LOG") or str(
        cfg["log_dir"] / f"{task or 'case'}-{guest.host}-{int(time.time())}.jsonl")
    _ca(cfg)
    task_allow = cfg["task_allow"]
    if task_allow is not None and not task_allow.is_file():
        raise GuardError(f"task_allow file missing: {task_allow}")
    if not proxy_healthy(cfg):
        raise GuardError("guard proxy is not running")
    install = "\n".join(install_ca(guest, sudo))
    purge = guest.command(sudo("bash -c " + shlex.quote(PURGE)), timeout=300).strip().splitlines()[-1:]
    if not purge or not purge[0].startswith("browser_cache_purged="):
        raise GuardError("browser cache purge failed")
    name, mac = container_for_ip(guest.host)
    entry = {"log": log, "terms": terms, "attempt": str(Path(log).parent), "container": name,
             "task": task, "unit": "", "at": time.time()}
    enable(guest.host, mac, entry)
    try:
        check = guest.command(sudo("bash -c " + shlex.quote(CHECK)), timeout=120)
    except Exception:
        disable(guest.host)
        raise
    report = {"schema": 1, "rules_version": cfg["rules_version"],
              "addon_sha256": _sha256(Path(guard_addon.__file__)),
              "task_allow_sha256": _sha256(task_allow) if task_allow is not None else None,
              "ca_sha256": _sha256(cfg["ca_cert"]), "container": name, "ip": guest.host, "mac": mac,
              "log": log, "terms": terms, "install": install.strip().splitlines()[-12:], "browser_cache": purge[0],
              "self_check": check.strip(), "enabled_at": time.time()}
    if "blocked_rule=osworld_source" not in check or "github=200" not in check or "system_store=ok" not in install:
        disable(guest.host)
        raise GuardError("network guard self-check failed: " + check.strip()[-200:])
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    Path(log + ".guard.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    guest.guard_log = log
    return report


def unharden(guest) -> None:
    """Remove the redirect of this case (after the agent is quiescent)."""
    disable(guest.host)
    log = getattr(guest, "guard_log", None)
    if log:
        with open(log + ".guard.json", "a") as stream:
            stream.write(json.dumps({"disabled_at": time.time()}) + "\n")
