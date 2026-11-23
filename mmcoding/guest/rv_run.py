#!/usr/bin/env python3
"""Root-side helper of the harnesses (guest side).

Installed as /usr/local/sbin/rv-run (root-owned) by ``setup`` and invoked by the
host through ``sudo``. Subcommands:

* ``setup``   one-time per VM: audit, preinstalled libraries, agent network
              namespace + firewall, D-Bus filter proxy, desktop broker (gui, os_helix, mm_coding),
              service daemon, DevTools forwarders, task-service relays (the task's
              localhost services become reachable from the agent namespace), and a
              mask over the image's one corrupted directory entry.
* ``worker``  start the DSH guest worker as a sandboxed transient system unit.
* ``service`` start a persistent sandboxed service + localhost forwarder
              (used by the ``rvd`` daemon on behalf of ``rv-serve``).
* ``stop``    stop a worker unit.
* ``collect`` gather audit/journal/broker evidence into the evidence directory.
* ``daemon``  the ``rvd`` service-request daemon (root).
* ``probe``   run one command inside the agent sandbox (isolation tests).

The agent sandbox (every harness, identical): runs as ``user`` inside a private user
namespace (no access to other processes' /proc roots), with no_new_privs (no
sudo/setuid), in the ``rv`` network namespace (no abstract X11/ibus sockets;
internet through NAT; local control ports firewalled except Chrome DevTools),
with the ICE socket directory empty, the Xauthority/AT-SPI/systemd/pipewire
endpoints inaccessible, and the session bus replaced by a filtering proxy. The
X11 socket directory is empty too, except in the X harnesses (os_helix / mm_coding /
vision_tool / cli), whose code may use the display (DISPLAY=:0, admitted by the server's
localuser rule; an abstract-socket relay serves AppArmor-confined applications).
DNS from the namespace is limited to the configured gateway and public resolver.
Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import socketserver
import subprocess
import sys
import time
from pathlib import Path

USER, UID = "user", 1000
GUEST_ROOT = "/home/user/.local/share/osworld-dsh"
RV_DIR = f"{GUEST_ROOT}/rv"
ASSETS = f"{GUEST_ROOT}/rv-assets"
EVIDENCE = f"{GUEST_ROOT}/rv-evidence"
RUN = "/run/rv"  # user-owned runtime sockets (desktop broker, bus proxy)
RVD = "/run/rvd"  # root-owned service-request socket
STATE = "/var/lib/rv"  # root-only state
NETNS, HOST_IF, NS_IF = "rv", "rvh", "rvn"
HOST_IP, NS_IP, NET = "10.250.0.1", "10.250.0.2", "10.250.0.0/30"
INSTALLED = "/usr/local/sbin/rv-run"
SERVE_CLIENT = "/usr/local/bin/rv-serve"
CDP_PORTS = (9222, 1337)
# Copies of mmcoding/harnesses.py (this file runs inside the VM without the package; tests check they agree).
HARNESSES = ("gui", "os_helix", "mm_coding", "vision_tool", "cli")
X_HARNESSES = ("os_helix", "mm_coding", "vision_tool", "cli")
BROKER = {"gui": "act", "os_helix": "act_guarded", "mm_coding": "observe"}
BROKER_PROFILES = ("act", "act_guarded", "observe")
PUBLIC_DNS = "8.8.8.8"
# Never reachable from the agent namespace (20.20.20.0/24 is the VM-side network of the
# QEMU-in-Docker provider image).
PRIVATE_NETS = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10",
                "20.20.20.0/24", "127.0.0.0/8")
RESERVED_PORTS = {22, 53, 631, 1337, 5000, 5900, 8006, 8010, 8080, 9222}
# Never relayed into the agent namespace: control endpoints above, X11 / VNC / RDP / sound servers.
NO_RELAY_RANGES = ((6000, 6063), (5900, 5910), (3389, 3389), (4713, 4713))
TP_DIR = "/run/rvtp"  # root-owned; unix sockets of the task-service relays
# Directories masked (read-only empty tmpfs) for everyone at setup: the image has one inode with
# a bad checksum below this directory; touching it makes ext4 remount / read-only mid-task.
FS_MASKS = ("/usr/share/code/resources/app/extensions/node_modules/typescript/lib/pl",)
BUS_TALK = (
    "ca.desrt.dconf",  # gsettings / dconf writes
    "org.gtk.vfs.*",  # gio / gvfs file operations
    "org.gnome.evolution.dataserver.*",  # calendar / contacts data
    "org.freedesktop.Notifications",
    "org.mpris.MediaPlayer2.*",  # media player control API
)
# Single-instance GTK applications (GApplication) must own their application id
# to start; only meaningful in the X harnesses, where sandboxed code can open windows.
# Exact ids only: owning a name also grants talking to it, and a wildcard such as
# org.gnome.* would expose org.gnome.Shell / Mutter.RemoteDesktop / Terminal.
BUS_OWN = ("org.gnome.TextEditor", "org.gnome.gedit", "org.gnome.eog")
HIDDEN_ALWAYS = (
    "/run/user/1000/gdm",  # Xauthority
    "/run/user/1000/at-spi",  # accessibility bus
    "/run/user/1000/systemd",  # user manager private socket (systemd-run --user)
    "/run/user/1000/pipewire-0",
    "/run/user/1000/snapd-session-agent.socket",
    "/run/user/1000/pk-debconf-socket",
    "/run/user/1000/ICEauthority",
    "/run/user/1000/gnome-shell",
    STATE,
)
READ_ONLY = ("/home/user/server", f"{GUEST_ROOT}/code", RV_DIR, f"{GUEST_ROOT}/venv", ASSETS)
AUDIT_RULES = """-D
-b 32768
--backlog_wait_time 60000
-a always,exit -F arch=b64 -S execve,execveat -k rv_exec
-a always,exit -F arch=b32 -S execve -k rv_exec
-w /home/user/server -p wa -k rv_eval
-w /home/user/.local/share/osworld-dsh/code -p wa -k rv_harness
-w /home/user/.local/share/osworld-dsh/rv -p wa -k rv_harness
"""


def sh(cmd: list[str] | str, check: bool = True, timeout: float = 300, **kw) -> subprocess.CompletedProcess:
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)[:200]} -> {proc.returncode}: {proc.stderr.strip()[-600:]}")
    return proc


def audit_mark(text: str) -> None:
    sh(["auditctl", "-m", text], check=False)


# ------------------------------------------------------------------- sandbox


def sandbox_props(hidden: list[str] | tuple[str, ...] = (), harness: str = "gui") -> list[str]:
    """systemd unit properties shared by the worker and agent-started services."""
    x_access = harness in X_HARNESSES
    inaccessible = " ".join("-" + p for p in (*HIDDEN_ALWAYS, *hidden))
    read_only = " ".join("-" + p for p in READ_ONLY)
    env = {
        "HOME": "/home/user",
        "USER": USER,
        "LOGNAME": USER,
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "PATH": "/home/user/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LANG": "en_US.UTF-8",
        "SHELL": "/bin/bash",
    }
    if x_access:
        env["DISPLAY"] = ":0"
    # Optional network guard (host/guard.py): when its CA is installed, TLS clients in the sandbox trust it.
    if Path("/usr/local/share/ca-certificates/rv-answer-proxy.crt").is_file():
        bundle = "/etc/ssl/certs/ca-certificates.crt"
        env.update(SSL_CERT_FILE=bundle, REQUESTS_CA_BUNDLE=bundle, PIP_CERT=bundle,
                   NODE_EXTRA_CA_CERTS="/usr/local/share/ca-certificates/rv-answer-proxy.crt")
    return [
        f"User={USER}",
        f"Group={USER}",
        "PrivateUsers=yes",
        "NoNewPrivileges=yes",
        f"NetworkNamespacePath=/run/netns/{NETNS}",
        "TemporaryFileSystem=/tmp/.ICE-unix:ro" if x_access else "TemporaryFileSystem=/tmp/.X11-unix:ro /tmp/.ICE-unix:ro",
        f"InaccessiblePaths={inaccessible}",
        f"BindPaths={RUN}/bus:/run/user/1000/bus",
        f"BindReadOnlyPaths=/etc/netns/{NETNS}/resolv.conf:/etc/resolv.conf",
        f"ReadOnlyPaths={read_only}",
        "WorkingDirectory=/home/user",
        "KillMode=control-group",
        "TimeoutStopSec=20",
        "Environment=" + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()),
    ]


def systemd_run(unit: str, props: list[str], argv: list[str], extra: list[str] = ()) -> None:
    cmd = ["systemd-run", f"--unit={unit}", "--quiet", "--collect", *extra]
    for p in props:
        cmd += ["-p", p]
    sh(cmd + ["--", *argv])


def unit_active(unit: str) -> bool:
    return sh(["systemctl", "is-active", unit], check=False).stdout.strip() in {"active", "activating"}


def load_state() -> dict:
    path = Path(STATE) / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(state: dict) -> None:
    Path(STATE).mkdir(mode=0o700, exist_ok=True)
    (Path(STATE) / "state.json").write_text(json.dumps(state, indent=1))


# --------------------------------------------------------------------- setup


def dpkg_install(debs: list[Path], wait: float = 600, poll: float = 10) -> None:
    """dpkg -i, waiting out a task warmup that is still holding the dpkg lock (apt in the background)."""
    cmd = "DEBIAN_FRONTEND=noninteractive dpkg -i " + " ".join(shlex.quote(str(p)) for p in debs)
    deadline = time.monotonic() + wait
    while True:
        proc = sh(cmd, check=False, timeout=600)
        if proc.returncode == 0:
            return
        if "lock" not in proc.stderr or time.monotonic() >= deadline:
            raise RuntimeError(f"dpkg -i -> {proc.returncode}: {proc.stderr.strip()[-600:]}")
        time.sleep(poll)


def setup_audit(debs: Path) -> dict:
    if shutil.which("auditctl") is None:
        dpkg_install(sorted(debs.glob("*.deb")))
    conf = Path("/etc/audit/auditd.conf")
    text = conf.read_text()
    for key, value in (("max_log_file", "2048"), ("max_log_file_action", "IGNORE"), ("num_logs", "2"),
                       ("space_left", "200"), ("admin_space_left", "100"), ("disk_full_action", "SUSPEND")):
        text = re.sub(rf"^{key}\s*=.*$", f"{key} = {value}", text, flags=re.M)
    conf.write_text(text)
    Path("/etc/audit/rules.d/rv.rules").write_text(AUDIT_RULES)
    sh("systemctl restart auditd && sleep 1")
    sh(["auditctl", "-R", "/etc/audit/rules.d/rv.rules"])
    rules = sh(["auditctl", "-l"]).stdout
    if "rv_exec" not in rules:
        raise RuntimeError("audit rules not loaded")
    return {"auditd": sh(["systemctl", "is-active", "auditd"], check=False).stdout.strip(), "rules": rules.count("\n")}


def setup_libraries(assets: Path) -> dict:
    marker = Path(STATE) / "libs.done"
    if not marker.exists():
        sh(["pip3", "install", "--no-index", "--find-links", str(assets / "wheelhouse"),
            "-r", str(assets / "requirements.txt"), "-c", str(assets / "constraints.txt")], timeout=900)
        Path(STATE).mkdir(mode=0o700, exist_ok=True)
        marker.write_text(time.strftime("%FT%T"))
    probe = sh(["python3", "-c", "import docx, openpyxl, pptx, pandas, xlrd, odf, pypdf, pdfplumber, fitz, "
                "websocket, cv2, Xlib, mss; print('ok')"], check=False)
    return {"libraries": probe.stdout.strip() or probe.stderr.strip()[-300:]}


def uplink() -> str:
    out = sh(["ip", "-4", "route", "show", "default"]).stdout.split()
    return out[out.index("dev") + 1]


def dns_rules(resolvers) -> list[list[str]]:
    return [["-d", ip, "-p", proto, "--dport", "53", "-j", "ACCEPT"]
            for ip in dict.fromkeys(resolvers) for proto in ("udp", "tcp")]


def forward_rules(resolvers) -> list[list[str]]:
    # Only these resolver addresses override the private-network prohibition.
    return (dns_rules(resolvers) + [["-d", cidr, "-j", "DROP"] for cidr in PRIVATE_NETS]
            + [["-j", "ACCEPT"]])


def setup_network() -> dict:
    if not Path(f"/run/netns/{NETNS}").exists():
        sh(["ip", "netns", "add", NETNS])
    if sh(["ip", "link", "show", HOST_IF], check=False).returncode != 0:
        sh(["ip", "link", "add", HOST_IF, "type", "veth", "peer", "name", NS_IF])
        sh(["ip", "link", "set", NS_IF, "netns", NETNS])
        sh(["ip", "addr", "add", f"{HOST_IP}/30", "dev", HOST_IF])
        sh(["ip", "link", "set", HOST_IF, "up"])
        ns = ["ip", "netns", "exec", NETNS]
        sh(ns + ["ip", "addr", "add", f"{NS_IP}/30", "dev", NS_IF])
        sh(ns + ["ip", "link", "set", NS_IF, "up"])
        sh(ns + ["ip", "link", "set", "lo", "up"])
        sh(ns + ["ip", "route", "add", "default", "via", HOST_IP])
    ns = ["ip", "netns", "exec", NETNS]
    sh(["modprobe", "tcp_bbr"], check=False)
    sh(["sysctl", "-qw", "net.ipv4.ip_forward=1", "net.core.default_qdisc=fq"])
    sh(ns + ["sysctl", "-qw", "net.ipv4.tcp_congestion_control=bbr"])
    gateway = sh(["ip", "-4", "route", "show", "default"]).stdout.split()[2]
    resolvers = list(dict.fromkeys((PUBLIC_DNS, gateway)))
    Path(f"/etc/netns/{NETNS}").mkdir(parents=True, exist_ok=True)
    dns_config = Path("/usr/local/lib/rv-web-dns.json")
    if dns_config.is_file():
        unit = "rv-web-dns-agent"
        if not unit_active(unit):
            systemd_run(unit, [f"NetworkNamespacePath=/run/netns/{NETNS}"],
                        ["/usr/bin/python3", "/usr/local/lib/rv-web-dns.py", "--domain",
                         json.loads(dns_config.read_text())["gitlab_domain"], "--upstream", gateway])
        Path(f"/etc/netns/{NETNS}/resolv.conf").write_text("nameserver 127.0.0.1\noptions timeout:3\n")
    else:
        Path(f"/etc/netns/{NETNS}/resolv.conf").write_text(
            "".join(f"nameserver {ip}\n" for ip in resolvers) + "options timeout:2\n")
    dev = uplink()
    ipt = ["iptables", "-w"]
    for chain in ("RV_IN", "RV_FWD"):
        sh(ipt + ["-N", chain], check=False)
        sh(ipt + ["-F", chain])
    # Inbound from the agent namespace to this host: only Chrome DevTools.
    for rule in (["-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
                 ["-p", "tcp", "--dport", "9222", "-j", "ACCEPT"],
                 ["-j", "DROP"]):
        sh(ipt + ["-A", "RV_IN", *rule])
    for rule in forward_rules(resolvers):
        sh(ipt + ["-A", "RV_FWD", *rule])
    for spec in ((["INPUT", "-i", HOST_IF, "-j", "RV_IN"]),
                 (["FORWARD", "-i", HOST_IF, "-j", "RV_FWD"]),
                 (["FORWARD", "-o", HOST_IF, "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"])):
        if sh(ipt + ["-C", *spec], check=False).returncode != 0:
            sh(ipt + ["-I", spec[0], "1", *spec[1:]])
    nat = ["-s", NET, "-o", dev, "-j", "MASQUERADE"]
    if sh(ipt + ["-t", "nat", "-C", "POSTROUTING", *nat], check=False).returncode != 0:
        sh(ipt + ["-t", "nat", "-A", "POSTROUTING", *nat])
    # DevTools inside the namespace: localhost:9222 and :1337 reach the host's 9222 relay.
    for port in CDP_PORTS:
        unit = f"rv-cdp-{port}"
        if not unit_active(unit):
            systemd_run(unit, [f"NetworkNamespacePath=/run/netns/{NETNS}", "DynamicUser=yes"],
                        ["/usr/bin/socat", f"TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork", f"TCP:{HOST_IP}:9222"])
    cc = sh(ns + ["cat", "/proc/sys/net/ipv4/tcp_congestion_control"]).stdout.strip()
    return {"netns": NETNS, "uplink": dev, "gateway": gateway, "ns_congestion_control": cc,
            "dns_resolvers": resolvers}


def relay_candidates(ss_output: str, exclude: set[int] = frozenset()) -> dict[int, str]:
    """Task-service TCP ports listening in the root namespace -> address to connect to.

    Reserved control ports, X11 / VNC / RDP / sound servers and ``exclude`` (forwarders of
    agent services) are skipped. Wildcard binds are reached through 127.0.0.1.
    """
    ports: dict[int, str] = {}
    for line in ss_output.splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        addr, _, port_s = cols[3].rpartition(":")
        if not port_s.isdigit():
            continue
        port = int(port_s)
        addr = addr.strip("[]").split("%")[0]
        if (port in RESERVED_PORTS or port in CDP_PORTS or port in exclude or port >= 32768
                or any(lo <= port <= hi for lo, hi in NO_RELAY_RANGES)):
            continue
        if addr in ("0.0.0.0", "*", "::", "::ffff:0.0.0.0") or addr.startswith("127."):
            target = "127.0.0.1"
        elif addr == "::1":
            target = "[::1]"
        else:
            continue  # bound to one non-loopback interface only: not a local desktop service
        if ports.get(port) != "127.0.0.1":
            ports[port] = target
    return ports


def agent_service_ports() -> set[int]:
    path = Path(EVIDENCE) / "services.jsonl"
    ports = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                ports.add(int(json.loads(line).get("port") or 0))
            except ValueError:
                continue
    return ports - {0}


def setup_task_relays() -> dict:
    """Make the task's local services (e.g. a website on localhost:4000) reachable from the
    agent namespace at the same localhost port, as on an ordinary desktop.

    Root side: socat UNIX-LISTEN (/run/rvtp/<port>.sock) -> TCP <loopback>:<port>; namespace
    side: socat TCP-LISTEN 127.0.0.1:<port> -> that socket. Unix sockets cross network
    namespaces and bind no TCP port in the root namespace, so the evaluator can still restart
    the service on its port. Idempotent: newly listening ports are added on every call.
    """
    os.makedirs(TP_DIR, exist_ok=True)
    os.chmod(TP_DIR, 0o755)
    ss = sh(["ss", "-Hltn"], check=False).stdout
    ports = relay_candidates(ss, agent_service_ports())
    for port, target in sorted(ports.items()):
        sock = f"{TP_DIR}/{port}.sock"
        host_unit, ns_unit = f"rv-tp-{port}-host", f"rv-tp-{port}-ns"
        if not unit_active(host_unit):
            proto = "TCP6" if target.startswith("[") else "TCP4"
            systemd_run(host_unit, [], ["/usr/bin/socat", f"UNIX-LISTEN:{sock},fork,mode=0666,unlink-early",
                                        f"{proto}:{target}:{port}"])
        if not unit_active(ns_unit):
            props = [f"NetworkNamespacePath=/run/netns/{NETNS}", "DynamicUser=yes"]
            if port < 1024:
                props.append("AmbientCapabilities=CAP_NET_BIND_SERVICE")
            for _ in range(30):
                if os.path.exists(sock):
                    break
                time.sleep(0.1)
            systemd_run(ns_unit, props, ["/usr/bin/socat", f"TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork",
                                         f"UNIX-CONNECT:{sock}"])
    state = load_state()
    known = sorted(set(state.get("task_ports", [])) | set(ports))
    state["task_ports"] = known
    save_state(state)
    return {"task_ports": known}


def setup_fs_guard() -> dict:
    masked = []
    mounts = Path("/proc/mounts").read_text()
    for path in FS_MASKS:
        if not os.path.isdir(path):
            continue
        if f" {path} " not in mounts:
            sh(["mount", "-t", "tmpfs", "-o", "ro,mode=0755,size=4k", "rv-mask", path])
        masked.append(path)
    return {"fs_masks": masked}


def setup_bus_proxy() -> dict:
    unit = "rv-dbusproxy"
    if not unit_active(unit):
        if os.path.exists(f"{RUN}/bus"):
            os.unlink(f"{RUN}/bus")
        argv = ["/usr/bin/xdg-dbus-proxy", "unix:path=/run/user/1000/bus", f"{RUN}/bus", "--filter", "--log"]
        argv += [f"--talk={name}" for name in BUS_TALK]
        argv += [f"--own={name}" for name in BUS_OWN]
        systemd_run(unit, [f"User={USER}", f"Group={USER}",
                           "Environment=XDG_RUNTIME_DIR=/run/user/1000"], argv)
        for _ in range(50):
            if os.path.exists(f"{RUN}/bus"):
                break
            time.sleep(0.1)
    if not os.path.exists(f"{RUN}/bus"):
        raise RuntimeError("D-Bus filter proxy socket missing")
    return {"bus_proxy": "filtered", "talk": list(BUS_TALK), "own": list(BUS_OWN)}


def setup_desktop(harness: str, profile: str | None = "") -> dict:
    """Broker tool profile: act / act_guarded = all actions, observe = screenshot only.
    ``profile`` "" = the harness's default, None = no broker."""
    profile = BROKER.get(harness) if profile == "" else profile
    if profile is None:
        sh(["systemctl", "stop", "rv-desktop"], check=False)
        if os.path.exists(f"{RUN}/desktop.sock"):
            os.unlink(f"{RUN}/desktop.sock")
        return {"desktop_broker": None}
    unit = "rv-desktop"
    sock = f"{RUN}/desktop.sock"
    if unit_active(unit) and load_state().get("broker", BROKER.get(load_state().get("harness"))) != profile:
        sh(["systemctl", "stop", unit], check=False)
        if os.path.exists(sock):
            os.unlink(sock)
    if not unit_active(unit):
        systemd_run(unit, [f"User={USER}", f"Group={USER}",
                           "Environment=DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority HOME=/home/user"],
                    ["/usr/bin/python3", f"{RV_DIR}/rv_desktop.py", "--profile", profile, "--socket", sock,
                     "--log", f"{RUN}/desktop_actions.jsonl"])
        for _ in range(100):
            if os.path.exists(sock):
                break
            time.sleep(0.1)
    if not os.path.exists(sock):
        raise RuntimeError("desktop broker socket missing")
    return {"desktop_broker": profile}


def setup_x_relay(harness: str) -> dict:
    """X harnesses: serve the abstract X socket inside the agent namespace.

    AppArmor-confined applications (evince, LibreOffice, ...) cannot connect to the
    filesystem X socket from inside the sandbox's mount namespace ("disconnected
    path"); their profiles do allow the abstract socket, which is per network
    namespace. The relay runs as ``user`` in the namespace but outside the
    sandbox's mount restrictions, so the X server's localuser rule admits it.
    """
    unit = "rv-x11"
    if harness not in X_HARNESSES:
        sh(["systemctl", "stop", unit], check=False)
        return {"x_relay": None}
    if not unit_active(unit):
        systemd_run(unit, [f"NetworkNamespacePath=/run/netns/{NETNS}", f"User={USER}", f"Group={USER}"],
                    ["/usr/bin/socat", "ABSTRACT-LISTEN:/tmp/.X11-unix/X0,fork", "UNIX-CONNECT:/tmp/.X11-unix/X0"])
        time.sleep(0.5)
    return {"x_relay": unit_active(unit)}


def setup_daemon() -> dict:
    unit = "rvd"
    if not unit_active(unit):
        systemd_run(unit, [], ["/usr/bin/python3", INSTALLED, "daemon"])
        for _ in range(50):
            if os.path.exists(f"{RVD}/ctl.sock"):
                break
            time.sleep(0.1)
    return {"rvd": os.path.exists(f"{RVD}/ctl.sock")}


def cmd_setup(args) -> None:
    started = time.time()
    if Path(INSTALLED).resolve() != Path(__file__).resolve():
        raise SystemExit("run setup through the installed copy")
    for path, mode, owner in ((RUN, 0o700, UID), (RVD, 0o755, 0), (STATE, 0o700, 0)):
        os.makedirs(path, exist_ok=True)
        os.chmod(path, mode)
        os.chown(path, owner, owner)
    os.makedirs(EVIDENCE, exist_ok=True)
    os.chown(EVIDENCE, UID, UID)
    shutil.copyfile(f"{RV_DIR}/rv_serve.py", SERVE_CLIENT)
    os.chmod(SERVE_CLIENT, 0o755)
    broker = BROKER.get(args.harness) if args.broker is None else (None if args.broker == "none" else args.broker)
    report = {"harness": args.harness, "hidden": args.hidden}
    report.update(setup_audit(Path(ASSETS) / "debs"))
    report.update(setup_libraries(Path(ASSETS)))
    report.update(setup_network())
    report.update(setup_bus_proxy())
    report.update(setup_desktop(args.harness, broker))
    report.update(setup_x_relay(args.harness))
    report.update(setup_daemon())
    report.update(setup_fs_guard())
    save_state({**report, "broker": broker, "setup_at": time.time()})
    report.update(setup_task_relays())
    report["elapsed_s"] = round(time.time() - started, 1)
    print(json.dumps(report))


# -------------------------------------------------------------------- worker


def snapshot(tag: str) -> None:
    ps = sh(["ps", "-eo", "pid,ppid,uid,etimes,comm,args", "--no-headers"], check=False).stdout
    path = Path(EVIDENCE) / f"ps_{tag}.txt"
    path.write_text(ps)
    os.chown(path, UID, UID)
    shell = sh(["pgrep", "-x", "gnome-shell"], check=False).stdout.split()
    session = {"time": time.time()}
    if shell:
        try:
            session.update(gnome_shell_pid=int(shell[0]),
                           audit_session=int(Path(f"/proc/{shell[0]}/sessionid").read_text().strip()),
                           loginuid=int(Path(f"/proc/{shell[0]}/loginuid").read_text().strip()))
        except (OSError, ValueError):
            pass
    path = Path(EVIDENCE) / f"session_{tag}.json"
    path.write_text(json.dumps(session))
    os.chown(path, UID, UID)


def cmd_worker(args) -> None:
    if not re.fullmatch(r"osworld-dsh-[A-Za-z0-9_-]+", args.unit):
        raise SystemExit("bad unit name")
    attempt = Path(args.attempt)
    if not str(attempt).startswith(GUEST_ROOT + "/attempts/") or not (attempt / "request.json").exists():
        raise SystemExit("bad attempt")
    setup_fs_guard()
    relays = setup_task_relays()  # services the task started after setup
    state = load_state()
    snapshot("agent_start")
    audit_mark(f"rv_agent_start unit={args.unit} task_ports={','.join(map(str, relays['task_ports']))}")
    props = sandbox_props(state.get("hidden", []), state.get("harness", "gui")) + [
        f"RuntimeMaxSec={int(args.runtime_max)}",
        f"StandardOutput=append:{attempt}/worker.stdout",
        f"StandardError=append:{attempt}/worker.stderr",
    ]
    try:
        systemd_run(args.unit, props, [f"{GUEST_ROOT}/venv/bin/python", f"{RV_DIR}/worker_rv.py",
                                       f"{attempt}/request.json"])
        record = {"unit": args.unit, "started": time.time(), "props": props}
        (Path(EVIDENCE) / "worker_unit.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({"unit": args.unit}))
    except BaseException:
        # A launch timeout may still have started the unit; the host can retry stop after a failure.
        stop_worker_unit(args.unit)
        raise


def cmd_probe(args) -> None:
    """Run one command inside the agent sandbox (isolation tests); prints its output."""
    state = load_state()
    cmd = ["systemd-run", "--wait", "--pipe", "--quiet", "--collect"]
    for p in sandbox_props(state.get("hidden", []), state.get("harness", "gui")):
        cmd += ["-p", p]
    proc = subprocess.run(cmd + ["--", *args.argv], text=True, capture_output=True, timeout=600)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.exit(proc.returncode)


def stop_worker_unit(unit: str) -> None:
    """Blocking stop, including the launch-timeout case; fail closed if still alive."""
    try:
        stopped = sh(["systemctl", "stop", unit], check=False, timeout=60).returncode == 0
    except subprocess.TimeoutExpired:
        stopped = False
    if not stopped:
        sh(["systemctl", "kill", "--kill-who=all", "--signal=KILL", unit], check=False)
        sh(["systemctl", "stop", unit], check=False, timeout=60)
    status = sh(["systemctl", "show", unit, "--property=ActiveState", "--value"], check=False)
    if not ((status.returncode == 0 and status.stdout.strip() in {"inactive", "failed"})
            or (status.returncode == 1 and "not found" in status.stderr.lower())
            or status.returncode == 4):
        raise RuntimeError(f"worker {unit} is not quiescent: {status.stdout.strip() or status.stderr.strip()}")


def cmd_stop(args) -> None:
    if not re.fullmatch(r"osworld-dsh-[A-Za-z0-9_-]+", args.unit):
        raise SystemExit("bad unit name")
    stop_worker_unit(args.unit)
    audit_mark(f"rv_agent_stop unit={args.unit}")
    snapshot("agent_stop")


# ------------------------------------------------------------------- services


def cmd_service(args) -> None:
    name, port = args.name, int(args.port or 0)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,30}", name):
        raise SystemExit("name must match [a-z0-9-]{1,31}")
    if port and (not 1024 <= port <= 65535 or port in RESERVED_PORTS):
        raise SystemExit(f"port {port} is not allowed")
    if port and port in load_state().get("task_ports", []):
        raise SystemExit(f"port {port} belongs to a service of the task; it is already reachable from your "
                         f"sandbox at localhost:{port}, so do not replace it")
    cwd = os.path.realpath(args.cwd)
    if not (cwd.startswith("/home/user") or cwd.startswith("/tmp")) or not os.path.isdir(cwd):
        raise SystemExit("cwd must be an existing directory under /home/user or /tmp")
    if not args.argv:
        raise SystemExit("missing command")
    listening = sh(["ss", "-Hltn", f"sport = :{port}"], check=False).stdout.strip() if port else ""
    if listening:
        raise SystemExit(f"port {port} is already in use on the desktop host")
    state = load_state()
    unit, fwd = f"rv-svc-{name}", f"rv-fwd-{name}"
    for u in (unit, fwd):
        sh(["systemctl", "stop", u], check=False)
    props = [p for p in sandbox_props(state.get("hidden", []), state.get("harness", "gui"))
             if not p.startswith("WorkingDirectory=")]
    systemd_run(unit, props + [f"WorkingDirectory={cwd}"], args.argv)
    if port:
        systemd_run(fwd, [], ["/usr/bin/socat", f"TCP-LISTEN:{port},bind=127.0.0.1,reuseaddr,fork",
                              f"TCP:{NS_IP}:{port}"])
    record = {"name": name, "port": port, "cwd": cwd, "argv": args.argv, "unit": unit,
              "forwarder": fwd if port else None, "started": time.time()}
    services = Path(EVIDENCE) / "services.jsonl"
    with services.open("a") as f:
        f.write(json.dumps(record) + "\n")
    os.chown(services, UID, UID)
    audit_mark(f"rv_service_start name={name} port={port}")
    reply = {"ok": True, "unit": unit}
    if port:
        reply.update(url=f"http://localhost:{port}/",
                     note="the service must listen on 0.0.0.0 (or 10.250.0.2) inside the sandbox")
    print(json.dumps(reply))


class _Daemon(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def cmd_daemon(_args) -> None:
    path = f"{RVD}/ctl.sock"
    os.makedirs(RVD, exist_ok=True)
    if os.path.exists(path):
        os.unlink(path)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            try:
                request = json.loads(self.rfile.readline() or b"{}")
                if request.get("op") != "serve":
                    raise ValueError("unsupported op")
                argv = request.get("argv")
                if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
                    raise ValueError("argv must be a list of strings")
                proc = sh([sys.executable, INSTALLED, "service", "--name", str(request.get("name")),
                           "--port", str(request.get("port")), "--cwd", str(request.get("cwd")), "--", *argv],
                          check=False, timeout=60)
                reply = json.loads(proc.stdout) if proc.returncode == 0 else {
                    "ok": False, "error": (proc.stderr.strip() or proc.stdout.strip())[-500:]}
            except Exception as exc:  # noqa: BLE001
                reply = {"ok": False, "error": str(exc)[:500]}
            self.wfile.write((json.dumps(reply) + "\n").encode())

    server = _Daemon(path, Handler)
    os.chmod(path, 0o666)
    server.serve_forever()


# -------------------------------------------------------------------- collect


def cmd_collect(_args) -> None:
    ev = Path(EVIDENCE)
    ev.mkdir(exist_ok=True)
    logs = sorted(Path("/var/log/audit").glob("audit.log*"), key=lambda p: p.stat().st_mtime)
    with (ev / "audit.log").open("wb") as out:
        for p in logs:
            out.write(p.read_bytes())
    units = ["rv-dbusproxy", "rv-desktop", "rvd"]
    worker = ev / "worker_unit.json"
    if worker.exists():
        units.append(json.loads(worker.read_text())["unit"])
    services = ev / "services.jsonl"
    if services.exists():
        for line in services.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                units += [u for u in (record["unit"], record.get("forwarder")) if u]
    for unit in dict.fromkeys(units):
        out = sh(["journalctl", "-u", unit, "-o", "short-unix", "--no-pager"], check=False, timeout=120).stdout
        (ev / f"journal_{unit}.log").write_text(out)
    for name in ("desktop_actions.jsonl",):
        src = Path(RUN) / name
        if src.exists():
            shutil.copyfile(src, ev / name)
    (ev / "iptables.txt").write_text(sh(["iptables-save", "-c"], check=False).stdout)
    (ev / "ip6tables.txt").write_text(sh(["ip6tables-save", "-c"], check=False).stdout)
    state = load_state()
    (ev / "network_policy.json").write_text(json.dumps({k: v for k, v in state.items()
        if k.startswith("dns_")}, indent=1))
    for p in ev.iterdir():
        os.chown(p, UID, UID)
        os.chmod(p, 0o600)
    print(json.dumps({"files": sorted(p.name for p in ev.iterdir())}))


def main(argv: list[str] | None = None) -> None:
    if os.geteuid() != 0:
        raise SystemExit("rv-run must run as root")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--harness", choices=list(HARNESSES), required=True)
    p.add_argument("--hidden", action="append", default=[], help="extra path hidden from the sandbox")
    p.add_argument("--broker", choices=[*BROKER_PROFILES, "none"], default=None,
                   help="desktop broker profile (default: the harness's)")
    p = sub.add_parser("worker")
    p.add_argument("--unit", required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--runtime-max", type=int, required=True)
    p = sub.add_parser("stop")
    p.add_argument("--unit", required=True)
    p = sub.add_parser("service")
    p.add_argument("--name", required=True)
    p.add_argument("--port", default="0")
    p.add_argument("--cwd", required=True)
    p.add_argument("argv", nargs=argparse.REMAINDER)
    sub.add_parser("collect")
    sub.add_parser("daemon")
    p = sub.add_parser("probe")
    p.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if getattr(args, "argv", None) and args.argv[0] == "--":
        args.argv = args.argv[1:]
    {"setup": cmd_setup, "worker": cmd_worker, "stop": cmd_stop, "service": cmd_service,
     "collect": cmd_collect, "daemon": cmd_daemon, "probe": cmd_probe}[args.cmd](args)


if __name__ == "__main__":
    main()
