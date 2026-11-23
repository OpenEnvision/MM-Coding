"""Bake the WeaveBench VM image used by the runner: the published image + openssh-server + your SSH key.

    $RV_DATA/images/weavebench/Ubuntu.qcow2  ->  $RV_DATA/images/weavebench/Ubuntu_rv.qcow2

Nothing else changes (the rv sandbox is deployed per task). The guest SSH key is the public half of
$RV_SSH_KEY (default ~/.ssh/mm_coding_vm_ed25519), authorized for ``user``; the result therefore
differs per key, and its sha256 is written next to it. Every harness boots the same baked image.

usage (OSWorld-V2 venv; starts one VM container):
  cd $RV_OSWORLD && uv run --no-sync python $RV_REPO/docker/bake_weavebench_image.py
"""
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

import requests

from mmcoding import paths

paths.use_upstream("osworld")
from desktop_env.desktop_env import DesktopEnv  # noqa: E402

from mmcoding.host.deploy import configure_docker_vm  # noqa: E402

BASE = str(paths.WEAVEBENCH_BASE_IMAGE)
OUT = paths.WEAVEBENCH_IMAGE
packages = ["openssh-server"]
KEY = Path(str(paths.SSH_KEY) + ".pub").read_text().strip()
overlay = OUT.with_name(OUT.name + ".overlay")

configure_docker_vm("8G", "4")
env = DesktopEnv(provider_name="docker", os_type="Ubuntu", headless=True, require_a11y_tree=False,
                 path_to_vm=BASE, client_password="password", screen_size=(1920, 1080))
c = env.provider.container
c.reload()
ip = next(iter(c.attrs["NetworkSettings"]["Networks"].values()))["IPAddress"]
pkgs = " ".join(packages)
inner = ("export DEBIAN_FRONTEND=noninteractive; "
         "systemctl stop packagekit unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null; "
         "mkdir -p /tmp/aptoff; mv /etc/apt/sources.list.d/google-chrome*.list /tmp/aptoff/ 2>/dev/null; "
         "ok=0; for i in $(seq 1 30); do apt-get -o DPkg::Lock::Timeout=300 update -qq && ok=1 && break; sleep 10; done; "
         f"[ $ok = 1 ] && apt-get -o DPkg::Lock::Timeout=600 install -y -qq --no-install-recommends {pkgs} "
         "&& systemctl enable ssh; rc=$?; "
         "mv /tmp/aptoff/*.list /etc/apt/sources.list.d/ 2>/dev/null; apt-get clean; exit $rc")
b64 = base64.b64encode(inner.encode()).decode()
checks = " && ".join(f"dpkg -s {p} | grep -q 'Status: install ok installed'" for p in packages)
script = (f"echo {b64} | base64 -d > /tmp/rv_bake.sh && echo password | sudo -S -p '' bash /tmp/rv_bake.sh "
          f"&& {checks} && systemctl is-enabled ssh && "
          f"umask 077 && mkdir -p ~/.ssh && echo '{KEY}' > ~/.ssh/authorized_keys && rm -f /tmp/rv_bake.sh && echo BAKE_OK")
r = requests.post(f"http://{ip}:5000/execute", json={"command": ["bash", "-lc", script], "shell": False,
                                                     "timeout": 900}, timeout=960).json()
print(r.get("output", "")[-800:], r.get("error", "")[-800:])
assert "BAKE_OK" in r.get("output", ""), r
try:
    requests.post(f"http://{ip}:5000/execute", json={"command": ["bash", "-lc",
                  "sync; echo password | sudo -S -p '' systemctl poweroff"], "shell": False}, timeout=30)
except Exception:  # noqa: BLE001 - the connection drops while powering off
    pass
for _ in range(120):
    c.reload()
    if c.status != "running":
        break
    time.sleep(5)
print("container status", c.status)
if c.status == "running":
    c.stop(timeout=60)
subprocess.run(["docker", "cp", f"{c.name}:/boot.qcow2", str(overlay)], check=True)
info = subprocess.run(["qemu-img", "info", "--output=json", str(overlay)], capture_output=True,
                      text=True, check=True).stdout
print("overlay backing:", json.loads(info).get("backing-filename"))
subprocess.run(["qemu-img", "rebase", "-u", "-b", BASE, "-F", "qcow2", str(overlay)], check=True)
subprocess.run(["qemu-img", "convert", "-p", "-O", "qcow2", str(overlay), str(OUT)], check=True)
overlay.unlink()
c.remove(force=True)
digest = hashlib.sha256()
with open(OUT, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        digest.update(block)
rel = OUT.relative_to(paths.IMAGES)
OUT.with_name(OUT.name + ".sha256").write_text(f"{digest.hexdigest()}  {rel}\n")
print("baked", OUT, OUT.stat().st_size, digest.hexdigest())
