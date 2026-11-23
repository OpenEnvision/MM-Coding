#!/usr/bin/env python3
"""Download the sandbox assets installed in every guest and verify them.

The guest installs its Python libraries offline from a wheelhouse and the audit daemon from two
Ubuntu 22.04 packages (``mmcoding/guest/rv_run.py``). The exact files and their sha256 are listed
in ``configs/sandbox/assets.sha256``; this script fetches each one (wheels from PyPI, packages
from the Ubuntu archive) into ``$RV_ASSET_DIR`` (default ``data/assets/sandbox``) and fails on any
mismatch. A few pure-Python projects publish only source archives; their wheels are built here
with ``pip wheel`` (same name and version, but the archive bytes and therefore the sha256 differ
from the listed ones, which is reported). Files already present with the listed hash are kept.

usage: python3 scripts/fetch_sandbox_assets.py [--dest DIR] [--check]
  PYPI_JSON_URL   default https://pypi.org/pypi
  UBUNTU_MIRROR   default http://archive.ubuntu.com/ubuntu
  PIP             command used to build wheels from source archives (default: python3 -m pip)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "configs" / "sandbox" / "assets.sha256"
PYPI = os.getenv("PYPI_JSON_URL", "https://pypi.org/pypi").rstrip("/")
UBUNTU = os.getenv("UBUNTU_MIRROR", "http://archive.ubuntu.com/ubuntu").rstrip("/")
DEB_POOL = {"auditd": "main/a/audit", "libauparse0": "main/a/audit"}


def entries() -> list[tuple[str, str]]:
    """(sha256, relative path) pairs of the lock file."""
    rows = []
    for line in LOCK.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, rel = line.split(maxsplit=1)
            rows.append((digest, rel.strip()))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wheel_url(filename: str) -> str | None:
    """Download URL of a published wheel; None when the release has no such file."""
    name, version = filename.split("-")[:2]
    with urllib.request.urlopen(f"{PYPI}/{name}/{version}/json", timeout=60) as response:
        release = json.load(response)
    return next((item["url"] for item in release["urls"] if item["filename"] == filename), None)


def build_wheel(filename: str, target: Path) -> None:
    """Build ``filename`` from the project's source archive (pure-Python projects without wheels)."""
    name, version = filename.split("-")[:2]
    pip = shlex.split(os.getenv("PIP", f"{sys.executable} -m pip"))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([*pip, "wheel", "--no-deps", "--no-binary", ":all:", "-w", tmp, f"{name}=={version}"],
                       check=True, stdout=subprocess.DEVNULL)
        built = list(Path(tmp).glob("*.whl"))
        if len(built) != 1:
            raise RuntimeError(f"building {name} {version} produced {[p.name for p in built]}")
        built[0].replace(target)


def deb_url(filename: str) -> str:
    return f"{UBUNTU}/pool/{DEB_POOL[filename.split('_')[0]]}/{filename}"


def fetch(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    tmp.replace(target)


def main(argv=None) -> int:
    default = os.getenv("RV_ASSET_DIR") or str(Path(os.getenv("RV_DATA") or REPO / "data") / "assets" / "sandbox")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", type=Path, default=Path(default))
    p.add_argument("--check", action="store_true", help="only verify the files already present")
    args = p.parse_args(argv)
    bad = rebuilt = 0
    for digest, rel in entries():
        target = args.dest / rel
        if target.is_file() and sha256(target) == digest:
            continue
        if args.check:
            if target.is_file():
                print(f"differs from the listed sha256 (built from source?): {rel}", file=sys.stderr)
                rebuilt += 1
            else:
                print(f"missing: {rel}", file=sys.stderr)
                bad += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = wheel_url(target.name) if rel.startswith("wheelhouse/") else deb_url(target.name)
        if url is None:
            print(f"build {target.name} from its source archive")
            build_wheel(target.name, target)
            rebuilt += 1
            continue
        print(f"fetch {url}")
        fetch(url, target)
        if sha256(target) != digest:
            print(f"sha256 mismatch: {rel}", file=sys.stderr)
            target.unlink()
            bad += 1
    total = len(entries())
    print(f"{total - bad - rebuilt} of {total} assets match the listed sha256 in {args.dest}"
          + (f"; {rebuilt} built from source archives" if rebuilt else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
