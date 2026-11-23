"""Download and verify the ALE v1.1 Ubuntu qcow2 into $RV_DATA/images/ale/qemu with ALE's own provider code.

The QEMU provider also downloads the disk on first use; fetching it beforehand keeps the first run
from waiting for about 93 GB.

usage (ALE venv): cd $RV_ALE && uv run --no-sync python $RV_REPO/docker/fetch_ale_disk.py
"""
import asyncio
import time

from mmcoding import paths

paths.use_upstream("ale")
from ale_run.environments.providers.qemu import QemuProvider, _build_snapshot_config  # noqa: E402

from mmcoding.benchmarks.ale import QEMU_SNAPSHOTS  # noqa: E402

snap = _build_snapshot_config(dict(QEMU_SNAPSHOTS["cpu-free-ubuntu"]))


async def main():
    t = time.time()
    p = QemuProvider.__new__(QemuProvider)
    p._image_locks = {}
    path = await QemuProvider._fetch_hf_disk(p, snap)
    print("DISK_READY", path, round((time.time() - t) / 60, 1), "min", flush=True)

asyncio.run(main())
