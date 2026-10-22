# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Evaluate MT3 trajectory replay on LIBERO-PRO ``libero_10_swap``.

Tasks 0-8 use ``configs/tasks_0_8`` and task 9 uses ``configs/task_9``. Each
(task, seed) episode is one ``rpent --planner mt3`` run with a fresh
environment. With ``--scenes`` the fixture placement of every episode is
restored from a recorded snapshot, which makes episodes reproducible across
processes; without it, LIBERO's fixture sampler draws from NumPy's global RNG
and placements can differ between processes at the same seed.

Usage:
    python robots/libero/mt3/long_swap/run_long_swap.py --out runs/long_swap \
        --data /path/to/long_swap_data --seeds 10-49 --procs 40
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def parse_range(text: str) -> list[int]:
    """``"10-49"`` or ``"0,3,5"`` to a list of ints."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="unpacked long_swap_data (library/, scenes/)",
    )
    p.add_argument("--tasks", default="0-9")
    p.add_argument(
        "--seeds",
        default="10-49",
        help="official init states; the leaderboard uses 0-9",
    )
    p.add_argument(
        "--procs",
        type=int,
        default=40,
        help="parallel episodes (each uses about 1-2 CPU cores)",
    )
    p.add_argument(
        "--egl-devices", default="0", help="comma-separated GPU ids for EGL rendering"
    )
    p.add_argument(
        "--no-scenes",
        action="store_true",
        help="native initialization instead of snapshots",
    )
    p.add_argument("--timeout", type=int, default=1800, help="seconds per episode")
    args = p.parse_args()

    rpent = shutil.which("rpent")
    if rpent is None:
        raise SystemExit(
            "rpent CLI not found; install this repository with the libero extra first"
        )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = args.data.resolve()
    # The configs name the library relative to themselves (../../data/library).
    link = HERE / "data"
    if not link.exists():
        link.symlink_to(data)
    elif link.resolve() != data:
        raise SystemExit(f"{link} already points to {link.resolve()}, not {data}")
    devices = args.egl_devices.split(",")
    jobs = [(t, s) for t in parse_range(args.tasks) for s in parse_range(args.seeds)]

    def run(job: tuple[int, int]) -> dict:
        task, seed = job
        cell = out / f"libero_10_swap_t{task}_s{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        config = HERE / "configs" / ("task_9" if task == 9 else "tasks_0_8")
        cmd = [
            rpent,
            "--robot",
            "libero",
            "--planner",
            "mt3",
            "--suite",
            "libero_10_swap",
            "--task",
            str(task),
            "--seed",
            str(seed),
            "--memory-profile",
            "local",
            "--memory-dir",
            str(config),
            "--libero-type",
            "pro",
            "--max-episode-steps",
            "10000",
            "--output-dir",
            str(cell),
        ]
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO),
            "LIBERO_TYPE": "pro",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": devices[seed % len(devices)],
            "ROBOT_PLATFORM": "LIBERO",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "RPENT_MT3_FIXTURE_SEED_POLICY": "rpent",
        }
        scene = data / "scenes" / f"libero_10_swap_t{task}_s{seed}.json"
        if not args.no_scenes:
            if not scene.exists():
                raise SystemExit(f"missing scene snapshot {scene}")
            env["RPENT_MT3_SCENE_SNAPSHOT"] = str(scene)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with (cell / "cli.log").open("w") as log:
            try:
                rc = subprocess.run(
                    cmd,
                    cwd=cell,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        episode = cell / "mt3_episode.json"
        success = (
            json.loads(episode.read_text()).get("success") if episode.exists() else None
        )
        print(f"task {task} seed {seed}: {success} (rc={rc})", flush=True)
        return {"task": task, "seed": seed, "rc": rc, "success": success}

    with ThreadPoolExecutor(args.procs) as pool:
        rows = list(pool.map(run, jobs))
    per_task = {}
    for r in rows:
        done = per_task.setdefault(r["task"], [0, 0])
        done[0] += r["success"] is True
        done[1] += 1
    summary = {
        "episodes": len(rows),
        "success": sum(r["success"] is True for r in rows),
        "no_result": sum(r["success"] is None for r in rows),
        "per_task": {f"t{t}": f"{a}/{n}" for t, (a, n) in sorted(per_task.items())},
        "rows": rows,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
