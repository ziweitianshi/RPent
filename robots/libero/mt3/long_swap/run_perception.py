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

"""Run strict-perception Long Swap with one MT3 config and the v2r library."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_long_swap import parse_range

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def main() -> None:
    """Resolve portable data, run independent episodes, and save their results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tasks", default="0-9")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--procs", type=int, default=20)
    parser.add_argument("--egl-devices", default="0")
    parser.add_argument("--sam3-endpoint", default="http://127.0.0.1:8114")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--align-step", type=float, default=0.005)
    args = parser.parse_args()
    if args.procs < 1 or args.align_step <= 0:
        parser.error("--procs and --align-step must be positive")
    tasks, seeds = parse_range(args.tasks), parse_range(args.seeds)
    if not tasks or any(t not in range(10) for t in tasks) or not seeds:
        parser.error("choose tasks 0-9 and a nonempty seed range")
    jobs = list(dict.fromkeys((t, s) for t in tasks for s in seeds))
    data, out = args.data.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--out must be absent or empty to preserve earlier results")
    perception = json.loads((data / "perception.json").read_text())
    base = json.loads((HERE / "configs/strict_perception/mt3.json").read_text())
    base.update(library=str(data / "library"), align_step=args.align_step)
    rpent = shutil.which("rpent")
    if not rpent:
        parser.error("install RPent's LIBERO dependencies and activate its environment")
    devices = [v.strip() for v in args.egl_devices.split(",") if v.strip()]
    if not devices:
        parser.error("--egl-devices must contain at least one device")
    configs = {}
    for task in dict.fromkeys(tasks):
        if not (data / "library/libero_10" / f"task_{task:02d}.npz").is_file():
            parser.error(f"missing library for task {task}")
        bodies = perception[str(task)]["bodies"]
        for spec in bodies.values():
            for key in ("model", "refs"):
                if spec.get(key):
                    path = (data / spec[key]).resolve()
                    if not path.is_file():
                        parser.error(f"missing {key}: {path}")
                    spec[key] = str(path)
        configs[task] = {**base, "perception": bodies}
    out.mkdir(parents=True, exist_ok=True)
    config_dirs = {}
    for task, cfg in configs.items():
        cdir = out / "configs" / f"task_{task:02d}"
        cdir.mkdir(parents=True)
        (cdir / "mt3.json").write_text(json.dumps(cfg, indent=2) + "\n")
        config_dirs[task] = cdir
    (out / "cmd.json").write_text(
        json.dumps({"argv": sys.argv, "cwd": str(Path.cwd())}, indent=2) + "\n"
    )
    git = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff"], capture_output=True, text=True
    )
    (out / "git.txt").write_text(git.stdout + diff.stdout)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(HERE / "runtime"), str(REPO)]),
        "LIBERO_TYPE": "pro",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "ROBOT_PLATFORM": "LIBERO",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "RPENT_MT3_FIXTURE_SEED_POLICY": "rpent",
        "RPENT_MT3_SAM3_ENDPOINT": args.sam3_endpoint,
    }
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("RPENT_MT3_SCENE_SNAPSHOT", None)

    def run(task: int, seed: int) -> dict:
        cell = out / f"libero_10_swap_t{task}_s{seed}"
        cell.mkdir()
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
            str(config_dirs[task]),
            "--libero-type",
            "pro",
            "--max-episode-steps",
            "10000",
            "--output-dir",
            str(cell),
        ]
        (cell / "cmd.json").write_text(json.dumps(cmd, indent=2) + "\n")
        started = time.monotonic()
        with (cell / "stdout.log").open("w") as log:
            try:
                rc = subprocess.run(
                    cmd,
                    cwd=cell,
                    env={**env, "MUJOCO_EGL_DEVICE_ID": devices[seed % len(devices)]},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        path = cell / "mt3_episode.json"
        episode = json.loads(path.read_text()) if path.exists() else {}
        calls = episode.get("locator_calls", [])
        return {
            "task": task,
            "seed": seed,
            "rc": rc,
            "success": episode.get("success"),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "locator_failed": any(
                c.get("returncode", 0) != 0 or c.get("error") for c in calls
            ),
            "cell": str(cell),
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.procs) as pool:
        futures = [pool.submit(run, t, s) for t, s in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            ordered = sorted(rows, key=lambda r: (r["task"], r["seed"]))
            summary = {
                "episodes": len(rows),
                "success": sum(r["success"] is True for r in rows),
                "no_result": sum(r["success"] is None for r in rows),
                "locator_failed": sum(r["locator_failed"] for r in rows),
                "per_task": {
                    f"t{t}": f"{sum(r['success'] is True for r in rows if r['task'] == t)}/{sum(r['task'] == t for r in rows)}"
                    for t in sorted(set(tasks))
                },
                "rows": ordered,
            }
            (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(
                f"task {row['task']} seed {row['seed']}: {row['success']} (rc={row['rc']})",
                flush=True,
            )


if __name__ == "__main__":
    main()
