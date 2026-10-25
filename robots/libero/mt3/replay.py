# Copyright 2026 The RPent Authors.
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
"""Run one LIBERO cell with MT3 against the toolkit's live environment.

The memory directory holds ``mt3.json``: the execution settings
(:class:`~robots.libero.mt3.core.MT3Config`) and ``library``, the directory of
task libraries (``<library>/libero_<family>/task_XX.npz``, absolute or relative
to the memory directory). A perturbed suite reuses its base suite's library:
``_swap`` moves objects and fixtures, ``_lan`` only rewords the instruction.

Object poses are read from the simulator (``env.poses``), not perceived. Each
segment runs as one ``mt3_segment`` tool call, so every call ends with the
toolkit's usual state capture and success bookkeeping.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from robots.libero.mt3.core import (
    GRIP_SITE,
    EnvHooks,
    MT3Config,
    MT3Runner,
    TaskLibrary,
    pose_from_wxyz,
    reach_retrieval,
)
from rpent.utils.logging import get_output_dir

#: ``<family>[_<perturbation>]_t<task>_s<seed>``, the tag the CLI builds per cell.
_CELL = re.compile(r"^(10|goal|object|spatial)(?:_(swap|lan))?_t(\d+)_s(\d+)$")
CONFIG_NAME = "mt3.json"

_SEGMENT_SPEC = {
    "name": "mt3_segment",
    "description": "Align to and replay one segment of the retrieved demonstration.",
    "input_schema": {
        "type": "object",
        "properties": {"segment": {"type": "integer"}},
        "required": ["segment"],
    },
}
_HOLD_SPEC = {
    "name": "mt3_hold",
    "description": "Hold the last MT3 command while objects settle.",
    "input_schema": {"type": "object", "properties": {}},
}
_REPLAY_SPEC = {
    "name": "mt3_replay_actions",
    "description": "Send the retrieved demonstration's raw actions.",
    "input_schema": {"type": "object", "properties": {}},
}


def parse_cell(cell_tag: str) -> tuple[str, int, int]:
    """Base suite, task and seed of a cell tag MT3 can run."""
    match = _CELL.match(cell_tag)
    if not match:
        raise ValueError(
            f"MT3 cannot run cell {cell_tag!r}: it needs a base, _swap or _lan "
            "suite, whose task i is the base suite's task i"
        )
    family, _, task, seed = match.groups()
    return f"libero_{family}", int(task), int(seed)


def load_config(memory_root: Path) -> tuple[MT3Config, Path]:
    """Settings and library directory named by ``<memory_root>/mt3.json``."""
    data = json.loads((memory_root / CONFIG_NAME).read_text())
    library = Path(data["library"])
    if not library.is_absolute():
        library = memory_root / library
    return MT3Config.from_json(data), library


def _perceiver(env: Any, cfg: MT3Config) -> Any:
    """Camera-based pose estimator for ``pose_source: perception``."""
    from robots.libero.mt3.perception import PerceivedPoses, parse_specs
    from rpent.robots.components.sam3_client import Sam3Client
    from rpent.utils.rpc.client_utils import make_rpc_client

    endpoint = cfg.perception_endpoint or os.environ.get("RPENT_MT3_SAM3_ENDPOINT", "")
    if not endpoint:
        raise ValueError(
            "pose_source=perception needs perception_endpoint or $RPENT_MT3_SAM3_ENDPOINT"
        )
    segmenter = Sam3Client(make_rpc_client(endpoint))
    if cfg.perception_locator == "astra":
        from robots.libero.mt3.astra import AstraLocator
        from robots.libero.mt3.perception import LocatedPoses

        return LocatedPoses(
            env,
            segmenter,
            parse_specs(cfg.perception),
            AstraLocator(
                model=cfg.perception_locator_model, task=env.get_task_language()
            ),
            cfg.perception_cameras,
            max_calls=cfg.perception_locator_calls,
            cross_camera=cfg.perception_cross_camera,
            camera_fallback=cfg.perception_camera_fallback,
            refresh_parts=cfg.perception_refresh_parts,
            share_fixture=cfg.perception_share_fixture,
        )
    if cfg.perception_locator:
        raise ValueError(f"unknown perception_locator {cfg.perception_locator!r}")
    return PerceivedPoses(env, segmenter, parse_specs(cfg.perception))


def _record_views(env: Any, bodies: list[str], cameras: list[str], index: int) -> None:
    """Save each camera's RGB, depth [mm] and metadata with the simulator poses
    of ``bodies`` and the gripper (data collection, see ``record_views``)."""
    from robots.libero.mt3.perception import render_view

    out = Path(get_output_dir()) / "views"
    out.mkdir(exist_ok=True)
    arrays: dict[str, Any] = {}
    meta: dict[str, Any] = {"bodies": bodies, "cameras": cameras}
    for camera in cameras:
        rgb, _, cam_meta, depth = render_view(env, camera, 512)
        arrays[f"{camera}/rgb"] = rgb
        arrays[f"{camera}/depth_mm"] = np.round(1000 * np.nan_to_num(depth)).astype(
            np.uint16
        )
        meta[camera] = {k: cam_meta[k] for k in ("intrinsic_K", "extrinsic_cam2world")}
    poses = env.poses(bodies, [GRIP_SITE])
    for name, values in poses.items():
        arrays[f"pose/{name}"] = pose_from_wxyz(values)
    arrays["meta"] = json.dumps(meta)
    np.savez_compressed(out / f"q{index:03d}.npz", **arrays)


def _goal_report(env: Any) -> dict[str, Any]:
    """Final goal predicates (diagnostics), or the error when the env cannot report."""
    try:
        return env.goal_report()
    except Exception as exc:  # noqa: BLE001 -- diagnostics must not fail the episode
        return {"error": f"{type(exc).__name__}: {exc}"}


def _hooks(
    toolkit: Any, cfg: MT3Config | None = None, perceiver: Any = None
) -> EnvHooks:
    primitives = toolkit.primitives
    env = primitives.env

    queries = [0]
    frozen: dict[str, np.ndarray] = {}

    def body_poses(bodies: list[str]) -> dict[str, np.ndarray]:
        if cfg is not None and cfg.freeze_poses and all(b in frozen for b in bodies):
            return {b: frozen[b].copy() for b in bodies}
        if cfg is not None and cfg.record_views:
            _record_views(env, list(bodies), cfg.record_views, queries[0])
            queries[0] += 1
        if perceiver is not None:
            poses = perceiver(list(bodies))
        else:
            poses = {k: pose_from_wxyz(v) for k, v in env.poses(bodies).items()}
        if cfg is not None and cfg.freeze_poses:
            for b, pose in poses.items():
                if cfg.freeze_parts or b.endswith("_main"):
                    frozen.setdefault(b, pose.copy())
        return poses

    def start_poses(bodies: list[str]) -> dict[str, np.ndarray]:
        """Return poses captured on first request, before replay changes them."""
        poses = body_poses(bodies)
        for body, pose in poses.items():
            frozen.setdefault(body, pose.copy())
        return {body: pose.copy() for body, pose in frozen.items() if body in bodies}

    def strict_planning_options():
        if cfg is None or not cfg.strict_planning_scene:
            return {}
        from scipy.spatial.transform import Rotation

        estimates = start_poses(
            [
                "white_yellow_mug_1_main",
                "microwave_1_main",
                "microwave_1_microdoorroot",
            ]
        )
        return {
            "scene_estimates": {
                name: np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]].tolist()
                for name, pose in estimates.items()
            },
            "robot_joints": env.robot_joints(),
        }

    def check_paths(paths, options=None):
        opts = dict(options or {})
        opts.update(strict_planning_options())
        return env.check_paths(paths, opts)

    return EnvHooks(
        step=primitives._step_env,
        eef_pose=lambda: pose_from_wxyz(env.poses([], [GRIP_SITE])[GRIP_SITE]),
        joint_positions=lambda: np.asarray(primitives._last_obs["states"][3:9], dtype=float),
        goal_report=env.goal_report,
        body_poses=body_poses,
        start_pose=(
            start_poses(["white_yellow_mug_1_main", "microwave_1_main"])
            if cfg is not None and cfg.grasp_tilt_by_dy
            else {}
        ),
        done=lambda: bool(env.terminated or env.truncated),
        check_paths=check_paths,
        plan_path=lambda goal, options=None: env.plan_path(
            goal, {**strict_planning_options(), **(options or {})}
        ),
    )


def execute(toolkit: Any, name: str, arguments: dict[str, Any]) -> None:
    """Execute one registered MT3 tool; raise on its error result.

    The tool's own diagnostics stay on the runner; the toolkit returns the
    state it captured after the call.
    """
    result = toolkit.execute_tool(name, arguments).result
    if error := result.get("error"):
        raise RuntimeError(f"{name} failed: {error}\n{result.get('traceback', '')}")


def _calibrate(toolkit: Any, lib: TaskLibrary, perceiver: Any, cell_tag: str) -> dict:
    """Save the camera frame and the simulator poses at the episode start; no motion.

    Calibration data for building point-cloud models on non-test scenes.
    """
    env = toolkit.primitives.env
    out = Path(get_output_dir())
    rgb, world = perceiver.frame()
    np.save(out / "perception_rgb.npy", rgb)
    np.save(out / "perception_world.npy", world.astype(np.float32))
    parents = {perceiver.specs[b].parent for b in lib.ref_bodies} - {""}
    record = {"cell": cell_tag, "gt": env.poses(sorted(set(lib.ref_bodies) | parents))}
    (out / "perception_calib.json").write_text(json.dumps(record, indent=1))
    (out / "mt3_episode.json").write_text(
        json.dumps(
            {"cell": cell_tag, "mode": "perception_calibrate", "success": False},
            indent=1,
        )
    )
    return {
        "done": False,
        "plan": 0,
        "anchors": 0,
        "program": "mt3/perception_calibrate",
    }


def run_mt3(
    toolkit: Any, cell_tag: str, note: Callable[[str], None] = lambda _: None
) -> dict:
    """The ``RobotSpec`` hook behind ``--planner mt3``: one episode, no retries."""
    suite, task, seed = parse_cell(cell_tag)
    cfg, library_dir = load_config(Path(toolkit.memory.root))
    lib = TaskLibrary(library_dir / suite / f"task_{task:02d}.npz")
    perceiver = (
        _perceiver(toolkit.primitives.env, cfg)
        if cfg.pose_source == "perception"
        else None
    )
    hooks = _hooks(toolkit, cfg, perceiver)
    if cfg.perception_calibrate:
        return _calibrate(toolkit, lib, perceiver, cell_tag)
    trace: dict[str, list] = {"states": [], "actions": []}
    if cfg.record_trace:
        env = toolkit.primitives.env
        step = hooks.step

        def recorded_step(action: np.ndarray) -> None:
            trace["states"].append(env.sim_state())
            trace["actions"].append(np.asarray(action, dtype=np.float64))
            step(action)

        hooks.step = recorded_step
    # Fixture roots are not contained in the saved qpos init-state vector.
    # Preserve their actual placement when evaluating native RPent seeding.
    (Path(get_output_dir()) / "initial_body_poses.json").write_text(
        json.dumps(toolkit.primitives.env.poses(), indent=1)
    )
    scene = hooks.body_poses(lib.ref_bodies)
    demo, retrieval = lib.nearest(scene, cfg)
    if cfg.retrieval == "reach" and hooks.check_paths is not None:
        demo, reach = reach_retrieval(lib, scene, cfg, hooks)
        retrieval = {**retrieval, "demo": demo, "reach": reach}
    runner = MT3Runner(lib, demo, cfg, hooks)
    note(
        f"MT3 {cfg.mode}: {suite} task {task}, demo {demo} of {len(lib)} "
        f"(scene distance {retrieval['dist']:.3f}), "
        f"{len(runner.segments)} segments"
    )

    def segment(segment: int) -> dict[str, Any]:
        return runner.run_segment(int(segment))

    for spec, handler in (
        (_SEGMENT_SPEC, segment),
        (_HOLD_SPEC, runner.hold),
        (_REPLAY_SPEC, runner.replay_actions),
    ):
        toolkit.add_tool(
            spec["name"],
            spec,
            partial(toolkit._execute_primitive, spec["name"], handler),
        )

    calls = 0
    if cfg.mode == "replay":
        execute(toolkit, "mt3_replay_actions", {})
        calls += 1
    else:
        for k in range(len(runner.segments)):
            if hooks.done():
                break
            execute(toolkit, "mt3_segment", {"segment": k})
            calls += 1
            note(
                f"  segment {k} ({runner.logs[-1]['ref']}): {runner.logs[-1]['status']}"
            )
            if runner.logs[-1].get("blocked"):
                break
        if not hooks.done() and not any(c.get("blocked") for c in runner.logs):
            execute(toolkit, "mt3_hold", {})
            calls += 1

    episode = {
        "cell": cell_tag,
        "suite": suite,
        "task": task,
        "seed": seed,
        "mode": cfg.mode,
        "fixture_seed_policy": os.environ.get(
            "RPENT_MT3_FIXTURE_SEED_POLICY", "deterministic"
        ),
        "config": cfg.__dict__,
        "library": str(lib.path),
        "retrieval": retrieval,
        "scene": {
            body: [round(float(v), 5) for v in pose[:3, 3]]
            + [round(float(np.arctan2(pose[1, 0], pose[0, 0])), 5)]
            for body, pose in scene.items()
        },
        "calls": runner.logs,
        "pose_source": cfg.pose_source,
        "perception_log": perceiver.log if perceiver is not None else None,
        "locator_calls": getattr(getattr(perceiver, "locator", None), "calls", None),
        "steps": runner.steps,
        "final_goal": _goal_report(toolkit.primitives.env),
        "success": bool(toolkit.solved()),
        "terminated": bool(toolkit.primitives.env.terminated),
        "truncated": bool(toolkit.primitives.env.truncated),
    }
    if cfg.record_trace and trace["actions"]:
        np.savez_compressed(
            Path(get_output_dir()) / "mt3_trace.npz",
            states=np.asarray(trace["states"]),
            actions=np.asarray(trace["actions"]),
            success=bool(toolkit.solved()),
        )
    (Path(get_output_dir()) / "mt3_episode.json").write_text(
        json.dumps(episode, indent=1)
    )
    return {
        "done": episode["success"],
        "plan": calls,
        "anchors": len(scene),
        "program": f"mt3/{cfg.mode}/demo{demo}",
    }
