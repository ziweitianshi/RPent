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

"""Object poses for MT3 estimated from the camera instead of the simulator.

Each query renders the agentview camera with depth and back-projects every
pixel to world coordinates (the map RPent's ``back_project`` tool uses). A body
is segmented with SAM3 from text prompts; several windows around the body's
expected location and the whole image are tried, and the best-scoring mask
whose points lie near the expected location wins (this also tells apart two
instances of the same object). The pose is then found by registering the
body's point-cloud model (points in the body frame, collected on training
scenes) to the masked points with ICP restricted to x, y, z and, for bodies
that turn, yaw, the degrees of freedom MT3 transfers. Without a model the
masked median plus a calibrated offset is used. Fixture parts (a stove button,
a cabinet drawer, a microwave door hinge) are placed at a fixed transform from
their fixture. Nothing here reads simulator object state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class BodySpec:
    """How to estimate one body's pose."""

    #: SAM3 text prompt, or prompts tried in order. Empty for bodies placed
    #: relative to ``parent`` or kept at ``expect``.
    prompt: str | list[str] = ""
    #: Fixture this part is placed relative to, and the part's pose in the
    #: fixture's frame (4x4, row-major).
    parent: str = ""
    parent_rel: list[list[float]] | None = None
    #: World offset [m] added to the masked median when there is no ``model``.
    offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: Nominal orientation, quaternion wxyz (the ICP start; kept if not ``yaw_free``).
    rotation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    #: Expected world position [m]: window centre and plausibility check.
    expect: list[float] | None = None
    #: Window half-size [px]; windows ``window``, ``2 * window`` and the whole
    #: image are tried.
    window: int = 0
    #: A mask is accepted only if its median lies within this distance [m] of
    #: ``expect`` in the table plane.
    max_dist: float = 0.15
    #: Body-frame point cloud model (.npy, N x 3); ICP registration when set.
    model: str = ""
    #: Whether ICP may turn the body about the vertical axis.
    yaw_free: bool = False


def parse_specs(raw: dict[str, Any]) -> dict[str, BodySpec]:
    known = BodySpec.__dataclass_fields__
    return {
        body: BodySpec(**{k: v for k, v in spec.items() if k in known})
        for body, spec in raw.items()
    }


def nearest_pixel(world: np.ndarray, point: np.ndarray) -> tuple[int, int]:
    """Pixel whose back-projected point is closest to ``point`` in the table plane."""
    d = np.linalg.norm(world[..., :2] - np.asarray(point[:2]), axis=-1)
    d[~np.isfinite(d)] = np.inf
    r, c = np.unravel_index(int(np.argmin(d)), d.shape)
    return int(r), int(c)


def masked_points(mask: np.ndarray, world: np.ndarray) -> np.ndarray:
    pts = world[mask].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    return pts[valid]


def quat_wxyz_to_matrix(quat: list[float]) -> np.ndarray:
    return Rotation.from_quat(np.roll(np.asarray(quat, dtype=float), -1)).as_matrix()


def register(
    live: np.ndarray,
    model: np.ndarray,
    start: np.ndarray,
    yaw_free: bool,
    iters: int = 30,
    keep: float = 0.8,
) -> tuple[np.ndarray, float]:
    """ICP of a body-frame ``model`` onto world points ``live`` from pose ``start``.

    Each step matches every live point to its nearest model point, drops the
    worst ``1 - keep`` of the pairs, and solves the x, y, z (and yaw) update in
    closed form. Returns the pose and the RMS distance of the kept pairs.
    """
    tree = cKDTree(model)
    pose = start.copy()
    rms = np.inf
    for _ in range(iters):
        body = (live - pose[:3, 3]) @ pose[:3, :3]
        dist, idx = tree.query(body)
        sel = dist <= np.quantile(dist, keep)
        q = live[sel]
        p = model[idx[sel]] @ pose[:3, :3].T + pose[:3, 3]
        cq, cp = q.mean(axis=0), p.mean(axis=0)
        angle = 0.0
        if yaw_free:
            a, b = p[:, :2] - cp[:2], q[:, :2] - cq[:2]
            angle = float(
                np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
            )
        turn = Rotation.from_euler("z", angle).as_matrix()
        delta = np.eye(4)
        delta[:3, :3] = turn
        delta[:3, 3] = cq - turn @ cp
        pose = delta @ pose
        rms = float(np.sqrt(np.mean(dist[sel] ** 2)))
        if np.linalg.norm(delta[:3, 3]) < 1e-5 and abs(angle) < 1e-5:
            break
    return pose, rms


class PerceivedPoses:
    """Callable ``bodies -> {body: 4x4 pose}`` from one fresh camera frame."""

    def __init__(
        self,
        env: Any,
        segmenter: Any,
        specs: dict[str, BodySpec],
        camera: str = "agentview",
        size: int = 512,
        min_score: float = 0.05,
    ):
        self.env = env
        self.segmenter = segmenter
        self.specs = specs
        self.camera = camera
        self.size = size
        self.min_score = min_score
        self.models = {
            body: np.load(spec.model) for body, spec in specs.items() if spec.model
        }
        #: Last estimate per body; reused when a body is not found (occluded).
        self.last: dict[str, np.ndarray] = {}
        #: Per-call diagnostics, appended for every estimated body.
        self.log: list[dict[str, Any]] = []

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB (H x W x 3) and world map (H x W x 3), both upright."""
        from robots.libero.tools import _metric_depth, _world_from_depth

        rgb, depth = self.env.render_camera(
            camera_name=self.camera, height=self.size, width=self.size, depth=True
        )
        meta = self.env.get_camera_meta(self.camera, self.size, self.size)
        world = _world_from_depth(_metric_depth(depth, meta)[::-1], meta)
        return np.ascontiguousarray(np.asarray(rgb)[::-1]), world

    def segment(
        self,
        rgb: np.ndarray,
        world: np.ndarray,
        prompt: str | list[str],
        spec: BodySpec,
        expect: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Points of the best plausible mask, and how it was found."""
        expect = np.asarray(spec.expect) if expect is None and spec.expect else expect
        windows = (
            [spec.window, 2 * spec.window, 0]
            if spec.window and expect is not None
            else [0]
        )
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        best: tuple[float, np.ndarray | None, dict[str, Any]] = (-1.0, None, {})
        for text in prompts:
            for win in windows:
                r0, r1, c0, c1 = 0, rgb.shape[0], 0, rgb.shape[1]
                if win:
                    r, c = nearest_pixel(world, expect)
                    r0, r1 = max(0, r - win), min(rgb.shape[0], r + win)
                    c0, c1 = max(0, c - win), min(rgb.shape[1], c + win)
                result = self.segmenter.segment(
                    np.ascontiguousarray(rgb[r0:r1, c0:c1]),
                    text_prompt=text,
                    min_score=self.min_score,
                )
                if not result.found or result.mask is None:
                    continue
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[r0:r1, c0:c1] = np.asarray(result.mask, dtype=bool)
                pts = masked_points(mask, world)
                if len(pts) < 10:
                    continue
                if expect is not None:
                    off = np.linalg.norm(np.median(pts, axis=0)[:2] - expect[:2])
                    if off > spec.max_dist:
                        continue
                score = float(result.score or 0.0)
                if score > best[0]:
                    best = (
                        score,
                        pts,
                        {"prompt": text, "window": win, "score": round(score, 3)},
                    )
        return best[1], best[2]

    def estimate(
        self,
        body: str,
        rgb: np.ndarray,
        world: np.ndarray,
        expect: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Pose of a segmented body (not a fixture part), or None if not found."""
        spec = self.specs[body]
        if not spec.prompt:
            return None, {}
        pts, info = self.segment(rgb, world, spec.prompt, spec, expect)
        if pts is None:
            return None, info
        start = np.eye(4)
        start[:3, :3] = quat_wxyz_to_matrix(spec.rotation)
        if body in self.models:
            model = self.models[body]
            start[:3, 3] = np.median(pts, axis=0) - start[:3, :3] @ np.median(
                model, axis=0
            )
            pose, rms = register(pts, model, start, spec.yaw_free)
            info.update(n=int(len(pts)), icp_rms_cm=round(100 * rms, 2))
            return pose, info
        start[:3, 3] = np.median(pts, axis=0) + np.asarray(spec.offset)
        info.update(n=int(len(pts)))
        return start, info

    def __call__(self, bodies: list[str]) -> dict[str, np.ndarray]:
        rgb, world = self.frame()
        fixtures: dict[str, tuple[np.ndarray | None, dict[str, Any]]] = {}
        poses: dict[str, np.ndarray] = {}
        for body in bodies:
            spec = self.specs[body]
            if spec.parent:
                if spec.parent not in fixtures:
                    fixtures[spec.parent] = self.estimate(spec.parent, rgb, world)
                fixture, info = fixtures[spec.parent]
                pose = (
                    None if fixture is None else fixture @ np.asarray(spec.parent_rel)
                )
            else:
                pose, info = self.estimate(body, rgb, world)
            found = pose is not None
            if not found:
                pose = self.last.get(body)
            if pose is None:
                # Never seen: the expected pose.
                pose = np.eye(4)
                pose[:3, :3] = quat_wxyz_to_matrix(spec.rotation)
                pose[:3, 3] = np.asarray(spec.expect or [0.0, 0.0, 0.0])
            self.last[body] = pose
            poses[body] = pose
            self.log.append({"body": body, "found": found, **info})
        return poses
