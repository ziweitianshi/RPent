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
masked median plus a calibrated offset is used. With ``refs`` the pose comes
instead from MT3's relative registration (Dreczkowski et al., original
``learning_thousand_tasks`` code: PointNet++ yaw, generalised ICP, 4-DOF
projection) of the live mask against each reference view of the body, a
training-scene frame whose pose is known; the median over the references is
kept. Fixture parts (a stove button,
a cabinet drawer, a microwave door hinge) are placed at a fixed transform from
their fixture. Nothing here reads simulator object state.
"""

from __future__ import annotations

import os
import sys
import time
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
    #: Reference views (.npz with stacked ``rgb``, ``depth_mm``, ``segmap``,
    #: ``K``, ``T_WC``, ``pose`` of training frames); MT3 relative registration
    #: against them when set, ahead of ``model``.
    refs: str = ""
    #: Object category and its description for a locator model (``LocatedPoses``);
    #: bodies sharing a category are told apart by ``expect``.
    category: str = ""
    describe: str = ""
    #: A tracked pixel is kept while its world point lies within this distance [m]
    #: of the body's last located point (not moved, not hidden).
    track_tol: float = 0.06


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


def depth_from_world(world: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    """Depth along the camera axis [m] of every pixel of a world map; 0 where invalid."""
    cam = (world.reshape(-1, 3).astype(np.float64) - cam2world[:3, 3]) @ cam2world[
        :3, :3
    ]
    z = np.where(np.isfinite(cam[:, 2]) & (cam[:, 2] > 0), cam[:, 2], 0.0)
    return z.reshape(world.shape[:2])


class MT3Registration:
    """Demo -> live 4-DOF transforms with the original MT3 code (imported unchanged
    from ``$RPENT_MT3_ORIG_REPO``, branch ``libero-adapt``)."""

    def __init__(self, repo: str = "", device: str = ""):
        repo = repo or os.environ.get(
            "RPENT_MT3_ORIG_REPO", "/data0/zsq/learning_thousand_tasks"
        )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from deployment.deploy_mt3 import apply_4dof_inductive_bias
        from thousand_tasks.core.utils.scene_state import SceneState
        from thousand_tasks.perception.pose_estimation.icp_6dof_pose_estimation_refinement import (
            Open3dIcpPoseRefinement,
        )
        from thousand_tasks.perception.pose_estimation.pnet_4dof_pose_regressor import (
            PointnetPoseRegressor_4dof,
        )

        if not device:
            gpu = os.environ.get("RPENT_MT3_TORCH_DEVICE") or os.environ.get(
                "MUJOCO_EGL_DEVICE_ID", "0"
            )
            device = gpu if not gpu.isdigit() else f"cuda:{gpu}"
        self.SceneState = SceneState
        self.bias = apply_4dof_inductive_bias
        self.device = torch.device(device)
        self.pnet: Any = None
        # Parameters of deploy_mt3.py.
        self.gicp = Open3dIcpPoseRefinement(
            error_metric="generalised-icp",
            max_correspondence_distance=0.1,
            max_iteration=20,
            depth_units="mm",
            timeout=3,
        )
        self._regressor = PointnetPoseRegressor_4dof

    def scene(self, rgb, depth_mm, mask, K):
        st = self.SceneState.initialise_from_dict(
            {
                "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
                "depth": depth_mm,
                "segmap": np.asarray(mask, dtype=bool),
                "intrinsic_matrix": np.asarray(K, dtype=np.float64),
            }
        )
        st.erode_segmap()
        st.crop_object_using_segmap()
        return st

    def transfer(
        self,
        demo,
        live,
        T_WC: np.ndarray,
        pivot: np.ndarray,
        T_WC_demo: np.ndarray | None = None,
    ) -> np.ndarray:
        """``W_T_delta_4dof`` taking the demo scene's object onto the live one
        (seen by the camera ``T_WC``; the demo by ``T_WC_demo``, default the
        same), rotated about ``pivot``."""
        T_WC_demo = T_WC if T_WC_demo is None else T_WC_demo
        if self.pnet is None:
            self.pnet = self._regressor(
                filter_pointcloud=True,
                n_points=2048,
                T_WC=T_WC,
                T_WC_demo=T_WC_demo,
                depth_units="mm",
                device=self.device,
            )
        self.pnet.extrinsics, self.pnet.extrinsics_demo = T_WC, T_WC_demo
        W_T = self.pnet.estimate_relative_pose(
            scene1_state=demo, scene2_state=live, visualise_pcds=False, verbose=False
        )
        C_T = np.linalg.inv(T_WC) @ W_T @ T_WC
        # The original different-camera path moves the demo points into the
        # live camera frame first.
        demo.T_WC, live.T_WC = T_WC_demo, T_WC
        C_ref = self.gicp.refine_relative_pose(
            scene1_state=demo,
            scene2_state=live,
            T_WC_live=T_WC,
            T_delta_init=C_T,
            verbose=False,
            visualise_pcds=False,
            different_cameras_live_demo=not np.allclose(T_WC_demo, T_WC),
        )
        W_ref = T_WC @ C_ref @ np.linalg.inv(T_WC)
        # deploy_mt3 rotates about the gripper tip, 0.24 m along the Sawyer wrist z;
        # here the pivot is given directly.
        T_WE = np.eye(4)
        T_WE[:3, 3] = np.asarray(pivot) - np.array([0.0, 0.0, 0.24])
        return self.bias(W_T_delta_6dof=W_ref, T_WE=T_WE)


def render_view(
    env: Any, camera: str, size: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Upright RGB, world map, camera metadata and metric depth of one camera."""
    from robots.libero.tools import _metric_depth, _world_from_depth

    rgb, depth = env.render_camera(
        camera_name=camera, height=size, width=size, depth=True
    )
    meta = env.get_camera_meta(camera, size, size)
    depth = _metric_depth(depth, meta)[::-1]
    world = _world_from_depth(depth, meta)
    return np.ascontiguousarray(np.asarray(rgb)[::-1]), world, meta, depth


def combine_poses(poses: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Median position and consensus yaw (the sample nearest all others) of
    pose estimates, with the median distance [m] of the positions to the median."""
    pos = np.stack([p[:3, 3] for p in poses])
    yaws = np.array([np.arctan2(p[1, 0], p[0, 0]) for p in poses])
    mid = np.median(pos, axis=0)
    spread = [np.abs(np.angle(np.exp(1j * (yaws - y)))).sum() for y in yaws]
    pose = poses[int(np.argmin(spread))].copy()
    pose[:3, 3] = mid
    return pose, float(np.median(np.linalg.norm(pos - mid, axis=1)))


def project(
    meta: dict[str, Any], point: np.ndarray, size: int
) -> tuple[int, int] | None:
    """Pixel (row, col) of a world point in an upright image, or None outside it."""
    K = np.asarray(meta["intrinsic_K"], dtype=float)
    T = np.asarray(meta["extrinsic_cam2world"], dtype=float)
    cam = (np.asarray(point, dtype=float) - T[:3, 3]) @ T[:3, :3]
    if cam[2] <= 0:
        return None
    u, v, w = K @ cam
    r, c = int(round(v / w)), int(round(u / w))
    return (r, c) if 0 <= r < size and 0 <= c < size else None


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
        self.refs = {}
        for body, spec in specs.items():
            if spec.refs:
                with np.load(spec.refs) as data:
                    self.refs[body] = {k: data[k] for k in data.files}
        self.registration: MT3Registration | None = (
            MT3Registration() if self.refs else None
        )
        #: Camera intrinsics and cam-to-world of the last frame (set by ``frame``).
        self.camera_meta: dict[str, Any] | None = None
        self._mask: np.ndarray | None = None
        #: Last estimate per body; reused when a body is not found (occluded).
        self.last: dict[str, np.ndarray] = {}
        #: Per-call diagnostics, appended for every estimated body.
        self.log: list[dict[str, Any]] = []

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB (H x W x 3) and world map (H x W x 3), both upright."""
        rgb, world, self.camera_meta, _ = render_view(self.env, self.camera, self.size)
        return rgb, world

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
        self._mask = None
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
                    self._mask = mask
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
        if body in self.refs and self.camera_meta is not None:
            pose = self._from_refs(body, rgb, world, self._mask, info)
            if pose is not None:
                return pose, info
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

    def _from_refs(self, body, rgb, world, mask, info) -> np.ndarray | None:
        """Median over the reference views of ``T_delta_4dof @ reference pose``."""
        reg, refs = self.registration, self.refs[body]
        K = np.asarray(self.camera_meta["intrinsic_K"], dtype=np.float64)
        T_WC = np.asarray(self.camera_meta["extrinsic_cam2world"], dtype=np.float64)
        depth_mm = np.round(1000 * depth_from_world(world, T_WC)).astype(np.uint16)
        tic = time.monotonic()
        poses, errors = [], 0
        for i in range(len(refs["pose"])):
            try:
                live = reg.scene(rgb, depth_mm, mask, K)
                demo = reg.scene(
                    refs["rgb"][i], refs["depth_mm"][i], refs["segmap"][i], refs["K"][i]
                )
                W4 = reg.transfer(demo, live, T_WC, refs["pose"][i][:3, 3])
            except Exception:  # noqa: BLE001 -- e.g. too few points; skip this reference
                errors += 1
                continue
            poses.append(W4 @ refs["pose"][i])
        info.update(
            refs_ok=len(poses),
            refs_failed=errors,
            refs_s=round(time.monotonic() - tic, 2),
        )
        if not poses:
            return None
        pose, spread = combine_poses(poses)
        info.update(refs_spread_cm=round(100 * spread, 2))
        return pose

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


class LocatedPoses(PerceivedPoses):
    """Poses from several cameras, with objects identified by a locator model.

    Every query renders all ``cameras``. A body's pixel in each camera comes from
    tracking (its last located point projected into the image, kept while the
    world point there stays within ``track_tol``: not moved, not hidden) or,
    when no camera keeps it, from one batched ``locator`` call for all such
    bodies (at most ``max_calls`` per episode). Each pixel prompts a SAM3 point
    mask; the masks are registered to the body's reference views (MT3 relative
    registration, see ``refs``; with ``camera_fallback`` only the first camera of
    ``cameras`` that yields pairs (text-prompt masks first), each camera against its own references unless
    ``cross_camera``) and the median over the pairs is the pose; a body
    that is not ``yaw_free`` keeps its nominal ``rotation``. A body whose masks
    moved less than 1 cm since its last estimate keeps that estimate.
    """

    def __init__(
        self,
        env: Any,
        segmenter: Any,
        specs: dict[str, BodySpec],
        locator: Any,
        cameras: list[str],
        size: int = 512,
        max_calls: int = 3,
        cross_camera: bool = False,
        camera_fallback: bool = True,
        max_spread: float = 0.02,
    ):
        super().__init__(env, segmenter, specs, camera=cameras[0], size=size)
        self.locator = locator
        self.cameras = list(cameras)
        self.max_calls = max_calls
        self.cross_camera = cross_camera
        self.camera_fallback = camera_fallback
        self.max_spread = max_spread
        #: Last located world point per segmented body (tracking anchor).
        self.anchor: dict[str, np.ndarray] = {}
        #: Per body: last estimate, its mask medians, and per camera the world
        #: point seen at the projection of its first median (a probe of whether
        #: the body moved).
        self._cache: dict[str, dict[str, Any]] = {}

    def views(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]]:
        out = {}
        for cam in self.cameras:
            rgb, world, meta, depth = render_view(self.env, cam, self.size)
            depth_mm = np.round(1000 * np.nan_to_num(depth)).astype(np.uint16)
            out[cam] = (rgb, world, meta, depth_mm)
        return out

    def _track(self, target: str, views) -> dict[str, list[int]]:
        if target not in self.anchor:
            return {}
        tol = self.specs[target].track_tol
        pixels = {}
        for cam, (_, world, meta, _) in views.items():
            px = project(meta, self.anchor[target], self.size)
            if px is None:
                continue
            xyz = world[px]
            if (
                np.isfinite(xyz).all()
                and np.linalg.norm(xyz - self.anchor[target]) < tol
            ):
                pixels[cam] = list(px)
        return pixels

    def _locate(self, targets: list[str], views) -> dict[str, dict[str, list[int]]]:
        from robots.libero.mt3.astra import assign, instances

        cats: dict[str, str] = {}
        for t in targets:
            spec = self.specs[t]
            if spec.category:
                cats[spec.category] = spec.describe or spec.category.replace("_", " ")
        if (
            not cats
            or self.locator is None
            or len(self.locator.calls) >= self.max_calls
        ):
            return {}
        answer = self.locator.locate({c: v[0] for c, v in views.items()}, cats)
        worlds = {c: v[1] for c, v in views.items()}
        found: dict[str, dict[str, list[int]]] = {}
        for cat in cats:
            # Every body of the category takes part, so instances go to the right one.
            members = [b for b, s in self.specs.items() if s.category == cat]
            expect = {
                b: self.anchor[b] if b in self.anchor else self.specs[b].expect
                for b in members
            }
            # A category with one body is one instance whatever face each camera saw.
            merge = (
                np.inf
                if len(members) == 1
                else max(0.08, *(self.specs[b].track_tol for b in members))
            )
            for body, inst in assign(
                members, instances(answer.get(cat, {}), worlds, merge), expect
            ).items():
                if body in targets:
                    found[body] = inst["pixels"]
        return found

    def _estimate(
        self, body: str, target: str, pixels, views, info
    ) -> np.ndarray | None:
        medians, masks, kinds = {}, {}, {}
        for cam, pixel in pixels.items():
            found = self._target_mask(target, pixel, views[cam])
            if found is None:
                continue
            mask, kinds[cam] = found
            masks[cam] = mask
            medians[cam] = np.median(masked_points(mask, views[cam][1]), axis=0)
        info["cams"] = sorted(masks)
        info["masks"] = kinds
        if not masks:
            return None
        self.anchor[target] = next(iter(medians.values()))
        cached = self._cache.get(body)
        if cached is not None and (
            any(
                np.linalg.norm(medians[cam] - cached["medians"][cam]) < 0.01
                for cam in medians
                if cam in cached["medians"]
            )
            or any(
                np.linalg.norm(probe - cached["probes"][cam]) < 0.01
                for cam, probe in self._probes(cached["point"], views).items()
                if cam in cached["probes"]
            )
        ):
            # Unmoved (a probe unchanged even if the gripper now hides part of it).
            info["reused"] = True
            return cached["pose"]
        refs = self.refs.get(body)
        if refs is None or self.registration is None:
            return None
        reg, poses, errors = self.registration, [], 0
        tic = time.monotonic()
        used = []
        # Text-prompt masks (as the references were chosen) before point masks,
        # then camera order.
        order = sorted(
            (c for c in self.cameras if c in masks),
            key=lambda c: (kinds[c] != "text", self.cameras.index(c)),
        )
        for cam in order:
            if poses and self.camera_fallback:
                break
            used.append(cam)
            mask = masks[cam]
            rgb, _, meta, depth_mm = views[cam]
            K = np.asarray(meta["intrinsic_K"], dtype=np.float64)
            T_WC = np.asarray(meta["extrinsic_cam2world"], dtype=np.float64)
            for i in range(len(refs["pose"])):
                if not self.cross_camera and str(refs["camera"][i]) != cam:
                    continue
                try:
                    live = reg.scene(rgb, depth_mm, mask, K)
                    demo = reg.scene(
                        refs["rgb"][i],
                        refs["depth_mm"][i],
                        refs["segmap"][i],
                        refs["K"][i],
                    )
                    W4 = reg.transfer(
                        demo, live, T_WC, refs["pose"][i][:3, 3], refs["T_WC"][i]
                    )
                except Exception:  # noqa: BLE001 -- e.g. too few points; skip this pair
                    errors += 1
                    continue
                poses.append(W4 @ refs["pose"][i])
        info.update(
            registered=used,
            refs_ok=len(poses),
            refs_failed=errors,
            refs_s=round(time.monotonic() - tic, 2),
        )
        if not poses:
            return None
        pose, spread = combine_poses(poses)
        info["refs_spread_cm"] = round(100 * spread, 2)
        if spread > self.max_spread:
            # The references disagree: an unreliable registration, not an estimate.
            return None
        spec = self.specs[body]
        if not spec.yaw_free:
            # Bodies that keep their orientation across scenes: only the position is estimated.
            pose[:3, :3] = quat_wxyz_to_matrix(spec.rotation)
        point = next(iter(medians.values()))
        self._cache[body] = {
            "pose": pose,
            "medians": medians,
            "point": point,
            "probes": self._probes(point, views),
        }
        return pose

    def _target_mask(self, target: str, pixel, view) -> tuple[np.ndarray, str] | None:
        """The body's mask at ``pixel``: among SAM3 masks from the body's text
        prompts in windows around the pixel (the masks its references were
        chosen from), the best-scoring one containing the pixel, else the one
        whose median lies nearest the pixel's world point (within 5 cm); else
        the SAM3 point mask at the pixel."""
        rgb, world = view[0], view[1]
        r, c = int(pixel[0]), int(pixel[1])
        spec = self.specs[target]
        prompts = [spec.prompt] if isinstance(spec.prompt, str) else list(spec.prompt)
        point = world[r, c]
        best, best_key = None, None
        for text in [t for t in prompts if t]:
            for win in [w for w in (spec.window, 2 * spec.window) if w] or [64]:
                r0, r1 = max(0, r - win), min(rgb.shape[0], r + win)
                c0, c1 = max(0, c - win), min(rgb.shape[1], c + win)
                res = self.segmenter.segment(
                    np.ascontiguousarray(rgb[r0:r1, c0:c1]),
                    text_prompt=text,
                    min_score=self.min_score,
                )
                if not res.found or res.mask is None:
                    continue
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                mask[r0:r1, c0:c1] = np.asarray(res.mask, dtype=bool)
                pts = masked_points(mask, world)
                if len(pts) < 10:
                    continue
                dist = (
                    float(np.linalg.norm(np.median(pts, axis=0) - point))
                    if np.isfinite(point).all()
                    else np.inf
                )
                # Containing the pixel ranks first, then score; else nearness.
                key = (1, float(res.score or 0.0)) if mask[r, c] else (0, -dist)
                if (mask[r, c] or dist < 0.05) and (best_key is None or key > best_key):
                    best, best_key = mask, key
        if best is not None:
            return best, "text"
        res = self.segmenter.segment(rgb, point=[r, c], min_score=0.0)
        if res.found and res.mask is not None:
            mask = np.asarray(res.mask, dtype=bool)
            if len(masked_points(mask, world)) >= 10:
                return mask, "point"
        return None

    def _probes(self, point: np.ndarray, views) -> dict[str, np.ndarray]:
        """World point seen at ``point``'s projection, per camera."""
        out = {}
        for cam, (_, world, meta, _) in views.items():
            px = project(meta, point, self.size)
            if px is not None and np.isfinite(world[px]).all():
                out[cam] = np.asarray(world[px], dtype=float)
        return out

    def __call__(self, bodies: list[str]) -> dict[str, np.ndarray]:
        views = self.views()
        target_of = {b: self.specs[b].parent or b for b in bodies}
        pixels = {t: self._track(t, views) for t in set(target_of.values())}
        lost = [t for t, px in pixels.items() if not px]
        located = self._locate(lost, views) if lost else {}
        for t, px in located.items():
            pixels[t] = px
        poses: dict[str, np.ndarray] = {}
        for body in bodies:
            target = target_of[body]
            info: dict[str, Any] = {
                "source": "located"
                if target in located
                else "tracked"
                if pixels[target]
                else "none"
            }
            pose = (
                self._estimate(body, target, pixels[target], views, info)
                if pixels[target]
                else None
            )
            found = pose is not None
            if not found:
                pose = self.last.get(body)
            if pose is None:
                spec = self.specs[body]
                pose = np.eye(4)
                pose[:3, :3] = quat_wxyz_to_matrix(spec.rotation)
                pose[:3, 3] = np.asarray(spec.expect or [0.0, 0.0, 0.0])
            self.last[body] = pose
            poses[body] = pose
            self.log.append({"body": body, "found": found, **info})
        return poses
