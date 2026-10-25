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

"""Static feasibility of grip-site paths in a LIBERO scene.

Runs inside the env worker on a copy of the simulator state, so the episode is
not touched. For every waypoint the Panda arm joints are solved by damped least
squares IK (warm-started from the previous waypoint), an object held between
the fingers is carried rigidly with the grip site, and contacts that penetrate
deeper than ``depth`` between the robot (or the held object) and the rest of
the scene are reported. Uses the scene's ground-truth collision geometry.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

GRIP_SITE = "gripper0_right_grip_site"
ROBOT_PREFIXES = ("robot0_", "gripper0_")
#: Robot bodies whose contacts with the mount / table are part of the setup.
_BASE_BODIES = ("robot0_base", "robot0_link0", "mount0_")


def _family(name: str) -> str:
    """Object instance a body belongs to: ``microwave_1_microdoorroot`` -> ``microwave_1``."""
    parts = name.split("_")
    for i, part in enumerate(parts):
        if part.isdigit():
            return "_".join(parts[: i + 1])
    return name


def _contact_exceeds_tolerance(mover: str, penetration: float, is_support: bool,
                               support_tolerance: float, general_tolerance: float) -> bool:
    tolerance = support_tolerance if "finger" in mover and is_support else general_tolerance
    return penetration > tolerance


class SceneChecker:
    """Feasibility queries against one MuJoCo model/state (copied per query)."""

    def __init__(self, model: Any, data: Any, *, scene_estimates=None, robot_joints=None, attached_body=None, attached_offset=None):
        import mujoco

        self.mj = mujoco
        self.model = model
        self.src = data
        # Strict-safe mode: build each query from model defaults and explicit
        # episode-start estimates. Never read scene qpos/qvel from ``data``.
        self.scene_estimates = scene_estimates
        self.robot_joints = robot_joints
        self.attached_body_name = attached_body
        self.attached_offset = None if attached_offset is None else np.asarray(attached_offset, dtype=float).reshape(4, 4)
        self.data = mujoco.MjData(model)
        m = model
        self.body_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            for i in range(m.nbody)
        ]
        self.attached_body = None
        if attached_body is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, attached_body)
            if bid >= 0:
                self.attached_body = int(model.body_rootid[bid])
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        self.arm_joints = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{k}")
            for k in range(1, 8)
        ]
        self.arm_qpos = np.array([m.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dof = np.array([m.jnt_dofadr[j] for j in self.arm_joints])
        self.lo = m.jnt_range[self.arm_joints, 0]
        self.hi = m.jnt_range[self.arm_joints, 1]
        self.robot_body = np.array(
            [n.startswith(ROBOT_PREFIXES) for n in self.body_names]
        )
        self.base_body = np.array([n.startswith(_BASE_BODIES) for n in self.body_names])
        #: Bodies whose root floats freely (objects that a touch can push aside)
        #: as opposed to fixtures, which stop the arm.
        free = mujoco.mjtJoint.mjJNT_FREE
        self.movable_body = np.array(
            [
                m.body_jntadr[r] >= 0 and m.jnt_type[m.body_jntadr[r]] == free
                for r in m.body_rootid
            ]
        )
        if self.scene_estimates is not None:
            # Model-default free objects have no trustworthy placement in
            # strict mode. Move them far outside the scene unless an estimate
            # was supplied; fixtures remain at their model defaults.
            estimated = set(self.scene_estimates)
            for root in range(m.nbody):
                if not self.movable_body[root]:
                    continue
                name = self.body_names[root]
                if name in estimated:
                    continue

    # ---- state helpers ----

    def _reset_copy(self) -> None:
        d, s = self.data, self.src
        if self.scene_estimates is None:
            d.qpos[:] = s.qpos
            d.mocap_pos[:] = s.mocap_pos
            d.mocap_quat[:] = s.mocap_quat
        else:
            self.mj.mj_resetData(self.model, d)
            estimated = set(self.scene_estimates)
            for root in range(self.model.nbody):
                if not self.movable_body[root] or self.body_names[root] in estimated:
                    continue
                jid = int(self.model.body_jntadr[root])
                adr = int(self.model.jnt_qposadr[jid])
                d.qpos[adr : adr + 3] = np.array([1000.0, 1000.0, 1000.0])
            for name, pose in self.scene_estimates.items():
                bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
                if bid < 0:
                    continue
                jid = int(self.model.body_jntadr[bid])
                if jid < 0:
                    pose = np.asarray(pose, dtype=float)
                    target_rot = np.empty(9)
                    self.mj.mju_quat2Mat(target_rot, pose[3:])
                    self.mj.mj_forward(self.model, d)
                    if pose.size != 7 or np.linalg.norm(d.xpos[bid] - pose[:3]) > 0.01 or np.linalg.norm(d.xmat[bid].reshape(3, 3) - target_rot.reshape(3, 3)) > 0.1:
                        raise ValueError(f"estimated static body {name} differs from model defaults")
                    continue
                jt = self.model.jnt_type[jid]
                adr = int(self.model.jnt_qposadr[jid])
                pose = np.asarray(pose, dtype=float)
                if jt == self.mj.mjtJoint.mjJNT_FREE and pose.size == 7:
                    d.qpos[adr : adr + 7] = pose
                elif jt == self.mj.mjtJoint.mjJNT_HINGE and pose.size == 7:
                    # Estimate articulated fixture angle from the observed
                    # hinge-body transform. Search only this one joint while
                    # all other state remains at model defaults.
                    target_pos = pose[:3]
                    target_rot = np.empty(9)
                    self.mj.mju_quat2Mat(target_rot, pose[3:])
                    target_rot = target_rot.reshape(3, 3)
                    lo, hi = self.model.jnt_range[jid]
                    best = (np.inf, d.qpos[adr])
                    for angle in np.linspace(lo, hi, 181):
                        d.qpos[adr] = angle
                        self.mj.mj_forward(self.model, d)
                        delta_rot = d.xmat[bid].reshape(3, 3) - target_rot
                        score = np.linalg.norm(d.xpos[bid] - target_pos) + 0.1 * np.linalg.norm(delta_rot)
                        if score < best[0]:
                            best = (score, angle)
                    d.qpos[adr] = best[1]
                elif jt == self.mj.mjtJoint.mjJNT_SLIDE and pose.size == 7:
                    target_pos = pose[:3]
                    base = d.qpos[adr]
                    lo, hi = self.model.jnt_range[jid]
                    best = (np.inf, base)
                    for value in np.linspace(lo, hi, 181):
                        d.qpos[adr] = value
                        self.mj.mj_forward(self.model, d)
                        score = np.linalg.norm(d.xpos[bid] - target_pos)
                        if score < best[0]:
                            best = (score, value)
                    d.qpos[adr] = best[1]
                elif self.model.jnt_type[jid] != self.mj.mjtJoint.mjJNT_FREE:
                    raise ValueError(f"unsupported estimated joint type for {name}")
            if self.robot_joints is not None:
                d.qpos[self.arm_qpos] = np.asarray(self.robot_joints, dtype=float)
        d.qvel[:] = 0.0
        self.mj.mj_kinematics(self.model, d)

    def _site_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.site]
        return pose

    def _body_pose(self, b: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[b].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[b]
        return pose

    def _set_held_pose(self, held: int, offset: np.ndarray) -> None:
        pose = self._site_pose() @ offset
        adr = self.model.jnt_qposadr[self.model.body_jntadr[held]]
        self.data.qpos[adr : adr + 3] = pose[:3, 3]
        self.data.qpos[adr + 3 : adr + 7] = _quat_wxyz(pose[:3, :3])
        self.mj.mj_forward(self.model, self.data)

    def held_body(self) -> int | None:
        """Free-floating body touching both finger pads in the live state, if any."""
        m, d = self.model, self.src
        touching: dict[int, set[str]] = {}
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for a, b in ((b1, b2), (b2, b1)):
                name = self.body_names[a]
                if "finger" in name and not self.robot_body[b]:
                    touching.setdefault(int(m.body_rootid[b]), set()).add(name)
        for root, fingers in touching.items():
            jnt = m.body_jntadr[root]
            if (
                len(fingers) >= 2
                and jnt >= 0
                and m.jnt_type[jnt] == self.mj.mjtJoint.mjJNT_FREE
            ):
                return root
        return None

    # ---- IK ----

    def solve(self, target: np.ndarray, iters: int = 100) -> tuple[float, float]:
        """Damped least squares on the arm joints toward ``target``; returns the
        remaining position [m] and rotation [deg] error."""
        mj, m, d = self.mj, self.model, self.data
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        pos_err, rot_err = np.inf, np.inf
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            cur = self._site_pose()
            dp = target[:3, 3] - cur[:3, 3]
            rel = target[:3, :3] @ cur[:3, :3].T
            dr = _rotvec(rel)
            pos_err, rot_err = (
                float(np.linalg.norm(dp)),
                float(np.degrees(np.linalg.norm(dr))),
            )
            if pos_err < 1e-3 and rot_err < 0.5:
                break
            mj.mj_jacSite(m, d, jacp, jacr, self.site)
            # Damping 1e-2 avoids the local minima that near-singular poses at
            # the edge of the workspace produce with lighter damping.
            jac = np.vstack([jacp[:, self.arm_dof], 0.5 * jacr[:, self.arm_dof]])
            err = np.concatenate([dp, 0.5 * dr])
            inv = jac.T @ np.linalg.inv(jac @ jac.T + 1e-2 * np.eye(6))
            dq = inv @ err
            # Null-space pull toward the starting posture, as the OSC
            # controller does, so the walk does not drift into far branches.
            dq += (np.eye(7) - inv @ jac) @ (0.1 * (self.q0 - d.qpos[self.arm_qpos]))
            q = d.qpos[self.arm_qpos] + np.clip(dq, -0.2, 0.2)
            d.qpos[self.arm_qpos] = np.clip(q, self.lo, self.hi)
        return pos_err, rot_err

    # ---- collisions ----

    def collisions(
        self, held: int | None, depth: float, ignore: set[int]
    ) -> list[tuple[str, str, float, bool]]:
        mj, m, d = self.mj, self.model, self.data
        mj.mj_forward(m, d)
        hits = []
        for k in range(d.ncon):
            c = d.contact[k]
            if c.dist > -depth:
                continue
            r1, r2 = (
                int(m.body_rootid[m.geom_bodyid[c.geom1]]),
                int(m.body_rootid[m.geom_bodyid[c.geom2]]),
            )
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            mover1 = self.robot_body[b1] or (held is not None and r1 == held)
            mover2 = self.robot_body[b2] or (held is not None and r2 == held)
            if mover1 == mover2:
                continue  # robot-robot, robot-held, or scene-scene
            mover, other = (b1, b2) if mover1 else (b2, b1)
            if (
                self.base_body[mover]
                or other in ignore
                or int(m.body_rootid[other]) in ignore
            ):
                continue
            hits.append(
                (
                    self.body_names[mover],
                    self.body_names[other],
                    float(-c.dist),
                    bool(self.movable_body[other]),
                )
            )
        return hits

    def check(
        self,
        path: list[np.ndarray],
        depth: float = 0.002,
        ik_pos_tol: float = 0.03,
        ik_rot_tol: float = 8.0,
        ignore_families: tuple[str, ...] = (),
        tail_families: tuple[str, ...] = (),
        tail: int = 0,
        contact_tolerance: bool = False,
        support_body: str | None = None,
        support_penetration: float = 0.02,
        general_penetration: float = 0.005,
    ) -> dict[str, Any]:
        """Walk ``path`` (4x4 grip-site poses) from the live state.

        Contacts with ``ignore_families`` never count; contacts with
        ``tail_families`` do not count on the last ``tail`` waypoints (the
        intended contact at the end of a placement into a container).
        """
        self._reset_copy()
        self.q0 = self.data.qpos[self.arm_qpos].copy()
        # Contact queries on ``src`` are live simulator observations. Strict
        # mode deliberately avoids them; attached-object constraints must be
        # supplied explicitly by the caller in a future constraint planner.
        held = self.attached_body if self.attached_body is not None else (None if self.scene_estimates is not None else self.held_body())
        held_offset = self.attached_offset
        if held is not None and held_offset is None:
            held_offset = np.linalg.inv(self._site_pose()) @ self._body_pose(held)
        if held is not None:
            self._set_held_pose(held, held_offset)
        support_roots: set[int] = set()
        if contact_tolerance and support_body:
            # Infer the support from frozen scene geometry: choose the closest
            # non-robot geom directly below the estimated object's footprint.
            self.mj.mj_forward(self.model, self.data)
            sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, support_body)
            if sid >= 0:
                root = int(self.model.body_rootid[sid])
                object_geoms = [g for g in range(self.model.ngeom)
                                if int(self.model.body_rootid[self.model.geom_bodyid[g]]) == root]
                object_bottom = min(
                    self.data.geom_xpos[g, 2] - float(np.max(self.model.geom_size[g]))
                    for g in object_geoms
                ) if object_geoms else np.inf
                object_radius = max(
                    (float(np.linalg.norm(self.model.geom_size[g, :2])) for g in object_geoms),
                    default=0.0,
                )
                for g in range(self.model.ngeom):
                    body = int(self.model.geom_bodyid[g])
                    other_root = int(self.model.body_rootid[body])
                    if self.robot_body[body] or other_root == root:
                        continue
                    top = self.data.geom_xpos[g, 2] + float(np.max(self.model.geom_size[g]))
                    horizontal = np.linalg.norm(self.data.geom_xpos[g, :2] - self.data.xpos[sid, :2])
                    support_radius = float(np.linalg.norm(self.model.geom_size[g, :2]))
                    if (
                        abs(object_bottom - top) <= support_penetration
                        and horizontal <= object_radius + support_radius + 0.01
                    ):
                        support_roots.add(other_root)
        fams = set(ignore_families)
        ignore = {
            i for i, n in enumerate(self.body_names) if fams and _family(n) in fams
        }
        tail_fams = set(tail_families)
        ignore_tail = ignore | {
            i for i, n in enumerate(self.body_names) if _family(n) in tail_fams
        }
        # Contacts present before moving (the held object on the table, a
        # finger already against a wall) count only once they press 5 mm
        # deeper than at the start.
        pre: dict[tuple[str, str], float] = {}
        query_depth = max(general_penetration, support_penetration) if contact_tolerance else depth
        for h in self.collisions(held, query_depth, ignore):
            pre[h[:2]] = max(pre.get(h[:2], 0.0), h[2])
        first_collision, first_ik_fail, bodies = None, None, set()
        first_movable, movable_pairs = None, set()
        goal_pos_err, goal_rot_err = np.inf, np.inf
        worst_pos, min_margin, deepest = 0.0, np.inf, 0.0
        for i, target in enumerate(path):
            pos_err, rot_err = self.solve(np.asarray(target))
            worst_pos = max(worst_pos, pos_err)
            q = self.data.qpos[self.arm_qpos]
            min_margin = min(
                min_margin,
                float(np.degrees(np.min(np.minimum(q - self.lo, self.hi - q)))),
            )
            if first_ik_fail is None and (pos_err > ik_pos_tol or rot_err > ik_rot_tol):
                first_ik_fail = i
            goal_pos_err, goal_rot_err = pos_err, rot_err
            if held is not None:
                self._set_held_pose(held, held_offset)
            now = ignore_tail if i >= len(path) - tail else ignore
            hits = [
                h
                for h in self.collisions(held, query_depth, now)
                if h[2] > pre.get(h[:2], -1.0) + 0.005
                and (
                    not contact_tolerance
                    or _contact_exceeds_tolerance(
                        h[0], h[2], self._body_root_id(h[1]) in support_roots,
                        support_penetration, general_penetration,
                    )
                )
            ]
            fixed = [h for h in hits if not h[3]]
            if fixed:
                bodies.update(f"{h[0]}|{h[1]}" for h in fixed)
                deepest = max(deepest, max(h[2] for h in fixed))
                if first_collision is None:
                    first_collision = i
            if len(fixed) < len(hits):
                movable_pairs.update(f"{h[0]}|{h[1]}" for h in hits if h[3])
                if first_movable is None:
                    first_movable = i
        return {
            "n": len(path),
            "held": self.body_names[held] if held is not None else None,
            "first_collision": first_collision,
            "first_ik_fail": first_ik_fail,
            "collision_pairs": sorted(bodies)[:8],
            # Touches of free objects: they are pushed rather than blocking,
            # but can knock an object over.
            "first_movable_contact": first_movable,
            "movable_pairs": sorted(movable_pairs)[:8],
            "max_ik_pos_err_cm": round(100 * worst_pos, 2),
            "max_penetration_cm": round(100 * deepest, 2),
            "min_joint_margin_deg": round(min_margin, 1),
            "goal_ik_pos_err_cm": round(100 * goal_pos_err, 2),
            "goal_ik_rot_err_deg": round(goal_rot_err, 1),
            # Unreachable intermediate waypoints are not disqualifying: the
            # controller cuts the corner as long as nothing is hit and the
            # final pose can be reached.
            "feasible": first_collision is None
            and goal_pos_err <= ik_pos_tol
            and goal_rot_err <= ik_rot_tol,
        }

    def _body_root_id(self, name: str) -> int:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        return int(self.model.body_rootid[bid]) if bid >= 0 else -1

    def plan(
        self,
        goal_pose: np.ndarray,
        *,
        seed: int = 0,
        max_iterations: int = 3000,
        timeout: float = 5.0,
        step_size: float = 0.25,
        edge_step: float = 0.05,
        shortcut_attempts: int = 100,
        pose_step: float = 0.01,
        depth: float = 0.002,
        goal_pos_tol: float = 0.03,
        goal_rot_tol: float = 8.0,
    ) -> dict[str, Any]:
        """Plan a collision-free joint path to ``goal_pose`` on a state copy.

        The goal tolerances match ``check``: poses at the edge of the workspace
        come within ~2-3 cm by IK, and settling closes the rest.
        """
        started = time.monotonic()
        rng = np.random.default_rng(seed)
        self._reset_copy()
        start = self.data.qpos[self.arm_qpos].copy()
        self.q0 = start.copy()
        held = self.attached_body if self.attached_body is not None else self.held_body()
        held_offset = self.attached_offset
        if held is not None and held_offset is None:
            held_offset = np.linalg.inv(self._site_pose()) @ self._body_pose(held)
        if held is not None:
            self._set_held_pose(held, held_offset)
        pre = {h[:2] for h in self.collisions(held, depth, set())}
        movable_pairs: set[str] = set()

        def set_q(q: np.ndarray) -> None:
            self.data.qpos[self.arm_qpos] = q
            self.mj.mj_forward(self.model, self.data)
            if held is not None:
                self._set_held_pose(held, held_offset)

        def valid(q: np.ndarray) -> bool:
            set_q(q)
            hits = [h for h in self.collisions(held, depth, set()) if h[:2] not in pre]
            movable_pairs.update(f"{h[0]}|{h[1]}" for h in hits if h[3])
            return not any(not h[3] for h in hits)

        goal_q = None
        best_error = (np.inf, np.inf)
        for attempt in range(9):
            if time.monotonic() - started >= timeout:
                break
            if attempt == 0:
                seed_q = start
            else:
                seed_q = rng.uniform(self.lo, self.hi)
            set_q(seed_q)
            self.q0 = seed_q.copy()
            pos_err, rot_err = self.solve(np.asarray(goal_pose), iters=150)
            q = self.data.qpos[self.arm_qpos].copy()
            if (pos_err, rot_err) < best_error:
                best_error = (pos_err, rot_err)
            if pos_err <= goal_pos_tol and rot_err <= goal_rot_tol and valid(q):
                goal_q = q
                break

        iterations = 0
        joint_path = None
        raw_path = []
        if goal_q is not None:
            joint_path, iterations = _rrt_connect(
                start,
                goal_q,
                valid,
                self.lo,
                self.hi,
                rng,
                max_iterations=max_iterations,
                timeout=timeout,
                started=started,
                step_size=step_size,
                edge_step=edge_step,
            )
        if joint_path:
            raw_path = joint_path
            joint_path = _shortcut_path(
                joint_path,
                valid,
                rng,
                attempts=shortcut_attempts,
                edge_step=edge_step,
            )
            joint_path = _resample_joint_path(joint_path, edge_step)
            poses = []
            for q in joint_path:
                set_q(q)
                poses.append(self._site_pose().copy())
            poses = _resample_pose_path(poses, pose_step)
        else:
            poses = []
        length = (
            float(sum(np.linalg.norm(b - a) for a, b in zip(raw_path, raw_path[1:])))
            if joint_path
            else 0.0
        )
        return {
            "success": bool(joint_path),
            "path": poses,
            "iterations": iterations,
            "time_s": round(time.monotonic() - started, 4),
            "path_length": round(length, 4),
            "movable_contacts": sorted(movable_pairs),
            "movable_contact_count": len(movable_pairs),
            "goal_ik_pos_err_cm": round(100 * best_error[0], 2),
            "goal_ik_rot_err_deg": round(best_error[1], 2),
        }


def _rotvec(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_rotvec()


def _quat_wxyz(rot: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return np.array([w, x, y, z])


def check_paths(sim: Any, paths: list, **kwargs: Any) -> list[dict[str, Any]]:
    """Check each path in ``paths`` against ``sim`` (robosuite MjSim)."""
    estimates = kwargs.pop("scene_estimates", None)
    joints = kwargs.pop("robot_joints", None)
    attached = kwargs.pop("attached_body", None)
    attached_offset = kwargs.pop("attached_offset", None)
    checker = SceneChecker(sim.model._model, sim.data._data, scene_estimates=estimates, robot_joints=joints,
                           attached_body=attached, attached_offset=attached_offset)
    tails = kwargs.pop("tails", None) or [0] * len(paths)
    return [
        checker.check([np.asarray(p) for p in path], tail=t, **kwargs)
        for path, t in zip(paths, tails)
    ]


def _edge_valid(a, b, valid, edge_step):
    count = max(1, int(np.ceil(np.max(np.abs(b - a)) / edge_step)))
    return all(valid(a + (b - a) * (i / count)) for i in range(1, count + 1))


def _extend(tree, target, valid, lo, hi, step_size, edge_step):
    nodes, parents = tree
    nearest = int(np.argmin(np.linalg.norm(nodes - target, axis=1)))
    delta = target - nodes[nearest]
    distance = np.linalg.norm(delta)
    q = nodes[nearest] + delta * min(1.0, step_size / max(distance, 1e-12))
    q = np.clip(q, lo, hi)
    if np.linalg.norm(q - nodes[nearest]) < 1e-8 or not _edge_valid(
        nodes[nearest], q, valid, edge_step
    ):
        return None
    nodes.append(q)
    parents.append(nearest)
    return len(nodes) - 1


def _trace(tree, index):
    nodes, parents = tree
    path = []
    while index >= 0:
        path.append(nodes[index])
        index = parents[index]
    return path[::-1]


def _rrt_connect(
    start,
    goal,
    valid,
    lo,
    hi,
    rng,
    *,
    max_iterations,
    timeout,
    started,
    step_size,
    edge_step,
):
    if not valid(start) or not valid(goal):
        return None, 0
    trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]
    for iteration in range(max_iterations):
        if time.monotonic() - started >= timeout:
            break
        a_idx = iteration % 2
        ta, tb = trees[a_idx], trees[1 - a_idx]
        sample = goal if iteration % 10 == 0 else rng.uniform(lo, hi)
        idx = _extend(ta, sample, valid, lo, hi, step_size, edge_step)
        if idx is None:
            continue
        q = ta[0][idx]
        while True:
            before = len(tb[0])
            j = _extend(tb, q, valid, lo, hi, step_size, edge_step)
            if j is None:
                break
            if np.linalg.norm(tb[0][j] - q) < 1e-8:
                left, right = _trace(ta, idx), _trace(tb, j)
                path = left + right[-2::-1] if a_idx == 0 else right + left[-2::-1]
                return path, iteration + 1
            if len(tb[0]) == before:
                break
        trees[a_idx], trees[1 - a_idx] = tb, ta
    return None, iteration + 1 if max_iterations else 0


def _shortcut_path(path, valid, rng, *, attempts, edge_step):
    path = [np.asarray(q).copy() for q in path]
    for _ in range(attempts):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j > i + 1 and _edge_valid(path[i], path[j], valid, edge_step):
            path[i + 1 : j] = []
    return path


def _resample_joint_path(path, step):
    result = [path[0]]
    for a, b in zip(path, path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        result.extend(a + (b - a) * (i / n) for i in range(1, n + 1))
    return result


def _resample_pose_path(path, step):
    from scipy.spatial.transform import Rotation, Slerp

    result = []
    for a, b in zip(path, path[1:]):
        distance = np.linalg.norm(b[:3, 3] - a[:3, 3])
        count = max(1, int(np.ceil(distance / step)))
        rotations = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
        slerp = Slerp([0.0, 1.0], rotations)
        for t in np.arange(count) / count:
            pose = np.eye(4)
            pose[:3, 3] = a[:3, 3] * (1 - t) + b[:3, 3] * t
            pose[:3, :3] = slerp([t]).as_matrix()[0]
            result.append(pose)
    if path:
        result.append(path[-1])
    return result


def plan_path(sim: Any, goal_pose: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    """Plan from the live simulator state without modifying it."""
    estimates = kwargs.pop("scene_estimates", None)
    joints = kwargs.pop("robot_joints", None)
    attached = kwargs.pop("attached_body", None)
    attached_offset = kwargs.pop("attached_offset", None)
    checker = SceneChecker(sim.model._model, sim.data._data, scene_estimates=estimates, robot_joints=joints,
                           attached_body=attached, attached_offset=attached_offset)
    return checker.plan(np.asarray(goal_pose), **kwargs)
