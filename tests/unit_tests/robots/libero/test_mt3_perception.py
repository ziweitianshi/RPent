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

import numpy as np

from robots.libero.mt3.core import make_pose, rz
from robots.libero.mt3.perception import PerceivedPoses, depth_from_world
from robots.libero.tools import _world_from_depth


def _camera():
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array(
        [[0.0, 0.53, -0.85], [1.0, 0.0, 0.0], [0.0, -0.85, -0.53]]
    )
    cam2world[:3, :3] /= np.linalg.norm(cam2world[:3, :3], axis=0)
    cam2world[:3, 3] = [0.6, 0.0, 0.96]
    K = np.array([[618.0, 0.0, 32.0], [0.0, 618.0, 32.0], [0.0, 0.0, 1.0]])
    return {"intrinsic_K": K.tolist(), "extrinsic_cam2world": cam2world.tolist()}


def test_depth_from_world_inverts_back_projection():
    meta = _camera()
    depth = np.random.default_rng(0).uniform(0.6, 1.4, size=(64, 64))
    world = _world_from_depth(depth, meta)
    back = depth_from_world(world, np.array(meta["extrinsic_cam2world"]))
    assert np.allclose(back, depth, atol=1e-9)


class _FakeRegistration:
    """Returns a fixed transform per reference; the third one is an outlier."""

    def __init__(self, deltas):
        self.deltas = iter(deltas)

    def scene(self, *args):
        return None

    def transfer(self, demo, live, T_WC, pivot, T_WC_demo=None):
        return next(self.deltas)


def test_refs_median_rejects_an_outlier_reference():
    n = 5
    refs_pose = [make_pose([0.1 * i, 0.0, 0.9], rz(0.1 * i)) for i in range(n)]
    live = make_pose([0.3, -0.2, 0.9], rz(0.5))
    deltas = [live @ np.linalg.inv(p) for p in refs_pose]
    deltas[2] = make_pose([0.4, 0.0, 0.0], rz(1.5)) @ deltas[2]  # wrong registration
    perceiver = PerceivedPoses(None, None, {})
    perceiver.registration = _FakeRegistration(deltas)
    perceiver.refs["body"] = {
        "pose": np.stack(refs_pose),
        "rgb": np.zeros((n, 8, 8, 3), np.uint8),
        "depth_mm": np.zeros((n, 8, 8), np.uint16),
        "segmap": np.zeros((n, 8, 8), bool),
        "K": np.stack([np.eye(3)] * n),
    }
    perceiver.camera_meta = _camera()
    info = {}
    world = np.zeros((8, 8, 3))
    pose = perceiver._from_refs(
        "body", np.zeros((8, 8, 3), np.uint8), world, None, info
    )
    assert np.allclose(pose[:3, 3], live[:3, 3], atol=1e-9)
    assert np.allclose(pose[:3, :3], live[:3, :3], atol=1e-9)
    assert info["refs_ok"] == n and info["refs_failed"] == 0


class _Seg:
    def segment(self, rgb, point, min_score):
        mask = np.zeros(rgb.shape[:2], bool)
        r, c = point
        mask[max(0, r - 3) : r + 4, max(0, c - 3) : c + 4] = True
        return type("R", (), {"found": True, "mask": mask, "score": 1.0})()


class _Locator:
    def __init__(self, pixel):
        self.pixel = pixel
        self.calls = []

    def locate(self, images, categories):
        self.calls.append({"categories": list(categories)})
        return {cat: {"agentview": [self.pixel]} for cat in categories}


def test_located_poses_locates_once_then_tracks():
    from robots.libero.mt3.perception import BodySpec, LocatedPoses

    meta = _camera()
    depth = np.full((64, 64), 1.0)
    world = _world_from_depth(depth, meta)
    pixel = [32, 32]
    spot = world[32, 32]
    live = make_pose(spot, rz(0.3))
    spec = BodySpec(category="box", expect=list(spot), rotation=[1.0, 0.0, 0.0, 0.0])
    locator = _Locator(pixel)
    perceiver = LocatedPoses(
        None, _Seg(), {"box": spec}, locator, ["agentview"], size=64
    )
    ref = make_pose(spot + [0.1, 0.0, 0.0], np.eye(3))
    perceiver.refs["box"] = {
        "pose": np.stack([ref]),
        "rgb": np.zeros((1, 64, 64, 3), np.uint8),
        "depth_mm": np.zeros((1, 64, 64), np.uint16),
        "segmap": np.zeros((1, 64, 64), bool),
        "K": np.stack([np.eye(3)]),
        "T_WC": np.stack([np.asarray(meta["extrinsic_cam2world"])]),
        "camera": np.array(["agentview"]),
    }
    perceiver.registration = _FakeRegistration(iter([live @ np.linalg.inv(ref)] * 2))
    rgb = np.zeros((64, 64, 3), np.uint8)
    depth_mm = np.full((64, 64), 1000, np.uint16)
    perceiver.views = lambda: {"agentview": (rgb, world, meta, depth_mm)}

    first = perceiver(["box"])["box"]
    assert np.allclose(first[:3, 3], spot, atol=1e-9)
    # Not yaw_free: the nominal rotation, not the registered yaw.
    assert np.allclose(first[:3, :3], np.eye(3))
    second = perceiver(["box"])["box"]
    assert np.allclose(second, first)
    assert len(locator.calls) == 1
    assert [e["source"] for e in perceiver.log] == ["located", "tracked"]
    assert perceiver.log[0]["found"] and perceiver.log[0]["refs_ok"] == 1
    assert perceiver.log[1].get("reused")


def test_instances_merge_cameras_by_table_plane_distance():
    from robots.libero.mt3.astra import assign, instances

    front = np.zeros((4, 4, 3))
    wrist = np.zeros((4, 4, 3))
    front[0, 0] = [0.0, 0.0, 0.95]  # side of pot A
    front[1, 1] = [0.0, 0.40, 0.95]  # side of pot B
    wrist[2, 2] = [0.02, 0.41, 1.05]  # lid of pot B, 10 cm higher
    wrist[3, 3] = [0.01, 0.01, 1.05]  # lid of pot A
    found = instances(
        {"agentview": [[0, 0], [1, 1]], "wrist": [[2, 2], [3, 3]]},
        {"agentview": front, "wrist": wrist},
    )
    assert len(found) == 2
    assert found[0]["pixels"] == {"agentview": [0, 0], "wrist": [3, 3]}
    assert found[1]["pixels"] == {"agentview": [1, 1], "wrist": [2, 2]}
    chosen = assign(["b", "a"], found, {"a": [0.0, 0.0, 0.9], "b": [0.0, 0.4, 0.9]})
    assert chosen["a"] is found[0] and chosen["b"] is found[1]
