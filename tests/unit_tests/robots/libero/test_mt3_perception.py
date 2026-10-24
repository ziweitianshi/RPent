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

    def transfer(self, demo, live, T_WC, pivot):
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
