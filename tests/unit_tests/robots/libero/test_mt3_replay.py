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

import time
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.env_server import _configure_fixture_seed, _seed_fixture_sampler
from robots.libero.mt3.core import (
    POS_SCALE,
    EnvHooks,
    MT3Config,
    MT3Runner,
    grasp_tilt_recorded,
    held_object_eef_target,
    offset_recorded,
    segment_bottleneck,
    tilt_about,
)
from robots.libero.mt3.feasibility import SceneChecker, _rrt_connect, _shortcut_path


def test_cpgenfeas_flags_preserve_legacy_defaults():
    cfg = MT3Config()
    assert not cfg.cpgen_grasp_strict_ik
    assert not cfg.cpgen_grasp_local_check
    assert not cfg.cpgen_grasp_contact_tolerance
    assert not cfg.cpgen_grasp_rrt_fallback


def test_cpgen_strict_ik_rejects_intermediate_and_goal_errors():
    from robots.libero.mt3.core import _cpgen_ik_valid

    assert _cpgen_ik_valid(
        {"first_ik_fail": None, "goal_ik_pos_err_cm": 0.9, "goal_ik_rot_err_deg": 4.0},
        0.01, 5.0,
    )
    assert not _cpgen_ik_valid(
        {"first_ik_fail": 3, "goal_ik_pos_err_cm": 0.5, "goal_ik_rot_err_deg": 1.0},
        0.01, 5.0,
    )
    assert not _cpgen_ik_valid(
        {"first_ik_fail": None, "goal_ik_pos_err_cm": 1.1, "goal_ik_rot_err_deg": 1.0},
        0.01, 5.0,
    )


def test_cpgen_local_route_excludes_replaced_transit():
    from robots.libero.mt3.core import _cpgen_local_path

    goal, contact, lift = object(), [2], [3]
    assert _cpgen_local_path(goal, contact + lift) == [goal, 2, 3]
    assert _cpgen_local_path(goal, []) == [goal]


def test_cpgen_contact_tolerance_is_limited_to_fingers_and_support():
    from robots.libero.mt3.feasibility import _contact_exceeds_tolerance

    assert not _contact_exceeds_tolerance("gripper0_left_finger", 0.019, True, 0.02, 0.005)
    assert _contact_exceeds_tolerance("robot0_link6", 0.006, True, 0.02, 0.005)
    assert _contact_exceeds_tolerance("gripper0_left_finger", 0.006, False, 0.02, 0.005)
    assert _contact_exceeds_tolerance("gripper0_left_finger", 0.021, True, 0.02, 0.005)


def test_cpgen_rrt_fallback_order_is_bounded_and_zero_first():
    from robots.libero.mt3.core import _cpgen_rrt_order

    candidates = [
        {"tilt": 55, "min_joint_margin_deg": 40},
        {"tilt": -5, "min_joint_margin_deg": 30},
        {"tilt": 0, "min_joint_margin_deg": 20},
    ]
    ordered = _cpgen_rrt_order(candidates, 2)
    assert [row["tilt"] for row in ordered] == [0, -5]
    assert len(_cpgen_rrt_order(candidates, 0)) == 1
def test_held_object_retarget_places_object_on_demo_relative_pose():
    ref_target = np.eye(4)
    ref_target[:3, 3] = [0.4, -0.2, 0.3]
    obj_rel = np.eye(4)
    obj_rel[:3, 3] = [0.1, 0.0, 0.05]
    in_hand = np.eye(4)
    in_hand[:3, 3] = [0.0, 0.08, 0.02]
    eef = held_object_eef_target(ref_target, obj_rel, in_hand)
    np.testing.assert_allclose(eef @ in_hand, ref_target @ obj_rel)
def test_native_rpent_policy_preserves_seed_method(monkeypatch):
    """Benchmark evaluation must leave RPent's native seed behavior intact."""
    monkeypatch.setenv("RPENT_MT3_FIXTURE_SEED_POLICY", "rpent")
    calls = []
    seed = calls.append
    env = SimpleNamespace(seed=seed)
    assert _configure_fixture_seed(env) is env
    assert env.seed is seed
    env.seed(9)
    assert calls == [9]


def test_estimated_planning_state_ignores_live_scene_state():
    """Strict planning starts from model defaults + supplied estimates/proprioception."""
    import mujoco

    joints = "".join(
        f'<body name="robot0_link{k}" pos="0 0 .1"><joint name="robot0_joint{k}" range="-2 2"/><geom type="sphere" size=".01"/>'
        + ('<site name="gripper0_right_grip_site" pos="0 0 .01"/>' if k == 7 else "")
        + '</body>'
        for k in range(1, 8)
    )
    xml = f'<mujoco><worldbody><body name="robot0_base">{joints}</body><body name="white_yellow_mug_1_main"><freejoint/><geom type="sphere" size=".02"/></body></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    live = mujoco.MjData(model)
    estimates = {"white_yellow_mug_1_main": [0.4, 0.2, 0.1, 1, 0, 0, 0]}
    q = np.linspace(-0.3, 0.3, 7)
    checker = SceneChecker(model, live, scene_estimates=estimates, robot_joints=q)
    live.qpos[:] = 0.7
    checker._reset_copy()
    first = checker.data.qpos.copy()
    live.qpos[:] = -0.8
    live.qvel[:] = 1.0
    checker._reset_copy()
    np.testing.assert_array_equal(checker.data.qpos, first)
    np.testing.assert_array_equal(checker.data.qpos[checker.arm_qpos], q)


def test_strict_scene_moves_unestimated_free_bodies_out_and_keeps_fixtures():
    import mujoco

    joints = "".join(
        f'<body name="robot0_link{k}" pos="0 0 .1"><joint name="robot0_joint{k}" range="-2 2"/><geom type="sphere" size=".01"/>'
        + ('<site name="gripper0_right_grip_site" pos="0 0 .01"/>' if k == 7 else "")
        + "</body>"
        for k in range(1, 8)
    )
    xml = f'''<mujoco><worldbody>
      <body name="robot0_base">{joints}</body>
      <body name="estimated_obj"><freejoint/><geom type="sphere" size=".02"/></body>
      <body name="unknown_obj"><freejoint/><geom type="sphere" size=".02"/></body>
      <body name="fixture"><geom type="box" size=".1 .1 .1"/></body>
    </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    checker = SceneChecker(
        model, mujoco.MjData(model),
        scene_estimates={"estimated_obj": [0.3, 0.2, 0.1, 1, 0, 0, 0]},
        robot_joints=np.zeros(7),
    )
    checker._reset_copy()
    unknown = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "unknown_obj")
    fixture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fixture")
    estimated = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "estimated_obj")
    assert checker.data.xpos[unknown, 0] > 900
    np.testing.assert_allclose(checker.data.xpos[fixture], model.body_pos[fixture])
    np.testing.assert_allclose(checker.data.xpos[estimated], [0.3, 0.2, 0.1])


def test_strict_checker_attaches_estimated_object_to_grip_site():
    import mujoco

    joints = "".join(
        f'<body name="robot0_link{k}" pos="0 0 .1"><joint name="robot0_joint{k}" range="-2 2"/><geom type="sphere" size=".01"/>'
        + ('<site name="gripper0_right_grip_site" pos="0 0 .01"/>' if k == 7 else "")
        + "</body>"
        for k in range(1, 8)
    )
    xml = (
        f'<mujoco><worldbody><body name="robot0_base">{joints}</body>'
        '<body name="white_yellow_mug_1_main"><freejoint/><geom type="sphere" size=".02"/></body></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    checker = SceneChecker(
        model,
        data,
        scene_estimates={"white_yellow_mug_1_main": [0.4, 0.2, 0.1, 1, 0, 0, 0]},
        robot_joints=np.zeros(7),
        attached_body="white_yellow_mug_1_main",
        attached_offset=np.eye(4),
    )
    checker._reset_copy()
    held = checker.attached_body
    checker._set_held_pose(held, checker.attached_offset)
    np.testing.assert_allclose(checker.data.xpos[held], checker.data.site_xpos[checker.site], atol=1e-12)


def test_grasp_tilt_recorded_metadata_supports_per_demo_flags():
    lib = SimpleNamespace(meta={"grasp_tilt_recorded": [False, True]})
    assert not grasp_tilt_recorded(lib, 0)
    assert grasp_tilt_recorded(lib, 1)
    assert not grasp_tilt_recorded(lib, 2)


def test_grasp_dy_tilt_uses_start_pose_bands():
    # The scene-start dy is 0.27; the upper adjacent band is selected.
    poses = {
        "white_yellow_mug_1_main": np.eye(4),
        "microwave_1_main": np.eye(4),
    }
    poses["white_yellow_mug_1_main"][1, 3] = 0.27
    hooks = EnvHooks(
        lambda _: None,
        lambda: np.eye(4),
        lambda bodies: {body: poses[body] for body in bodies},
        lambda: False,
        start_pose=poses,
    )
    cfg = MT3Config(grasp_tilt_by_dy={"0.26:0.30": -15.0})
    lib = SimpleNamespace(
        frames=lambda _: slice(0, 1),
        eef=np.eye(4)[None],
        target=np.eye(4)[None],
        gripper=np.zeros(1),
        actions=np.zeros((1, 7)),
        segments=[[{"ref": "white_yellow_mug_1_main", "start": 0, "c": 0, "end": 0,
                    "events": [["grasp", 0, "white_yellow_mug_1_main"]]}]],
        pose={"white_yellow_mug_1_main": np.eye(4)[None]},
        meta={},
    )
    runner = MT3Runner(lib, 0, cfg, hooks)
    log = runner._segment(0)
    assert log["tilt"] == -15.0


def test_reference_specific_alignment_replans_preserve_default():
    lib = SimpleNamespace(
        frames=lambda _: slice(0, 1),
        eef=np.eye(4)[None],
        target=np.eye(4)[None],
        gripper=np.zeros(1),
        actions=np.zeros((1, 7)),
        segments=[[{"ref": "microwave_1_microdoorroot"}]],
        pose={},
    )
    hooks = EnvHooks(lambda _: None, lambda: np.eye(4), lambda _: {}, lambda: False)
    default = MT3Runner(lib, 0, MT3Config(), hooks)
    configured = MT3Runner(
        lib,
        0,
        MT3Config(alignment_replans_by_ref={"microdoorroot": 2}),
        hooks,
    )
    assert default._alignment_replans("microwave_1_microdoorroot") == 0
    assert default._alignment_replans("white_yellow_mug_1_main") == 0
    assert configured._alignment_replans("microwave_1_microdoorroot") == 2
    assert configured._alignment_replans("white_yellow_mug_1_main") == 0


def test_reference_specific_segment_overshoot_follows_last_five_target_frames():
    frames = 10
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = np.arange(frames) * 0.01
    poses[:, 1, 3] = np.arange(frames) * 0.002
    segments = [[{"ref": "microwave_1_microdoorroot", "start": 0, "c": 5, "end": 10}]]
    lib = SimpleNamespace(
        frames=lambda _: slice(0, frames),
        eef=poses,
        target=poses.copy(),
        gripper=np.arange(frames, dtype=float),
        actions=np.zeros((frames, 7)),
        segments=segments,
        pose={"microwave_1_microdoorroot": np.repeat(np.eye(4)[None], frames, axis=0)},
    )
    current = poses[0].copy()
    commanded = []

    def step(action):
        current[:3, 3] += action[:3] * POS_SCALE
        commanded.append(action.copy())

    hooks = EnvHooks(
        step,
        lambda: current.copy(),
        lambda names: {name: np.eye(4) for name in names},
        lambda: False,
    )
    runner = MT3Runner(
        lib,
        0,
        MT3Config(
            radius=0.0, settle_max=0, segment_overshoot_by_ref={"microdoorroot": 0.025}
        ),
        hooks,
    )
    result = runner.run_segment(0)

    # The transferred final target is x=.09/y=.018. The tail displacement
    # (frames 5..9) points along +x/+y; the extension ends 2.5 cm farther.
    expected_direction = np.array([0.01, 0.002, 0.0]) / np.linalg.norm([0.01, 0.002])
    last_target = poses[9, :3, 3]
    end_goal = last_target + expected_direction * 0.025
    np.testing.assert_allclose(current[:3, 3], end_goal, atol=1e-7)
    assert len(commanded) >= 13  # source frames, alignment, and 3 overshoot steps
    assert commanded[-1][6] == lib.gripper[9]
    assert commanded[-2][6] == lib.gripper[9]
    assert result["overshoot_m"] == 0.025


def test_fixture_placement_repeats_after_reseeding():
    """Both placement samplers and the wrapped environment receive the seed."""
    state = np.random.get_state()
    try:
        received = []
        env = _seed_fixture_sampler(SimpleNamespace(seed=received.append))
        env.seed(17)
        first = np.random.uniform(size=12)
        np.random.uniform(size=29)
        env.seed(17)
        np.testing.assert_array_equal(first, np.random.uniform(size=12))
        assert received == [17, 17]
    finally:
        np.random.set_state(state)


def test_continuous_replay_keeps_transit_frames_and_blends_reference_changes():
    """All source frames run, including transit between independent objects."""
    frames = 24
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = np.arange(frames) * 0.001
    segments = [
        {"ref": "a", "start": 0, "c": 8, "end": 12},
        {"ref": "b", "start": 12, "c": 21, "end": 24},
    ]
    lib = SimpleNamespace(
        frames=lambda _: slice(0, frames),
        eef=poses,
        target=poses.copy(),
        gripper=np.full(frames, -1.0),
        actions=np.zeros((frames, 7)),
        segments=[segments],
        pose={name: np.repeat(np.eye(4)[None], frames, axis=0) for name in ("a", "b")},
    )
    live = {"a": np.eye(4), "b": np.eye(4)}
    live["b"][1, 3] = 0.02
    current = poses[0].copy()
    achieved = []

    def step(action):
        current[:3, 3] += action[:3] * POS_SCALE
        achieved.append(current.copy())

    hooks = EnvHooks(
        step,
        lambda: current.copy(),
        lambda names: {k: live[k] for k in names},
        lambda: False,
    )
    runner = MT3Runner(lib, 0, MT3Config(mode="continuous", radius=0.004), hooks)
    for k in range(2):
        runner.run_segment(k)
    assert len(achieved) == frames
    np.testing.assert_allclose(np.array(achieved)[:12], poses[:12], atol=1e-12)
    assert achieved[12][1, 3] == 0.0
    assert 0.0 < achieved[14][1, 3] < 0.02
    assert achieved[21][1, 3] == 0.02
    assert all(log["align_steps"] == 0 for log in runner.logs)


def test_stored_bottleneck_overrides_the_radius_rule():
    """A generated demo resumes from the frame its interaction started at rest."""
    eef = np.tile(np.eye(4), (6, 1, 1))
    eef[:, 0, 3] = [0.0, 0.05, 0.11, 0.11, 0.11, 0.2]
    seg = {"ref": "obj", "start": 0, "c": 5, "end": 6}
    assert segment_bottleneck(eef, seg, 0.1) == 1
    assert segment_bottleneck(eef, {**seg, "b": 4}, 0.1) == 4


def test_blocked_alignment_does_not_execute_source_grasp():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = 0.2
    lib = SimpleNamespace(
        frames=lambda _: slice(0, 4),
        eef=poses,
        target=poses.copy(),
        gripper=np.array([-1.0, -1.0, 1.0, 1.0]),
        actions=np.zeros((4, 7)),
        segments=[[{"ref": "object", "start": 0, "c": 2, "end": 4}]],
        pose={"object": np.repeat(np.eye(4)[None], 4, axis=0)},
    )
    actions = []
    hooks = EnvHooks(
        actions.append,
        lambda: np.eye(4),
        lambda names: {name: np.eye(4) for name in names},
        lambda: False,
    )
    runner = MT3Runner(lib, 0, MT3Config(settle_max=0, alignment_guard_pos=0.02), hooks)
    result = runner.run_segment(0)
    assert result["blocked"]
    assert all(action[6] == -1 for action in actions)


def test_rrt_connect_routes_around_joint_space_wall_and_shortcuts_safely():
    """A box obstacle blocks the direct edge in a 2-D toy configuration space."""
    lo, hi = np.zeros(2), np.ones(2)

    def valid(q):
        return not (0.42 <= q[0] <= 0.58 and 0.0 <= q[1] <= 0.72)

    start, goal = np.array([0.15, 0.2]), np.array([0.85, 0.2])
    assert not all(valid(start + (goal - start) * t) for t in np.linspace(0, 1, 50))
    path, iterations = _rrt_connect(
        start,
        goal,
        valid,
        lo,
        hi,
        np.random.default_rng(12),
        max_iterations=3000,
        timeout=5.0,
        started=time.monotonic(),
        step_size=0.16,
        edge_step=0.02,
    )
    assert path is not None
    assert iterations > 0
    assert np.allclose(path[0], start)
    assert np.allclose(path[-1], goal)

    def edge_valid(a, b):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / 0.02)))
        return all(valid(a + (b - a) * i / count) for i in range(1, count + 1))

    smooth = _shortcut_path(
        path,
        valid,
        np.random.default_rng(9),
        attempts=100,
        edge_step=0.02,
    )
    assert all(edge_valid(a, b) for a, b in zip(smooth, smooth[1:]))


def test_minimal_route_uses_rrt_path_when_candidates_fail():
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.03, 4)
    calls = []
    planned_path = [np.eye(4), np.eye(4)]
    planned_path[1] = np.eye(4)
    planned_path[1][1, 3] = 0.01
    lib = SimpleNamespace(
        frames=lambda _: slice(0, 4),
        eef=poses,
        target=poses.copy(),
        gripper=np.full(4, -1.0),
        actions=np.zeros((4, 7)),
        segments=[[{"ref": "object", "start": 0, "c": 2, "end": 4}]],
        pose={"object": np.repeat(np.eye(4)[None], 4, axis=0)},
        ref_bodies=["object"],
        ranked=lambda *_: [],
    )
    current = np.eye(4)
    hooks = EnvHooks(
        lambda action: calls.append(action),
        lambda: current.copy(),
        lambda names: {name: np.eye(4) for name in names},
        lambda: False,
        check_paths=lambda paths, options=None: [{"feasible": False} for _ in paths],
        plan_path=lambda goal: {
            "success": True,
            "path": planned_path,
            "iterations": 3,
            "time_s": 0.01,
            "path_length": 0.5,
            "movable_contact_count": 0,
        },
    )
    cfg = MT3Config(feasibility_check="minimal", rrt_fallback=True, settle_max=0)
    runner = MT3Runner(lib, 0, cfg, hooks)
    result = runner.run_segment(0)
    assert result["selection"]["stage"] == "rrt"
    assert result["selection"]["iterations"] == 3
    assert result["align_steps"] == len(planned_path)
    assert len(calls) > len(planned_path)


def _pick_place_library(frames: int = 8) -> SimpleNamespace:
    """Grasp ``obj`` (frames 0-3), then release it at ``box`` (frames 4-7)."""
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.07, frames)
    segments = [
        {"ref": "obj", "start": 0, "c": 2, "end": 4, "events": [["grasp", 2, "obj"]]},
        {"ref": "box", "start": 4, "c": 6, "end": 8, "events": [["release", 6, "box"]]},
    ]
    return SimpleNamespace(
        frames=lambda _: slice(0, frames),
        eef=poses,
        target=poses.copy(),
        gripper=np.array([-1.0, -1.0] + [1.0] * (frames - 2)),
        actions=np.zeros((frames, 7)),
        segments=[segments],
        # The demo holds ``obj`` exactly at the grip site.
        pose={"obj": poses.copy(), "box": np.repeat(np.eye(4)[None], frames, axis=0)},
        ref_bodies=["obj", "box"],
        ranked=lambda *_: [],
    )


def test_compensated_placement_routes_are_checked_as_executed():
    """Route checks shift the placement by the measured in-hand offset."""
    lib = _pick_place_library()
    hand = np.eye(4)
    held = np.eye(4)
    held[0, 3] = 0.01  # the object sits 1 cm ahead of the grip site
    checked = []
    hooks = EnvHooks(
        lambda action: None,
        lambda: hand.copy(),
        lambda names: {"obj": held, "box": np.eye(4)},
        lambda: False,
        check_paths=lambda paths, options=None: (
            checked.extend(paths) or [{"feasible": True} for _ in paths]
        ),
    )
    cfg = MT3Config(radius=0.0, grasp_compensation=True)
    runner = MT3Runner(lib, 0, cfg, hooks)
    runner._feasibility(1, np.eye(4), 1.0, [0], (0.0,), (0.0,))
    b = segment_bottleneck(lib.eef, lib.segments[0][1], cfg.radius)
    np.testing.assert_allclose(
        checked[0][-1][:3, 3], lib.eef[b][:3, 3] - [0.01, 0.0, 0.0], atol=1e-12
    )


def test_grasp_yaw_turns_about_the_configured_body_point():
    lib = _pick_place_library()
    hooks = EnvHooks(
        lambda action: None,
        lambda: np.eye(4),
        lambda names: {name: np.eye(4) for name in names},
        lambda: False,
    )
    live = np.eye(4)
    live[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cfg = MT3Config(yaw_pivot={"obj": [-0.02, 0.0, 0.0]})
    runner = MT3Runner(lib, 0, cfg, hooks)
    np.testing.assert_allclose(
        runner._pivot(0, 0, np.eye(4), live), [0.0, -0.02, 0.0], atol=1e-12
    )
    assert np.allclose(
        MT3Runner(lib, 0, MT3Config(), hooks)._pivot(0, 0, np.eye(4), live), 0
    )


def test_grasp_tilt_keeps_the_pinch_and_swings_the_approach():
    grip = np.eye(4)
    grip[:3, 3] = [0.3, 0.1, 0.9]
    transform = np.eye(4)
    transform[:3, 3] = [0.05, 0.0, 0.0]
    turned = tilt_about(transform, np.radians(30), grip, axis=0) @ grip
    before = transform @ grip
    np.testing.assert_allclose(turned[:3, 3], before[:3, 3], atol=1e-12)
    np.testing.assert_allclose(turned[:3, 0], before[:3, 0], atol=1e-12)
    assert np.degrees(np.arccos(turned[:3, 2] @ before[:3, 2])) == pytest.approx(30)


def test_reference_offset_shifts_transfers_in_the_reference_frame():
    live = np.eye(4)
    live[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    live[:3, 3] = [0.5, 0.0, 0.9]
    demo = np.eye(4)
    cfg = MT3Config(ref_offset={"microwave_1_main": [0.0, 0.04, 0.0]})
    shifted, _ = cfg.transfer("microwave_1_main", live, demo)
    plain, _ = cfg.transfer("microwave_1_microdoorroot", live, demo)
    np.testing.assert_allclose(shifted[:3, 3] - plain[:3, 3], [-0.04, 0.0, 0.0])
    np.testing.assert_allclose(shifted[:3, :3], plain[:3, :3])


def test_reference_offset_skips_demos_recorded_with_it():
    """A demo recorded with ``ref_offset`` applied is not shifted a second time."""
    live = np.eye(4)
    live[:3, 3] = [0.5, 0.0, 0.9]
    cfg = MT3Config(ref_offset={"microwave_1_main": [0.0, 0.04, 0.0]})
    shifted, _ = cfg.transfer("microwave_1_main", live, np.eye(4))
    recorded, _ = cfg.transfer("microwave_1_main", live, np.eye(4), recorded=True)
    np.testing.assert_allclose(shifted[:3, 3] - recorded[:3, 3], [0.0, 0.04, 0.0])
    lib = SimpleNamespace(meta={"placement_offset_recorded": [False, True]})
    assert not offset_recorded(lib, 0)
    assert offset_recorded(lib, 1)
    assert not offset_recorded(SimpleNamespace(meta={}), 0)


def test_microwave_insert_skill_is_config_gated_and_uses_start_pose_estimates():
    frames = 8
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.07, frames)
    mug = np.eye(4)
    mug[2, 3] = 0.9
    microwave = np.eye(4)
    microwave[:3, 3] = [0.2, -0.3, 0.92]
    segments = [
        {
            "ref": "white_yellow_mug_1_main",
            "start": 0,
            "c": 2,
            "end": 4,
            "events": [["grasp", 2, "white_yellow_mug_1"]],
        },
        {
            "ref": "microwave_1_main",
            "start": 4,
            "c": 6,
            "end": 8,
            "events": [["release", 6, "microwave_1"]],
        },
    ]
    lib = SimpleNamespace(
        frames=lambda _: slice(0, frames),
        eef=poses,
        target=poses.copy(),
        # Segment event metadata is the robust held-object signal; the stored
        # gripper sample at c varies across library demos.
        gripper=np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0]),
        actions=np.zeros((frames, 7)),
        segments=[segments],
        pose={
            "white_yellow_mug_1_main": np.repeat(mug[None], frames, axis=0),
            "microwave_1_main": np.repeat(microwave[None], frames, axis=0),
        },
    )
    current = poses[0].copy()
    commands = []

    def step(action):
        commands.append(action.copy())
        current[:3, 3] += action[:3] * POS_SCALE

    hooks = EnvHooks(
        step,
        lambda: current.copy(),
        lambda names: {
            "white_yellow_mug_1_main": mug.copy(),
            "microwave_1_main": microwave.copy(),
        },
        lambda: False,
    )
    disabled = MT3Runner(lib, 0, MT3Config(), hooks)
    assert disabled.cfg.placement_skill_by_ref == {}
    enabled = MT3Runner(
        lib,
        0,
        MT3Config(placement_skill_by_ref={"microwave_1_main": {"target_y": 0.025}}),
        hooks,
    )
    enabled._start_poses = {
        "white_yellow_mug_1_main": mug.copy(),
        "microwave_1_main": microwave.copy(),
    }
    assert enabled._segment_is_placement(1)
    enabled._dispatch = lambda _: {}
    enabled.run_segment(0)
    np.testing.assert_array_equal(enabled._mug_start, mug)
    enabled._grasp_eef = np.eye(4)
    enabled._in_hand = np.linalg.inv(enabled._grasp_eef) @ mug
    enabled._last_gripper = 1.0
    log = enabled._placement_skill(
        1, enabled.cfg.placement_skill_by_ref["microwave_1_main"]
    )
    assert log["skill"] == "microwave_insert"
    assert "episode_start_pose_estimates" in log["used_inputs"]
    assert len(commands) > 20
    assert commands[-1][6] == -1.0


def test_agent_replay_catalog_and_bounds(monkeypatch, tmp_path):
    from robots.libero.mt3 import agent

    class Library:
        keep = np.array([True])
        ref_bodies = ["bowl"]
        segments = [[{"ref": "bowl", "events": [["grasp", 3, "bowl"]]}]]

        def __init__(self, path):
            pass

        def __len__(self):
            return 1

        def nearest(self, scene, cfg):
            return 0, {"demo": 0}

    cfg = MT3Config(alignment_guard_pos=0.025)
    monkeypatch.setattr(agent, "load_config", lambda _: (cfg, tmp_path))
    monkeypatch.setattr(agent, "TaskLibrary", Library)
    monkeypatch.setattr(
        agent,
        "_hooks",
        lambda _: EnvHooks(
            lambda _: pytest.fail("catalog must not move the robot"),
            lambda: np.eye(4),
            lambda _: {},
            lambda: False,
        ),
    )
    tools = {}
    toolkit = SimpleNamespace(
        add_tool=lambda name, spec, handler: tools.update({name: handler}),
        _execute_primitive=lambda name, handler, **kwargs: handler(**kwargs),
    )
    agent.register_agent_tools(toolkit, "10_swap_t0_s0", str(tmp_path / "mt3.json"))
    catalog = tools["trajectory_catalog"]()
    assert catalog["nearest_demo"] == 0
    assert catalog["segments"][0]["reference"] == "bowl"
    with pytest.raises(ValueError, match="demo"):
        tools["trajectory_replay"](segment=0, demo=-1)
    with pytest.raises(ValueError, match="segment"):
        tools["trajectory_replay"](segment=-1, demo=0)
