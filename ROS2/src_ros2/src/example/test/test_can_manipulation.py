"""Offline regressions for motion planning and state boundaries."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from example.rgbd_function.can_manipulation import CanManipulation, MotionFailure
from example.rgbd_function import can_manipulation
from kinematics.forward_kinematics import ForwardKinematics


def planner():
    node = CanManipulation.__new__(CanManipulation)
    node.fk = ForwardKinematics(debug=False)
    node.contact_offset = .032
    node.pitch = 38.
    node.cancel = threading.Event()
    node.hardware_lock = threading.RLock()
    node.joint_positions = lambda: np.array([650, 811, 75, 92, 500, 200])
    return node


def test_contact_offset_follows_gripper_axis():
    node = planner()
    center = np.array([.175, .133, .159])
    tip = node.grasp_tip(center, 45.)
    assert np.isclose(np.linalg.norm(tip-center), .032)
    assert tip[2] < center[2]
    assert np.linalg.norm(tip[:2]) > np.linalg.norm(center[:2])


def test_complete_can_grasp_and_lift_path_is_reachable():
    node = planner()
    tip = node.grasp_tip(np.array([.165, .1374, .0709+.1336*.90]), 38.)
    hover = node.solve(tip+[0, 0, .027], 38., node.joint_positions())
    descend = node.cartesian_path(tip, 38., 200, start=np.r_[hover, 200])
    lift = node.cartesian_path(tip+[0, 0, .035], 38., 430, start=descend[-1])
    actual, _ = node.tool_pose(lift[-1])
    assert np.linalg.norm(actual-(tip+[0, 0, .035])) < .002
    for first, second in zip(descend+lift, (descend+lift)[1:]):
        assert np.max(np.abs(second[:5]-first[:5])) <= 90


def test_unreachable_goal_fails_before_motion():
    node = planner()
    with pytest.raises(MotionFailure, match='Unreachable'):
        node.solve(np.array([.8, 0, .1]), 45., node.joint_positions())


def test_nonfinite_goal_is_rejected():
    node = planner()
    with pytest.raises(MotionFailure, match='finite'):
        node.solve(np.array([np.nan, 0, .1]), 45., node.joint_positions())


def test_rest_does_not_release_an_unverified_grip():
    node = planner()
    node.held = None
    node.grip_closed = True
    with pytest.raises(MotionFailure, match='return'):
        node.rest()


def test_stop_prevents_new_motion_commands():
    node = planner()
    node.cancel.set()
    with pytest.raises(MotionFailure, match='canceled'):
        node.check_cancel()


def test_camera_transform_ignores_wrist_roll():
    node = planner()
    node.camera_offset = np.array([-.101, .010, .057])
    initial = node.joint_positions()
    rolled = initial.copy()
    rolled[4] = 850
    assert np.allclose(node.camera_pose(initial), node.camera_pose(rolled))


def test_out_of_model_feedback_returns_motion_error():
    node = planner()
    with pytest.raises(MotionFailure, match='kinematic model range'):
        node.tool_pose(np.array([331, 99, 393, 214, 401, 330]))


def test_snapshot_rejects_different_capture_times_despite_same_arrival(monkeypatch):
    node = planner()
    node.phase = 'survey'
    node.worker = threading.current_thread()
    node.frame_lock = threading.Lock()
    rgb = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0)))
    depth = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=101, nanosec=0)))
    node.frames = dict(rgb=(10., rgb), depth=(10., depth), info=(10., None))
    clock = [10.]
    monkeypatch.setattr(can_manipulation.time, 'monotonic', lambda: clock[0])
    node.pause = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
    with pytest.raises(MotionFailure, match='synchronized'):
        node.snapshot()


@pytest.mark.parametrize('contact_request', [275, 345, 400, 435])
def test_pick_and_place_preserve_detected_close_request(monkeypatch, contact_request):
    node = planner()
    node.phase, node.held, node.grip_closed = 'rest', None, False
    can = dict(position=[.165, .1374, .1377], bottom_z=.0709, support_z=.0709,
               height=.1336)
    joints = node.joint_positions().copy()
    commands = []

    def step(goal):
        joints[:] = goal
        commands.append((node.phase, int(goal[5])))
        joints[5] = min(int(goal[5]), contact_request-25)
        return joints.copy()

    node.joint_positions = lambda: joints.copy()
    node.step_joints = step
    node.move_joints = step
    node.survey = lambda: can
    node.feedback = lambda phase: setattr(node, 'phase', phase)
    node.event = lambda *args, **kwargs: None
    node.snapshot = lambda *args: dict(
        rgb_path='unused', intrinsics=[1, 1, 0, 0],
        camera_to_arm=np.eye(4).tolist(), can=can)
    node.rest = lambda: None
    monkeypatch.setattr(can_manipulation.cv2, 'imread', lambda path: np.zeros((1, 1, 3), np.uint8))
    monkeypatch.setattr(can_manipulation, 'verify_visual_lift',
                        lambda *args: dict(verified=True, displacement=[0., 0., .035]))

    node.pick()
    assert node.held['contact'] == contact_request-25
    assert node.held['jaw_command'] == contact_request
    lift_commands = [jaw for phase, jaw in commands if phase == 'test-lift']
    assert lift_commands and all(jaw == contact_request for jaw in lift_commands)

    open_gripper = node.open_gripper

    def release():
        node.feedback('opening')
        open_gripper()

    node.open_gripper = release
    commands.clear()
    node.place(dict(frame='arm_base', position=[.165, .1374, .0709]))
    place_commands = [jaw for phase, jaw in commands if phase == 'placing']
    assert place_commands and all(jaw == contact_request for jaw in place_commands)


@pytest.mark.parametrize('support_offset', [0., .03])
def test_placement_verifies_independent_support_height(support_offset):
    node = planner()
    node.held = dict(pitch=38., jaw_command=345, can=dict(height=.134),
                     can_in_tool=[0., 0., 0.])
    node.solve = lambda *args: np.full(5, 500)
    node.tool_pose = lambda *args: (np.zeros(3), np.eye(3))
    node.cartesian_path = lambda destination, pitch, jaw, start: [np.r_[start[:5], jaw]]
    node.move_joints = lambda *args: None
    node.follow = lambda *args: None
    node.open_gripper = lambda: None
    node.feedback = lambda *args: None
    rests = []
    node.rest = lambda: rests.append(True)

    def snapshot(label, expected_xy, support_z=None):
        assert support_z is None, 'Placement must independently measure the support plane'
        return dict(can=dict(position=[.165, .1374, .138],
                             support_z=.071+support_offset))

    node.snapshot = snapshot
    goal = dict(frame='arm_base', position=[.165, .1374, .071])
    if support_offset:
        with pytest.raises(MotionFailure, match='placement could not be verified'):
            node.place(goal)
        assert not rests
    else:
        node.place(goal)
        assert rests
