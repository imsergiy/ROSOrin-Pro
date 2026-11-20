"""Offline regressions for motion planning and state boundaries."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from example.rgbd_function.can_manipulation import (
    CanManipulation, FOLDED_ENTRY, MotionFailure, REST, SERVO_IDS)
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
    monkeypatch.setattr(can_manipulation.time, 'sleep',
                        lambda seconds: clock.__setitem__(0, clock[0]+seconds))
    with pytest.raises(MotionFailure, match='synchronized'):
        node.snapshot()


@pytest.mark.parametrize('at_home', [False, True])
def test_stopped_arm_can_still_supply_onboard_observation(monkeypatch, tmp_path, at_home):
    node = planner()
    if at_home:
        node.joint_positions = lambda: REST.copy()
    node.phase, node.worker, node.run = 'cooling', None, None
    node.cancel.set()
    node.frame_lock = threading.Lock()
    node.root = tmp_path
    node.radius, node.height = .025, .135
    node.camera_offset = np.array([-.101, .010, .057])
    node.observations = []
    stamp = SimpleNamespace(sec=100, nanosec=0)
    image = SimpleNamespace(header=SimpleNamespace(stamp=stamp))
    info = SimpleNamespace(k=[80., 0., 32., 0., 80., 24., 0., 0., 1.])
    now = can_manipulation.time.monotonic()
    node.frames = dict(rgb=(now, image), depth=(now, image), info=(now, info))
    bgr = np.zeros((48, 64, 3), np.uint8)
    depth = np.full((48, 64), 500, np.uint16)
    depth[0, 0] = 0
    node.bridge = SimpleNamespace(
        imgmsg_to_cv2=lambda msg, desired_encoding: bgr if desired_encoding == 'bgr8' else depth)
    node.event = lambda *args, **kwargs: None
    monkeypatch.setattr(can_manipulation.time, 'sleep', lambda seconds: None)
    result = node.snapshot()
    assert result['rgb_path'] and result['depth_path']
    assert result['joints'] == node.joint_positions().tolist()
    assert len(list(tmp_path.glob('*.jpg'))) == 1
    assert len(list(tmp_path.glob('*-depth.png'))) == 1
    saved_depth = can_manipulation.cv2.imread(result['depth_path'], -1)
    assert saved_depth[1, 1]*result['depth_scale_m'] == .5
    assert saved_depth[0, 0] == result['depth_invalid_value'] == 0
    if at_home:
        assert 'camera_to_arm' not in result
        assert 'kinematic model range' in result['detection_error']
    else:
        assert 'camera_to_arm' in result
    assert node.cancel.is_set()


@pytest.mark.parametrize('contact_request', [275, 345, 400, 435])
@pytest.mark.parametrize('verification_case', ['valid', 'canceled', 'missing_before'])
def test_pick_and_place_preserve_detected_close_request(
        monkeypatch, contact_request, verification_case):
    node = planner()
    node.phase, node.held, node.grip_closed = 'rest', None, False
    can = dict(position=[.165, .1374, .1377], bottom_z=.0709, top_z=.2045, support_z=.0709,
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

    def snapshot(*args):
        image = dict(rgb_path='unused', intrinsics=[1, 1, 0, 0], can=can)
        if verification_case != 'missing_before':
            image['camera_to_arm'] = np.eye(4).tolist()
        return image

    node.snapshot = snapshot
    node.rest = lambda: None
    monkeypatch.setattr(can_manipulation.cv2, 'imread', lambda path: np.zeros((1, 1, 3), np.uint8))

    def verify(*args):
        if verification_case == 'canceled':
            node.cancel.set()
            node.phase = 'cooling'
        return dict(verified=True, displacement=[0., 0., .035])

    monkeypatch.setattr(can_manipulation, 'verify_visual_lift', verify)

    if verification_case == 'missing_before':
        with pytest.raises(MotionFailure, match='camera transform'):
            node.pick()
        assert node.held is None and not node.grip_closed
        assert not any(phase == 'test-lift' for phase, jaw in commands)
        return
    if verification_case == 'canceled':
        with pytest.raises(MotionFailure, match='canceled'):
            node.pick()
        assert node.held is None and node.phase == 'cooling'
        return
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


@pytest.mark.parametrize('support_offset,rim_offset', [(0., 0.), (.03, 0.), (0., .02)])
def test_placement_verifies_independent_support_height(support_offset, rim_offset):
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
                             top_z=.071+support_offset+.134+rim_offset,
                             support_z=.071+support_offset))

    node.snapshot = snapshot
    goal = dict(frame='arm_base', position=[.165, .1374, .071])
    if support_offset or rim_offset:
        with pytest.raises(MotionFailure, match='placement could not be verified'):
            node.place(goal)
        assert not rests
    else:
        node.place(goal)
        assert rests


def test_recovery_uses_measured_contact_joints_without_new_ik():
    node = planner()
    node.grip_closed = True
    node.contact_tip = np.array([.18, .15, .17])
    node.contact_joints = np.array([660, 455, 198, 302, 500, 200])
    calls = []
    node.move_joints = lambda goal: calls.append(goal.tolist())
    node.open_gripper = lambda: calls.append('open')
    node.feedback = lambda phase: None
    node.recover_source()
    assert calls == [[660, 455, 198, 302, 500, 200], 'open']
    assert not node.grip_closed


def monitor_node(temperatures):
    node = planner()
    node.phase = 'held'
    node.held = dict(since=0.)
    node.grip_closed = True
    node.worker = None
    node.sensors = lambda: dict(temperature_c=[[value] for value in temperatures])
    node.event = lambda *args, **kwargs: None
    node.feedback = lambda phase: setattr(node, 'phase', phase)
    return node


def test_canceled_hold_cannot_restart_automatically():
    node = monitor_node([55, 55, 55, 55, 55, 60])
    node.cancel.set()
    calls = []
    node.return_can = lambda *args: calls.append('return')
    node.monitor_hold()
    assert not calls


@pytest.mark.parametrize('temperatures,jaw_off', [
    ([28, 66, 30, 30, 30, 30], False), ([28, 28, 30, 30, 30, 76], True)])
def test_critical_temperature_removes_power_during_unverified_motion(temperatures, jaw_off):
    node = monitor_node(temperatures)
    node.phase = 'test-lift'
    node.held = None
    node.worker = SimpleNamespace(is_alive=lambda: True)
    commands = []
    node.board = SimpleNamespace(
        bus_servo_stop=lambda ids: commands.append(('stop', ids)),
        bus_servo_enable_torque=lambda servo, enabled: commands.append((servo, enabled)))
    node.monitor_hold()
    assert node.cancel.is_set()
    assert node.phase == 'cooling'
    assert commands[0] == ('stop', [1, 2, 3, 4, 5, 10])
    assert all((servo, False) in commands for servo in (1, 2, 3, 4, 5))
    assert ((10, False) in commands) == jaw_off


def test_warm_recovery_can_unload_below_critical_temperature():
    node = monitor_node([28, 64, 30, 30, 30, 30])
    node.sensors = lambda: dict(temperature_c=[[28], [64], [30], [30], [30], [30]],
                                voltage_mv=[12000])
    node.last_health = 0.
    with pytest.raises(MotionFailure, match='cooling'):
        node.health()
    node.phase = 'recovering'
    node.health()


class HomeBoard:
    def __init__(self, positions=FOLDED_ENTRY):
        self.positions = np.array(positions, copy=True)
        self.temperatures = [28]*6
        self.torque = [0]*6
        self.calls = []
        self.block_shoulder = False
        self.ignore_disable = None
        self.drift_after_disable = None

    def bus_servo_set_position(self, duration, positions):
        goal = self.positions.copy()
        for servo, position in positions:
            goal[SERVO_IDS.index(servo)] = position
        self.calls.append(('position', duration, goal.tolist()))
        if self.block_shoulder:
            goal[1] = self.positions[1]
        self.positions = goal

    def bus_servo_enable_torque(self, servo, enabled):
        self.calls.append(('torque', servo, enabled))
        if enabled or servo != self.ignore_disable:
            self.torque[SERVO_IDS.index(servo)] = int(enabled)
        if not any(self.torque[:5]) and self.drift_after_disable is not None:
            self.positions += self.drift_after_disable
            self.drift_after_disable = None

    def bus_servo_read_position(self, servo):
        value = int(self.positions[SERVO_IDS.index(servo)])
        self.calls.append(('position_read', servo, value))
        return [value]

    def bus_servo_read_temp(self, servo):
        return [self.temperatures[SERVO_IDS.index(servo)]]

    def bus_servo_read_vin(self, servo):
        return [12000]

    def bus_servo_read_torque_state(self, servo):
        value = self.torque[SERVO_IDS.index(servo)]
        self.calls.append(('torque_read', servo, value))
        return [value]

    def bus_servo_stop(self, servos):
        self.calls.append(('stop', list(servos)))


def homing_node(positions=FOLDED_ENTRY):
    node = planner()
    node.board = HomeBoard(positions)
    node.joint_positions = CanManipulation.joint_positions.__get__(node)
    node.phase, node.held, node.grip_closed = 'unhomed', None, False
    node.run, node.worker = None, None
    node.last_health = 0.
    node.motion_started = False
    node.pause = lambda seconds: node.check_cancel()
    return node


def motor_commands(node):
    return [call for call in node.board.calls if call[0] in ('position', 'torque', 'stop')]


def test_manual_fold_homes_directly_without_raising_shoulder():
    node = homing_node()
    node.home()
    commands = motor_commands(node)
    assert commands[0] == ('position', .1, FOLDED_ENTRY.tolist())
    first_move = next(index for index, call in enumerate(commands[1:], 1)
                      if call[0] == 'position')
    assert commands[1:first_move] == [('torque', servo, True) for servo in SERVO_IDS]
    targets = [call[2] for call in commands if call[0] == 'position']
    assert all(target[1] <= FOLDED_ENTRY[1] for target in targets)
    assert targets[-1] == REST.tolist()
    assert node.phase == 'rest'
    assert node.board.torque == [0, 0, 0, 0, 0, 1]
    last_disable = max(index for index, call in enumerate(node.board.calls)
                       if call[0] == 'torque' and not call[2])
    after_disable = node.board.calls[last_disable+1:]
    assert all(('position_read', servo, int(REST[index])) in after_disable
               for index, servo in enumerate(SERVO_IDS[:5]))
    assert all(('torque_read', servo, 0) in after_disable for servo in SERVO_IDS[:5])


@pytest.mark.parametrize('joint', range(5))
@pytest.mark.parametrize('direction', [-1, 1])
def test_home_rejects_unmeasured_corners_inside_old_rest_box(joint, direction):
    actual = REST.copy()
    actual[joint] += direction*90
    assert np.max(np.abs(actual[:5]-REST[:5])) < 110
    node = homing_node(actual)
    with pytest.raises(MotionFailure, match='folded entry profile'):
        node.home()
    assert not motor_commands(node)
    assert node.phase == 'unhomed'


@pytest.mark.parametrize('condition', ['hot', 'canceled', 'cooling'])
def test_home_preflight_rejects_before_energizing(condition):
    node = homing_node()
    if condition == 'hot':
        node.board.temperatures[1] = 65
    elif condition == 'canceled':
        node.cancel.set()
    else:
        node.phase = 'cooling'
    with pytest.raises(MotionFailure):
        node.home()
    assert not motor_commands(node)


def test_home_rejects_pose_drift_after_arm_torque_is_off():
    node = homing_node()
    node.board.drift_after_disable = np.array([0, 20, 0, 0, 0, 0])
    with pytest.raises(MotionFailure, match='after torque removal'):
        node.home()
    assert node.phase == 'unhomed'
    assert node.cancel.is_set()
    assert node.board.positions[1] == REST[1]+20
    assert node.board.torque[:5] == [0]*5


def test_home_requires_measured_torque_shutdown():
    node = homing_node()
    node.board.ignore_disable = 3
    with pytest.raises(MotionFailure, match='after torque removal'):
        node.home()
    assert node.phase == 'unhomed'
    assert node.board.torque[2] == 1


def test_blocked_cold_pick_stops_without_a_second_home_attempt(tmp_path):
    node = homing_node()
    node.board.block_shoulder = True
    node.root = tmp_path
    node.goal_lock = threading.Lock()
    node.get_logger = lambda: SimpleNamespace(error=lambda message: None)
    attempts = []

    def home():
        attempts.append(True)
        CanManipulation.home(node)

    node.home = home
    response = node.start('pick', node.pick, SimpleNamespace())
    node.worker.join(timeout=2.)
    assert not node.worker.is_alive()
    assert response.success
    assert attempts == [True]
    assert node.result['success'] is False
    assert 'Joint tracking error' in node.result['error']
    assert 'recovery_error' not in node.result
    assert [call for call in motor_commands(node) if call[0] == 'stop'] == [
        ('stop', list(SERVO_IDS))]
    assert node.phase == 'unhomed'
    assert node.cancel.is_set()


def test_missing_feedback_cleanup_cannot_enable_a_later_unknown_home(tmp_path):
    node = homing_node()
    read_position = node.board.bus_servo_read_position

    def lose_feedback_after_motion(servo):
        if sum(call[0] == 'position' for call in node.board.calls) >= 2:
            return None
        return read_position(servo)

    node.board.bus_servo_read_position = lose_feedback_after_motion
    with pytest.raises(MotionFailure, match='feedback is missing'):
        node.home()
    assert len([call for call in motor_commands(node) if call[0] == 'position']) == 2
    assert ('stop', list(SERVO_IDS)) in motor_commands(node)
    assert node.phase == 'unhomed'
    assert node.cancel.is_set()

    node.board.bus_servo_read_position = read_position
    node.board.positions = np.array([500, 500, 500, 500, 500, 330])
    node.board.calls.clear()
    node.root = tmp_path
    node.goal_lock = threading.Lock()
    node.get_logger = lambda: SimpleNamespace(error=lambda message: None)
    response = node.start_home(None, SimpleNamespace())
    node.worker.join(timeout=2.)
    assert not node.worker.is_alive()
    assert response.success
    assert not node.cancel.is_set()
    assert node.result['success'] is False
    assert 'folded entry profile' in node.result['error']
    assert node.phase == 'unhomed'
    assert not motor_commands(node)


def test_cold_survey_establishes_supported_home_before_outward_route():
    node = homing_node()
    events = []
    node.event = lambda name, **values: events.append((name, values))
    can = dict(position=[.165, .1374, .1377])
    node.snapshot = lambda label: dict(can=can)
    assert node.survey() == can
    verified = next(index for index, event in enumerate(events) if event[0] == 'home_verified')
    preparing = events.index(('phase', dict(phase='preparing')))
    assert verified < preparing
    assert events[verified][1]['joints'] == REST.tolist()
    assert events[verified][1]['sensors']['torque'][:5] == [[0]]*5
    outbound = [event[1]['command'] for event in events[preparing+1:]
                if event[0] == 'joint_command']
    first_raise = next(index for index, row in enumerate(outbound) if row[1] > REST[1])
    assert any(row[5] == 200 for row in outbound[:first_raise])
    assert all(row[5] == 200 for row in outbound[first_raise:])


def test_rest_uses_the_same_home_operation():
    node = planner()
    calls = []
    node.home = lambda: calls.append('home')
    node.rest()
    assert calls == ['home']
