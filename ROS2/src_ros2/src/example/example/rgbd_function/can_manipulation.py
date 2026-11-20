#!/usr/bin/env python3
"""Onboard RGB-D can manipulation with measured joint feedback."""

import base64
import json
import math
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from interfaces.srv import SetString
from kinematics.forward_kinematics import ForwardKinematics
from kinematics.inverse_kinematics import get_ik
from kinematics.transform import angle2pulse, pulse2angle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_robot_controller.ros_robot_controller_sdk import Board
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger

from example.rgbd_function.can_perception import detect_can, verify_visual_lift


SERVO_IDS = (1, 2, 3, 4, 5, 10)
OPTICAL_ROTATION = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
REST = np.array([331, 99, 393, 214, 401, 330])
FOLDED_ENTRY = np.array([355, 127, 308, 258, 497, 332])
SIDE_CLEARANCE = [
    [331, 158, 381, 212, 401, 330],
    [331, 192, 380, 173, 401, 300],
    [331, 222, 400, 132, 401, 250],
    [331, 270, 360, 124, 450, 200],
    [331, 335, 322, 121, 500, 200],
    [331, 424, 255, 132, 500, 200],
    [331, 491, 276, 85, 500, 200],
    [331, 573, 268, 96, 500, 200],
]
SURVEY = np.array([500, 811, 75, 92, 500, 200])


class MotionFailure(RuntimeError):
    """The requested manipulation could not be completed with verified feedback."""


class CanManipulation(Node):
    """One serial owner and one motion worker, with independent observation callbacks."""

    def __init__(self):
        super().__init__('can_manipulation')
        self.declare_parameter('serial_device', '/dev/ttyACM0')
        self.declare_parameter('data_directory', '/var/tmp/rosorin-manipulation')
        self.declare_parameter('can_radius', .025)
        self.declare_parameter('can_height', .135)
        self.declare_parameter('contact_offset', .032)
        self.declare_parameter('grasp_pitch', 38.)
        self.declare_parameter('camera_offset', [-.101, .010, .057])
        self.root = Path(self.get_parameter('data_directory').value)
        self.root.mkdir(parents=True, exist_ok=True)
        self.radius = float(self.get_parameter('can_radius').value)
        self.height = float(self.get_parameter('can_height').value)
        self.contact_offset = float(self.get_parameter('contact_offset').value)
        self.pitch = float(self.get_parameter('grasp_pitch').value)
        self.camera_offset = np.array(self.get_parameter('camera_offset').value)
        self.board = Board(self.get_parameter('serial_device').value, timeout=.15)
        self.board.enable_reception()
        self.hardware_lock = threading.RLock()
        self.frame_lock = threading.Lock()
        self.goal_lock = threading.Lock()
        self.cancel = threading.Event()
        self.frames = {}
        self.bridge = CvBridge()
        self.fk = ForwardKinematics(debug=False)
        self.worker = None
        self.phase = 'unhomed'
        self.result = None
        self.held = None
        self.run = None
        self.last_observation = None
        self.observations = []
        self.last_health = 0.
        self.grip_closed = False
        self.contact_tip = None
        self.contact_joints = None
        self.motion_started = False
        group = ReentrantCallbackGroup()
        self.create_subscription(Image, '/depth_cam/rgb0/image_raw',
                                 lambda msg: self.store_frame('rgb', msg),
                                 qos_profile_sensor_data, callback_group=group)
        self.create_subscription(Image, '/depth_cam/depth0/image_raw',
                                 lambda msg: self.store_frame('depth', msg),
                                 qos_profile_sensor_data, callback_group=group)
        self.create_subscription(CameraInfo, '/depth_cam/rgb0/camera_info',
                                 lambda msg: self.store_frame('info', msg),
                                 qos_profile_sensor_data, callback_group=group)
        self.create_service(Trigger, '~/status', self.status, callback_group=group)
        self.create_service(Trigger, '~/observe', self.observe, callback_group=group)
        self.create_service(Trigger, '~/pick', self.start_pick, callback_group=group)
        self.create_service(SetString, '~/place', self.start_place, callback_group=group)
        self.create_service(Trigger, '~/return', self.return_can, callback_group=group)
        self.create_service(Trigger, '~/home', self.start_home, callback_group=group)
        self.create_service(Trigger, '~/rest', self.start_rest, callback_group=group)
        self.create_service(Trigger, '~/stop', self.stop, callback_group=group)
        self.create_service(SetString, '~/read', self.read_artifact, callback_group=group)
        self.create_timer(2., self.monitor_hold, callback_group=group)

    def store_frame(self, name, msg):
        with self.frame_lock:
            self.frames[name] = (time.monotonic(), msg)

    def joint_positions(self):
        with self.hardware_lock:
            readings = [self.board.bus_servo_read_position(i) for i in SERVO_IDS]
        if any(value is None or len(value) != 1 for value in readings):
            raise MotionFailure('Servo position feedback is missing')
        result = np.array([value[0] for value in readings])
        if not np.all((result >= 0) & (result <= 1000)):
            raise MotionFailure('Servo position feedback is invalid')
        return result

    def tool_pose(self, joints):
        result = self.fk.get_fk(pulse2angle(joints[:5]))
        if result is False:
            raise MotionFailure('Measured joints are outside the kinematic model range')
        position, quaternion = result
        rotation = Rotation.from_quat(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]).as_matrix()
        return np.array(position), rotation

    def camera_pose(self, joints):
        # The camera is attached before wrist roll, which does not rotate the camera.
        camera_joints = np.array(joints, copy=True)
        camera_joints[4] = 500
        tip, rotation = self.tool_pose(camera_joints)
        transform = np.eye(4)
        transform[:3, :3] = rotation @ OPTICAL_ROTATION
        transform[:3, 3] = tip + rotation @ self.camera_offset
        return transform

    def sensors(self):
        with self.hardware_lock:
            temperature = [self.board.bus_servo_read_temp(i) for i in SERVO_IDS]
            voltage = self.board.bus_servo_read_vin(3)
            torque = [self.board.bus_servo_read_torque_state(i) for i in SERVO_IDS]
        return {'temperature_c': temperature, 'voltage_mv': voltage, 'torque': torque}

    def health(self, *, parking=False):
        self.check_cancel()
        if time.monotonic() - self.last_health < .7:
            return
        sensors = self.sensors()
        self.event('health', **sensors)
        cooling = parking or self.phase in ('recovering', 'parking')
        limits = [65, 65, 65, 65, 65, 75] if cooling else [62, 62, 62, 62, 62, 70]
        if any(value is None or value[0] >= limit
               for value, limit in zip(sensors['temperature_c'], limits)):
            raise MotionFailure('Servo temperature requires cooling')
        voltage = sensors['voltage_mv']
        if voltage is None or voltage[0] < 10500:
            raise MotionFailure('Battery voltage is too low for manipulation')
        self.last_health = time.monotonic()

    def check_cancel(self):
        if self.cancel.is_set():
            raise MotionFailure('Goal canceled')

    def pause(self, seconds):
        if self.cancel.wait(seconds):
            raise MotionFailure('Goal canceled')

    def event(self, name, **values):
        if self.run is None:
            return
        record = {'time': time.time(), 'event': name, **values}
        with (self.run / 'events.jsonl').open('a') as output:
            output.write(json.dumps(record) + '\n')

    def feedback(self, phase):
        with self.hardware_lock:
            if self.phase == 'cooling' and phase != 'cooling':
                return
            self.phase = phase
            self.event('phase', phase=phase)

    def step_joints(self, goal, *, correction=True):
        """Move coordinated joints and compensate bounded loaded-position error."""
        self.health()
        desired = np.asarray(goal, dtype=int)
        current = self.joint_positions()
        delta = np.abs(desired-current)
        if np.max(delta[:5]) > 106 or delta[5] > 120:
            raise MotionFailure('Motion planner exceeded the joint step limit')
        if np.min(desired) < 10 or np.max(desired[:5]) > 940:
            raise MotionFailure('Joint goal exceeds calibrated limits')
        duration = max(.45, float(np.max(delta[:5])) / 120.)
        correction_pulses = np.zeros(5, dtype=int)
        for attempt in range(4 if correction else 1):
            self.check_cancel()
            command = desired.copy()
            command[:5] += correction_pulses
            if np.min(command[:5]) < 10 or np.max(command[:5]) > 950:
                raise MotionFailure('Loaded correction exceeds joint limits')
            with self.hardware_lock:
                self.check_cancel()
                self.motion_started = True
                self.event('joint_command', command=command.tolist(),
                           duration=duration if attempt == 0 else .35)
                self.board.bus_servo_set_position(
                    duration if attempt == 0 else .35,
                    list(zip(SERVO_IDS, map(int, command))))
            self.pause((duration if attempt == 0 else .35) + .15)
            actual = self.joint_positions()
            error = desired[:5]-actual[:5]
            self.event('joint_feedback', goal=desired.tolist(), actual=actual.tolist())
            if np.max(np.abs(error)) <= 6:
                return actual
            if not correction:
                return actual
            correction_pulses = np.clip(correction_pulses+error, -45, 45)
        if np.max(np.abs(error)) > 12:
            raise MotionFailure(f'Joint tracking error {error.tolist()}')
        return actual

    def move_joints(self, goal):
        start = self.joint_positions()
        goal = np.asarray(goal, dtype=int)
        count = max(1, int(np.ceil(np.max(np.abs(goal-start))/85.)))
        for fraction in np.linspace(0., 1., count+1)[1:]:
            self.step_joints(np.rint(start+(goal-start)*fraction).astype(int))

    def enable_motion(self, actual):
        self.health()
        with self.hardware_lock:
            self.check_cancel()
            self.board.bus_servo_set_position(.1, list(zip(SERVO_IDS, map(int, actual))))
            for servo in SERVO_IDS:
                self.board.bus_servo_enable_torque(servo, True)
            self.event('torque_request', enabled=True, joints=actual.tolist())

    @staticmethod
    def folded_pose(joints):
        span = REST[:5]-FOLDED_ENTRY[:5]
        offset = np.asarray(joints)[:5]-FOLDED_ENTRY[:5]
        fraction = np.clip(np.dot(offset, span)/np.dot(span, span), 0., 1.)
        return np.max(np.abs(offset-fraction*span)) <= 15

    def solve(self, position, pitch, current, wrist=500):
        if not np.all(np.isfinite(position)):
            raise MotionFailure('Target coordinates must be finite')
        solutions = []
        for entry in get_ik(list(map(float, position)), pitch, [pitch, pitch], 1.):
            for values in angle2pulse(entry[0]):
                row = np.rint(values).astype(int).tolist()
                row[4] = wrist
                if all(35 <= pulse <= 920 for pulse in row[:4]):
                    solutions.append(row)
        if not solutions:
            raise MotionFailure(f'Unreachable grasp pose {np.round(position, 3).tolist()}')
        return np.array(min(solutions, key=lambda row: np.sum(np.abs(row-current[:5]))))

    def cartesian_path(self, destination, pitch, jaw, start=None):
        joints = self.joint_positions() if start is None else np.asarray(start)
        origin, _ = self.tool_pose(joints)
        distance = np.linalg.norm(destination-origin)
        count = max(1, int(np.ceil(distance/.009)))
        path = []
        for fraction in np.linspace(0., 1., count+1)[1:]:
            position = origin+(destination-origin)*fraction
            row = self.solve(position, pitch, joints)
            if np.max(np.abs(row-joints[:5])) > 90:
                raise MotionFailure('Cartesian path has an abrupt joint change')
            joints = np.r_[row, jaw]
            path.append(joints)
        return path

    def follow(self, path):
        for joints in path:
            self.step_joints(joints)

    def snapshot(self, label='observation', expected_xy=None, support_z=None):
        if (self.worker is not None and self.worker.is_alive()
                and self.worker is not threading.current_thread()):
            raise MotionFailure('Wait until motion settles before an observation')
        time.sleep(.18)
        deadline = time.monotonic()+3.
        while True:
            with self.frame_lock:
                frames = self.frames.copy()
            if all(name in frames for name in ('rgb', 'depth', 'info')):
                rgb_time, rgb = frames['rgb']
                depth_time, depth = frames['depth']
                info_time, info = frames['info']
                rgb_stamp = rgb.header.stamp.sec+rgb.header.stamp.nanosec/1e9
                depth_stamp = depth.header.stamp.sec+depth.header.stamp.nanosec/1e9
                if (min(rgb_time, depth_time, info_time) > time.monotonic()-1.
                        and abs(rgb_stamp-depth_stamp) <= .035):
                    break
            if time.monotonic() >= deadline:
                raise MotionFailure('Fresh synchronized onboard RGB-D frames are unavailable')
            time.sleep(.03)
        joints = self.joint_positions()
        bgr = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        depth_mm = self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        if bgr.shape[:2] != depth_mm.shape:
            raise MotionFailure('Aligned RGB and depth resolutions differ')
        intrinsics = (info.k[0], info.k[4], info.k[2], info.k[5])
        directory = self.run if self.run is not None else self.root
        stamp = str(time.time_ns())
        rgb_path = directory / f'{label}-{stamp}.jpg'
        depth_path = directory / f'{label}-{stamp}-depth.png'
        cv2.imwrite(str(rgb_path), bgr)
        cv2.imwrite(str(depth_path), depth_mm)
        observation = {
            'rgb_path': str(rgb_path), 'depth_path': str(depth_path),
            'frame': 'arm_base', 'joints': joints.tolist(),
            'tool_position': self.tool_pose(joints)[0].tolist(),
            'intrinsics': list(intrinsics), 'time': time.time(),
            'rgb_stamp': rgb_stamp, 'depth_stamp': depth_stamp,
            'camera_to_arm': self.camera_pose(joints).tolist(),
        }
        try:
            observation['can'] = detect_can(
                bgr, depth_mm, intrinsics, self.camera_pose(joints),
                expected_xy=expected_xy, radius=self.radius,
                height=self.height, support_z=support_z)
        except ValueError as error:
            observation['detection_error'] = str(error)
        (directory / f'{label}-{stamp}.json').write_text(json.dumps(observation, indent=2))
        self.last_observation = observation
        self.observations = (self.observations+[observation])[-12:]
        self.event('observation', **observation)
        return observation

    def survey(self):
        if self.phase in ('unhomed', 'rest'):
            self.home()
            self.feedback('preparing')
            self.enable_motion(self.joint_positions())
            for joints in SIDE_CLEARANCE:
                self.move_joints(joints)
        self.feedback('survey')
        self.move_joints([331, 811, 75, 92, 500, 200])
        for yaw in (500, 650, 350):
            self.move_joints([yaw, *SURVEY[1:]])
            image = self.snapshot('survey')
            if 'can' in image:
                return image['can']
        raise MotionFailure('No supported Red Bull can is visible in the onboard search area')

    def grasp_tip(self, center, pitch):
        heading = math.atan2(center[1], center[0])
        # Tool X is the finger direction, set by the requested downward pitch.
        direction = np.array([math.cos(heading)*math.cos(math.radians(pitch)),
                              math.sin(heading)*math.cos(math.radians(pitch)),
                              -math.sin(math.radians(pitch))])
        return center+direction*self.contact_offset

    def pick(self):
        if self.phase not in ('unhomed', 'rest') or self.held is not None or self.grip_closed:
            raise MotionFailure('Pickup requires an empty gripper at home')
        can = self.survey()
        self.feedback('planning')
        center = np.asarray(can['position'], dtype=float)
        if np.linalg.norm(center[:2]) > .29 or not -.05 < center[2] < .26:
            raise MotionFailure('Can is outside the manipulation workspace')
        center[2] = can['bottom_z']+can['height']*.90
        current = self.joint_positions()
        tip = self.grasp_tip(center, self.pitch)
        hover = tip+np.array([0., 0., .027])
        hover_joints = self.solve(hover, self.pitch, current)
        approach = np.r_[hover_joints, 200]
        descend = self.cartesian_path(tip, self.pitch, 200, start=approach)
        lift = self.cartesian_path(tip+np.array([0., 0., .035]),
                                   self.pitch, 200, start=descend[-1])
        self.feedback('approach')
        self.move_joints(approach)
        self.follow(descend)
        self.feedback('gripping')
        contact = self.joint_positions()
        self.contact_tip = self.tool_pose(contact)[0]
        self.contact_joints = contact.copy()
        self.grip_closed = True
        for jaw in (275, 345, 400, 435):
            contact = self.step_joints(np.r_[contact[:5], jaw])
            if jaw-contact[5] > 18:
                break
        if contact[5] > 425:
            self.open_gripper()
            raise MotionFailure('Gripper closed without detecting can contact')
        prelift_tool = self.tool_pose(self.joint_positions())[0]
        before = self.snapshot('before-lift', center[:2], can['support_z'])
        self.feedback('test-lift')
        try:
            self.follow([np.r_[row[:5], jaw] for row in lift])
            after = self.snapshot('after-lift', center[:2], can['support_z'])
            tool, rotation = self.tool_pose(self.joint_positions())
            verification = verify_visual_lift(
                cv2.imread(before['rgb_path']), cv2.imread(after['rgb_path']),
                before['intrinsics'], np.asarray(before['camera_to_arm']),
                np.asarray(after['camera_to_arm']), can, tool-prelift_tool)
            measured_center = np.asarray(can['position'])+np.asarray(verification['displacement'])
            with self.hardware_lock:
                self.check_cancel()
                self.held = {
                    'source': can, 'can': {**can, 'position': measured_center.tolist()},
                    'can_in_tool': (rotation.T @ (measured_center-tool)).tolist(),
                    'pitch': self.pitch, 'contact': int(contact[5]), 'jaw_command': jaw,
                    'verification': verification,
                    'since': time.monotonic(),
                }
                self.event('verified_lift', held=self.held)
                self.feedback('held')
        except (MotionFailure, ValueError):
            if not self.cancel.is_set():
                self.recover_source()
            raise

    def recover_source(self):
        if self.grip_closed and self.contact_tip is not None:
            self.feedback('recovering')
            current = self.joint_positions()
            self.move_joints(np.r_[self.contact_joints[:5], int(current[5])])
            self.open_gripper()
            self.grip_closed = False

    def open_gripper(self):
        joints = self.joint_positions()
        count = max(1, int(np.ceil(abs(int(joints[5])-200)/85.)))
        for jaw in np.rint(np.linspace(joints[5], 200, count+1)[1:]).astype(int):
            joints = self.step_joints(np.r_[joints[:5], jaw])
        if abs(int(joints[5])-200) > 15:
            raise MotionFailure('Gripper opening could not be verified')
        self.grip_closed = False

    def place(self, goal):
        if self.held is None:
            raise MotionFailure('No verified can is held')
        position = np.asarray(goal['position'], dtype=float)
        if goal.get('frame') != 'arm_base' or position.shape != (3,):
            raise MotionFailure('Placement needs frame arm_base and three support coordinates')
        if not np.all(np.isfinite(position)) or not .12 < np.linalg.norm(position[:2]) < .29:
            raise MotionFailure('Placement is outside the manipulation workspace')
        if not -.05 <= position[2] <= .16:
            raise MotionFailure('Support height is outside the manipulation workspace')
        center = position+np.array([0., 0., self.held['can']['height']/2.])
        can_height = self.held['can']['height']
        current = self.joint_positions()
        seed = self.solve(center, self.held['pitch'], current)
        _, rotation = self.tool_pose(seed)
        tip = center-rotation @ np.asarray(self.held['can_in_tool'])
        raised = tip+np.array([0., 0., .035])
        raised_joints = self.solve(raised, self.held['pitch'], current)
        jaw = self.held['jaw_command']
        descend = self.cartesian_path(tip, self.held['pitch'], jaw,
                                      start=np.r_[raised_joints, jaw])
        self.feedback('placing')
        self.move_joints(np.r_[raised_joints, jaw])
        self.follow(descend)
        self.open_gripper()
        self.held = None
        self.follow([np.r_[row[:5], 200] for row in reversed(descend[:-1])])
        inspection = SURVEY.copy()
        heading = math.degrees(math.atan2(position[1], position[0]))
        inspection[0] = int(round(500+heading/.24))
        self.move_joints(inspection)
        image = self.snapshot('placed-inspection', position[:2])
        if ('can' not in image or np.linalg.norm(
                np.asarray(image['can']['position'])[:2]-position[:2]) > .02
                or abs(image['can']['support_z']-position[2]) > .015
                or abs(image['can']['top_z']-can_height-image['can']['support_z']) > .010):
            raise MotionFailure('Release completed but placement could not be verified')
        self.feedback('idle')
        self.rest()

    def rest(self):
        self.home()

    def home(self):
        if self.held is not None or self.grip_closed:
            raise MotionFailure('Place or return the held can before resting')
        if self.phase == 'cooling':
            raise MotionFailure('Arm torque is off; support and reset the arm')
        actual = self.joint_positions()
        folded = self.folded_pose(actual)
        if not folded and self.phase in ('unhomed', 'rest'):
            raise MotionFailure('Home start pose is outside the verified folded entry profile')
        self.health(parking=True)
        path = [REST] if folded else [*reversed(SIDE_CLEARANCE), REST]
        self.feedback('parking')
        try:
            self.enable_motion(actual)
            for joints in path:
                self.move_joints(joints)
            with self.hardware_lock:
                self.check_cancel()
                for servo in (2, 3, 4, 5, 1):
                    self.board.bus_servo_enable_torque(servo, False)
            self.pause(.35)
            settled = self.joint_positions()
            sensors = self.sensors()
            if (np.max(np.abs(settled[:5]-REST[:5])) > 15
                    or any(value is None or value[0] != 0 for value in sensors['torque'][:5])):
                raise MotionFailure('Supported home could not be verified after torque removal')
            with self.hardware_lock:
                self.check_cancel()
                self.event('home_verified', joints=settled.tolist(), sensors=sensors)
                self.feedback('rest')
        except Exception:
            with self.hardware_lock:
                self.cancel.set()
                self.feedback('unhomed')
                self.board.bus_servo_stop(list(SERVO_IDS))
                if self.folded_pose(self.joint_positions()):
                    for servo in (2, 3, 4, 5, 1):
                        self.board.bus_servo_enable_torque(servo, False)
            raise

    def start(self, operation, action, response):
        with self.goal_lock:
            if self.worker is not None and self.worker.is_alive():
                response.success = False
                response.message = json.dumps({'error': 'A manipulation goal is already running'})
                return response
            self.cancel.clear()
            self.run = self.root / uuid.uuid4().hex[:12]
            self.run.mkdir()
            self.result = None
            self.observations = []
            self.motion_started = False

            def execute():
                started = time.monotonic()
                try:
                    action()
                    with self.hardware_lock:
                        self.check_cancel()
                        self.result = {'success': True, 'operation': operation}
                except Exception as error:
                    self.result = {'success': False, 'operation': operation, 'error': str(error)}
                    self.get_logger().error(str(error))
                    if (operation == 'pick' and self.motion_started
                            and not self.cancel.is_set() and self.held is None
                            and self.phase != 'unhomed'):
                        try:
                            self.recover_source()
                            self.rest()
                        except (MotionFailure, ValueError, RuntimeError) as recovery_error:
                            self.result['recovery_error'] = str(recovery_error)
                finally:
                    self.result['elapsed_seconds'] = round(time.monotonic()-started, 2)
                    self.event('result', **self.result)
                    (self.run / 'result.json').write_text(json.dumps(self.result, indent=2))
            self.worker = threading.Thread(target=execute, daemon=True)
            self.worker.start()
            response.success = True
            response.message = json.dumps({'accepted': True, 'run': self.run.name})
            return response

    def start_pick(self, request, response):
        if self.held is not None:
            response.success, response.message = False, 'A can is already held'
            return response
        return self.start('pick', self.pick, response)

    def start_place(self, request, response):
        try:
            goal = json.loads(request.data)
        except (TypeError, json.JSONDecodeError):
            response.success, response.message = False, 'Placement data must be a JSON object'
            return response
        if not isinstance(goal, dict) or set(goal) != {'frame', 'position'}:
            response.success, response.message = False, 'Placement requires frame and position'
            return response
        return self.start('place', lambda: self.place(goal), response)

    def return_can(self, request, response):
        if self.phase == 'cooling':
            response.success = False
            response.message = 'Arm torque is off; support and reset the arm'
            return response
        if self.held is None and self.grip_closed and self.contact_tip is not None:
            def recover():
                self.recover_source()
                self.rest()
            return self.start('recover', recover, response)
        if self.held is None:
            response.success, response.message = False, 'No verified can is held'
            return response
        source = self.held['source']
        goal = {'frame': 'arm_base',
                'position': [*source['position'][:2], source['support_z']]}
        return self.start('return', lambda: self.place(goal), response)

    def start_rest(self, request, response):
        return self.start('rest', self.rest, response)

    def start_home(self, request, response):
        return self.start('home', self.home, response)

    def monitor_hold(self):
        if self.phase == 'rest' and self.held is None and not self.grip_closed:
            return
        sensors = self.sensors()
        critical = any(value is None or value[0] >= limit for value, limit in zip(
            sensors['temperature_c'], [65, 65, 65, 65, 65, 75]))
        if critical:
            with self.hardware_lock:
                self.cancel.set()
                self.board.bus_servo_stop(list(SERVO_IDS))
                for servo in (2, 3, 4, 5, 1):
                    self.board.bus_servo_enable_torque(servo, False)
                jaw_temp = sensors['temperature_c'][-1]
                if jaw_temp is None or jaw_temp[0] >= 75:
                    self.board.bus_servo_enable_torque(10, False)
                self.held = None
                self.feedback('cooling')
            self.event('thermal_stop', sensors=sensors)
            return
        if (self.cancel.is_set()
                or (self.worker is not None and self.worker.is_alive())):
            return
        if self.held is None and not self.grip_closed:
            return
        warming = any(value is None or value[0] >= limit for value, limit in zip(
            sensors['temperature_c'], [55, 55, 55, 55, 55, 63]))
        expired = self.held is not None and time.monotonic()-self.held['since'] > 60
        if warming or expired:
            self.event('hold_limit', sensors=sensors)
            self.return_can(None, Trigger.Response())

    def status(self, request, response):
        try:
            with self.hardware_lock:
                joints = self.joint_positions()
                sensors = self.sensors()
            response.message = json.dumps({
                'busy': self.worker is not None and self.worker.is_alive(),
                'phase': self.phase, 'result': self.result, 'held': self.held,
                'joints': joints.tolist(), 'sensors': sensors,
                'run': None if self.run is None else self.run.name,
                'observation': self.last_observation,
                'observations': self.observations,
            })
            response.success = True
        except MotionFailure as error:
            response.success, response.message = False, str(error)
        return response

    def observe(self, request, response):
        try:
            response.message = json.dumps(self.snapshot())
            response.success = True
        except (MotionFailure, ValueError) as error:
            response.success, response.message = False, str(error)
        return response

    def stop(self, request, response):
        self.cancel.set()
        with self.hardware_lock:
            self.board.bus_servo_stop(list(SERVO_IDS))
        self.event('cancel')
        response.success, response.message = True, 'Motion stop requested'
        return response

    def read_artifact(self, request, response):
        path = Path(request.data).resolve()
        if (not path.is_relative_to(self.root.resolve()) or
                path.suffix not in ('.jpg', '.png', '.json', '.jsonl') or
                not path.is_file() or path.stat().st_size > 4_000_000):
            response.success, response.message = False, 'Unknown onboard artifact'
            return response
        response.success = True
        response.message = base64.b64encode(path.read_bytes()).decode('ascii')
        return response


def main():
    rclpy.init()
    node = CanManipulation()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
