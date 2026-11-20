#!/usr/bin/env python3
"""Small client for the onboard can manipulation services."""

import base64
import json
from pathlib import Path
import sys


HELP = {
    'purpose': 'Pick up an upright Red Bull can using onboard sensors.',
    'commands': {
        'pick': 'Locate the can, grasp, lift, and verify. Returns a run ID.',
        'status': 'Read goal progress, result, measured servo state, and latest observation.',
        'observe': ('Save and return current onboard RGB-D images, calibration, '
                    'and can measurement.'),
        'place': ('Place at a support position. Argument is JSON with frame arm_base '
                  'and position [x,y,z] in meters.'),
        'return': 'Place the held can back at its measured source position and rest the arm.',
        'home': 'Park the empty arm at its supported home and verify arm torque is off.',
        'rest': 'Alias for home: park the empty arm and disable arm holding torque.',
        'stop': 'Cancel motion and hold the current position.',
        'image': 'Download an onboard artifact: image ONBOARD_PATH LOCAL_PATH.',
    },
    'usage': 'robot COMMAND [ARGUMENT]',
    'semantics': ('Motion requests are asynchronous. Poll status until busy is false '
                  'and inspect result.success. Pick success requires onboard '
                  'visual lift verification.'),
    'placement_frame': ('arm_base is centered at the arm rotation axis, X forward, Y left, '
                        'Z up. Placement position describes the bottom center of the can.'),
}


def call(command, argument=None):
    import rclpy
    from interfaces.srv import SetString
    from std_srvs.srv import Trigger

    service_type = SetString if command in ('place', 'read') else Trigger
    request = service_type.Request()
    if service_type is SetString:
        request.data = argument
    rclpy.init()
    node = rclpy.create_node('can_client')
    try:
        client = node.create_client(service_type, '/can_manipulation/'+command)
        if not client.wait_for_service(timeout_sec=8):
            raise RuntimeError('Robot manipulation API is unavailable')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15)
        if not future.done() or future.result() is None:
            raise RuntimeError('Robot API response timed out')
        response = future.result()
        if not response.success:
            raise RuntimeError(response.message)
        return response.message
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    arguments = sys.argv[1:]
    command = arguments[0] if arguments else 'help'
    operands = arguments[1:]
    if command == 'help':
        if operands:
            raise SystemExit('Usage: robot help')
        print(json.dumps(HELP, indent=2))
        return
    if command not in HELP['commands'] and command != 'read':
        raise SystemExit('Unknown command. Run robot help.')
    count = 2 if command == 'image' else 1 if command in ('place', 'read') else 0
    if len(operands) != count or any(not operand for operand in operands):
        usage = ' ONBOARD_PATH LOCAL_PATH' if command == 'image' else ' ARGUMENT' if count else ''
        raise SystemExit(f'Usage: robot {command}{usage}')
    try:
        if command == 'image':
            data = base64.b64decode(call('read', operands[0]), validate=True)
            output = Path(operands[1])
            output.write_bytes(data)
            print(json.dumps({'path': str(output), 'bytes': len(data)}, indent=2))
            return
        message = call(command, operands[0] if operands else None)
        try:
            value = json.loads(message)
        except json.JSONDecodeError:
            value = message
        print(json.dumps(value, indent=2))
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({'error': str(error)}))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
