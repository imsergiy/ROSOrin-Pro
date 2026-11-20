"""Offline checks for artifact downloads and command argument boundaries."""

import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).parents[1] / 'example/rgbd_function/can_client.py'
SPEC = importlib.util.spec_from_file_location('can_client', SOURCE)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class CanClientTest(unittest.TestCase):
    def invoke(self, *arguments):
        output = io.StringIO()
        with patch.object(client.sys, 'argv', ['robot', *arguments]):
            with contextlib.redirect_stdout(output):
                client.main()
        return json.loads(output.getvalue())

    def test_image_download_uses_read_and_preserves_binary(self):
        payload = b'\x89PNG\r\n\x00\xff\x80artifact'
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'image.png'
            with patch.object(client, 'call', return_value=base64.b64encode(payload)) as call:
                result = self.invoke('image', '/data/run/survey.jpg', str(destination))
            call.assert_called_once_with('read', '/data/run/survey.jpg')
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(result, {'path': str(destination), 'bytes': len(payload)})

    def test_image_requires_two_nonempty_paths_before_service_access(self):
        for arguments in ((), ('/data/a',), ('/data/a', 'a', 'extra'),
                          ('', 'a'), ('/data/a', '')):
            with self.subTest(arguments=arguments):
                with patch.object(client, 'call') as call:
                    with self.assertRaisesRegex(SystemExit, 'ONBOARD_PATH LOCAL_PATH'):
                        self.invoke('image', *arguments)
                    call.assert_not_called()

    def test_invalid_base64_does_not_create_or_replace_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'image.png'
            destination.write_bytes(b'original')
            with patch.object(client, 'call', return_value='invalid!'):
                with self.assertRaises(SystemExit) as failure:
                    self.invoke('image', '/data/run/a', str(destination))
            self.assertEqual(failure.exception.code, 1)
            self.assertEqual(destination.read_bytes(), b'original')

    def test_read_and_write_failures_return_cli_failure(self):
        with patch.object(client, 'call', side_effect=RuntimeError('Unknown onboard artifact')):
            with self.assertRaises(SystemExit) as failure:
                self.invoke('image', '/data/missing', 'unused')
        self.assertEqual(failure.exception.code, 1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(client, 'call', return_value='YWJj'):
                with self.assertRaises(SystemExit) as failure:
                    self.invoke('image', '/data/a', directory)
            self.assertEqual(failure.exception.code, 1)

    def test_help_documents_download_without_ros_access(self):
        with patch.object(client, 'call') as call:
            result = self.invoke('help')
            self.assertIn('ONBOARD_PATH LOCAL_PATH', result['commands']['image'])
            call.assert_not_called()

    def test_other_commands_reject_missing_and_extra_arguments(self):
        for arguments in (('status', 'extra'), ('place',), ('read',), ('help', 'extra'),
                          ('place', '{}', 'extra'), ('unknown',)):
            with self.subTest(arguments=arguments):
                with patch.object(client, 'call') as call:
                    with self.assertRaises(SystemExit):
                        self.invoke(*arguments)
                    call.assert_not_called()

    def test_status_and_place_keep_their_service_arguments(self):
        cases = (('status', (), None),
                 ('home', (), None),
                 ('place', ('{"frame":"arm_base"}',), '{"frame":"arm_base"}'))
        for command, operands, expected in cases:
            with self.subTest(command=command):
                with patch.object(client, 'call', return_value='{"accepted":true}') as call:
                    self.assertEqual(self.invoke(command, *operands), {'accepted': True})
                    call.assert_called_once_with(command, expected)


if __name__ == '__main__':
    unittest.main()
