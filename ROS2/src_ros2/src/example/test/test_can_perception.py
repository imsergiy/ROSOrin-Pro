"""Synthetic RGB-D regressions for geometry and independent lift verification."""

import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np


SOURCE = Path(__file__).parents[1] / 'example/rgbd_function/can_perception.py'
SPEC = importlib.util.spec_from_file_location('can_perception', SOURCE)
perception = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perception)


def scene(centers=((0., 0.),), lift=0., camera_lift=0., texture=True,
          can_height=.135, radius=.025):
    """Ray trace a red cylinder and horizontal support, with missing lid depth."""
    width, height = 400, 320
    intrinsics = np.array([360., 360., 200., 160.])
    camera = np.array([0., -.32, .30 + camera_lift])
    forward = np.array([0., 0., .12 + camera_lift]) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, down, forward))
    transform[:3, 3] = camera
    v, u = np.indices((height, width))
    rays = np.stack(((u - intrinsics[2]) / intrinsics[0],
                     (v - intrinsics[3]) / intrinsics[1], np.ones_like(u)), axis=-1)
    rays = rays @ transform[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = (.05 - camera[2]) / rays[:, :, 2]
    distance[distance <= 0] = np.inf
    bgr = np.full((height, width, 3), 125, np.uint8)
    lid_mask = np.zeros((height, width), bool)
    for x, y in centers:
        offset = camera[:2] - [x, y]
        a = np.sum(rays[:, :, :2] ** 2, axis=2)
        b = 2 * np.sum(rays[:, :, :2] * offset, axis=2)
        c = np.sum(offset ** 2) - radius ** 2
        discriminant = b ** 2 - 4 * a * c
        side_depth = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
        side_z = camera[2] + side_depth * rays[:, :, 2]
        side = ((discriminant >= 0) & (side_depth > 0) & (side_depth < distance)
                & (side_z >= .05 + lift) & (side_z <= .05 + can_height + lift))
        distance[side] = side_depth[side]
        bgr[side] = [20, 35, 210]
        if texture:
            surface = camera + side_depth[:, :, None] * rays
            angle = np.arctan2(surface[:, :, 1] - y, surface[:, :, 0] - x)
            checker = ((np.floor(angle * 8) + np.floor((side_z - lift) / .009)) % 2) == 0
            bgr[side & checker] = [12, 20, 110]
        lid_mask[side] = False
        with np.errstate(divide='ignore', invalid='ignore'):
            lid_depth = (.05 + can_height + lift - camera[2]) / rays[:, :, 2]
        lid_xy = camera[:2] + lid_depth[:, :, None] * rays[:, :, :2]
        radial = np.linalg.norm(lid_xy - [x, y], axis=2)
        lid = (lid_depth > 0) & (lid_depth < distance) & (radial <= radius)
        distance[lid] = lid_depth[lid]
        bgr[lid] = [185, 185, 185]
        bgr[lid & (radial > radius - .002)] = [245, 245, 245]
        lid_mask[lid] = True
    depth = np.where(np.isfinite(distance) & (distance < 1.5), distance * 1000, 0)
    depth = np.rint(depth).astype(np.uint16)
    depth[lid_mask] = 0
    return bgr, depth, intrinsics, transform


class CanPerceptionTest(unittest.TestCase):
    def detect(self, **kwargs):
        return perception.detect_can(*scene(**kwargs))

    def test_center_and_support_ignore_missing_lid_depth(self):
        result = self.detect()
        np.testing.assert_allclose(result['position'], [0., 0., .1175], atol=.004)
        self.assertAlmostEqual(result['support_z'], .05, delta=.003)
        self.assertAlmostEqual(result['top_z'], .185, delta=.004)
        self.assertGreater(result['depth_sample_count'], 80)
        self.assertLess(result['error_m'], .0025)

    def test_body_fit_tolerates_depth_noise_and_sparse_outliers(self):
        bgr, depth, intrinsics, transform = scene()
        rng = np.random.default_rng(4)
        depth = depth.astype(float)
        valid = depth > 0
        depth[valid] += rng.normal(0., .5, np.count_nonzero(valid))
        outliers = valid & (rng.random(depth.shape) < .08)
        depth[outliers] += 25
        result = perception.detect_can(bgr, depth, intrinsics, transform)
        np.testing.assert_allclose(result['position'], [0., 0., .1175], atol=.004)

    def test_warm_background_connected_through_missing_depth_is_separate(self):
        bgr, depth, intrinsics, transform = scene()
        bgr[20:100, 20:120] = [20, 35, 210]
        depth[20:100, 20:120] = 600
        cv2.line(bgr, (100, 80), (200, 180), (20, 35, 210), 3)
        cv2.line(depth, (100, 80), (200, 180), 0, 3)
        result = perception.detect_can(bgr, depth, intrinsics, transform)
        np.testing.assert_allclose(result['position'], [0., 0., .1175], atol=.004)

    def test_measures_height_when_upper_body_and_lid_depth_are_missing(self):
        for can_height, radius in ((.135, .025), (.16, .0275)):
            bgr, depth, intrinsics, transform = scene(can_height=can_height, radius=radius)
            v, u = np.indices(depth.shape)
            z = depth * .001
            camera = np.stack(((u - intrinsics[2]) * z / intrinsics[0],
                               (v - intrinsics[3]) * z / intrinsics[1], z), axis=-1)
            arm = camera @ transform[:3, :3].T + transform[:3, 3]
            depth[arm[:, :, 2] > .05 + can_height * .55] = 0
            result = perception.detect_can(bgr, depth, intrinsics, transform, radius=radius)
            self.assertAlmostEqual(result['height'], can_height, delta=.004)
            self.assertAlmostEqual(result['bottom_z'], .05, delta=.003)
            self.assertAlmostEqual(result['top_z'], .05 + can_height, delta=.004)

    def test_missing_color_and_missing_depth_fail(self):
        bgr, depth, intrinsics, transform = scene()
        with self.assertRaisesRegex(perception.PerceptionError, 'No can'):
            perception.detect_can(np.zeros_like(bgr), depth, intrinsics, transform)
        with self.assertRaisesRegex(perception.PerceptionError, 'inadequate depth'):
            perception.detect_can(bgr, np.zeros_like(depth), intrinsics, transform)

    def test_two_cans_are_ambiguous_and_expected_position_gates(self):
        data = scene(centers=((- .07, 0.), (.07, 0.)))
        with self.assertRaisesRegex(perception.PerceptionError, 'multiple'):
            perception.detect_can(*data)
        result = perception.detect_can(*data, expected_xy=[-.07, 0.])
        self.assertAlmostEqual(result['position'][0], -.07, delta=.003)

    def test_red_plane_is_not_a_cylinder(self):
        bgr, depth, intrinsics, transform = scene()
        red = bgr[:, :, 2].astype(int) > bgr[:, :, 0].astype(int) + 20
        depth[red] = 300
        with self.assertRaises(perception.PerceptionError):
            perception.detect_can(bgr, depth, intrinsics, transform)

    def test_clipped_rim_and_inadequate_body_coverage_fail(self):
        bgr, depth, intrinsics, transform = scene()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        rows, cols = np.nonzero(hsv[:, :, 1] > 100)
        kept = np.zeros_like(depth, bool)
        kept[:, int(np.median(cols)) - 1:int(np.median(cols)) + 2] = True
        with self.assertRaisesRegex(perception.PerceptionError, 'coverage|depth'):
            perception.detect_can(bgr, np.where(kept, depth, 0), intrinsics, transform)
        crop = int(rows.min()) + 10
        intrinsics[3] -= crop
        with self.assertRaisesRegex(perception.PerceptionError, 'rim'):
            perception.detect_can(bgr[crop:], depth[crop:], intrinsics, transform, support_z=.05)

    def test_support_prior_allows_base_out_of_view(self):
        bgr, depth, intrinsics, transform = scene(lift=.03, camera_lift=.03)
        result = perception.detect_can(bgr[:200], depth[:200], intrinsics,
                                       transform, support_z=.05)
        self.assertAlmostEqual(result['bottom_z'], .08, delta=.004)
        self.assertAlmostEqual(result['support_z'], .05, delta=.001)

    def test_visual_lift_with_fixed_pixels_confirms_world_translation(self):
        before_image, _, intrinsics, first = scene()
        after_image, _, _, second = scene(lift=.03, camera_lift=.03)
        can = self.detect()
        result = perception.verify_visual_lift(before_image, after_image, intrinsics,
                                               first, second, can, [0., 0., .03])
        self.assertGreaterEqual(result['feature_count'], 8)
        np.testing.assert_allclose(result['displacement'], [0., 0., .03], atol=.003)
        self.assertGreater(result['clearance_m'], .025)

    def test_visual_lift_rejects_stationary_can_and_slip(self):
        before_image, _, intrinsics, first = scene()
        can = self.detect()
        for can_lift in (0., .01):
            after_image, _, _, second = scene(lift=can_lift, camera_lift=.03)
            with self.assertRaises(perception.PerceptionError):
                perception.verify_visual_lift(before_image, after_image, intrinsics,
                                              first, second, can, [0., 0., .03])

    def test_visual_lift_requires_body_texture_and_valid_tracking(self):
        before_image, _, intrinsics, first = scene(texture=False)
        after_image, _, _, second = scene(lift=.03, camera_lift=.03, texture=False)
        can = self.detect()
        with self.assertRaisesRegex(perception.PerceptionError, 'texture|anchors|features'):
            perception.verify_visual_lift(before_image, after_image, intrinsics,
                                          first, second, can, [0., 0., .03])
        before_image, _, _, first = scene()
        with self.assertRaises(perception.PerceptionError):
            perception.verify_visual_lift(before_image, after_image, intrinsics,
                                          first, second, dict(can, error_m=.1), [0., 0., .03])
        with self.assertRaises(perception.PerceptionError):
            perception.verify_visual_lift(before_image, np.zeros_like(before_image), intrinsics,
                                          first, second, can, [0., 0., .03])

    def test_nonrigid_transform_and_wrong_depth_shape_fail(self):
        bgr, depth, intrinsics, transform = scene()
        transform[0, 0] = 2
        with self.assertRaises(perception.PerceptionError):
            perception.detect_can(bgr, depth, intrinsics, transform)
        with self.assertRaises(perception.PerceptionError):
            perception.detect_can(bgr, depth[:20], intrinsics, np.eye(4))


if __name__ == '__main__':
    unittest.main()
