"""RGB-D measurements of a known upright can, without robot or ROS access."""

import cv2
import numpy as np
from scipy.optimize import least_squares


class PerceptionError(ValueError):
    """The image does not contain one measurable can."""


def _camera_points(u, v, depth, intrinsics):
    fx, fy, cx, cy = intrinsics
    z = depth[v, u].astype(float) * .001
    return np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))


def _project(points, intrinsics, transform):
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    fx, fy, cx, cy = intrinsics
    return np.column_stack((camera[:, 0] * fx / camera[:, 2] + cx,
                            camera[:, 1] * fy / camera[:, 2] + cy)), camera[:, 2]


def _rim_height(bgr, center, body_top, radius, intrinsics, transform, support):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    circle = np.column_stack((center[0] + radius * np.cos(angles),
                              center[1] + radius * np.sin(angles),
                              np.zeros_like(angles)))
    scores = []
    heights = np.arange(max(body_top - .003, support + .08),
                        min(body_top + .10, support + .20), .0005)
    if len(heights) < 3:
        raise PerceptionError('Can body and support heights are inconsistent')
    for z in heights:
        circle[:, 2] = z
        pixels, distances = _project(circle, intrinsics, transform)
        u, v = np.rint(pixels).astype(int).T
        valid = ((distances > .08) & (u >= 2) & (u < gray.shape[1] - 2)
                 & (v >= 2) & (v < gray.shape[0] - 2))
        if valid.mean() < .75:
            scores.append(0.)
            continue
        scores.append(float(np.percentile(gradient[v[valid], u[valid]], 40)))
    best = int(np.argmax(scores))
    if scores[best] < 35 or best in (0, len(heights) - 1):
        raise PerceptionError('Can rim is missing, clipped, or not measurable')
    return float(heights[best])


def _support(points, center, bottom):
    distance = np.linalg.norm(points[:, :2] - center, axis=1)
    near = points[(distance > .035) & (distance < .14)
                  & (abs(points[:, 2] - bottom) < .025)]
    if len(near) < 80:
        raise PerceptionError('Support plane has inadequate depth')
    bins = np.floor(near[:, 2] / .003).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    z = (values[np.argmax(counts)] + .5) * .003
    plane = near[abs(near[:, 2] - z) < .004]
    if (len(plane) < 80 or np.ptp(plane[:, 0]) < .025
            or np.ptp(plane[:, 1]) < .025):
        raise PerceptionError('Support plane has inadequate spatial coverage')
    fit = least_squares(lambda p: plane[:, 2] - plane[:, :2] @ p[:2] - p[2],
                        [0., 0., z], loss='soft_l1', f_scale=.001)
    if (np.linalg.norm(fit.x[:2]) > .10
            or np.percentile(abs(fit.fun), 90) > .004):
        raise PerceptionError('Support surface is not a horizontal plane')
    return float(np.dot(center, fit.x[:2]) + fit.x[2])


def detect_can(bgr, depth_mm, intrinsics, camera_to_arm, expected_xy=None,
               *, radius=.025, height=.135, support_z=None):
    """Return an upright can measurement in meters, or raise PerceptionError.

    Intrinsics are (fx, fy, cx, cy). Depth must be aligned with the BGR image.
    Initial height is measured from the rim and support plane. A support_z prior
    permits views with the base outside the image and uses the supplied height;
    rim height is always measured independently. expected_xy gates to 8 cm.
    """
    bgr, depth = np.asarray(bgr), np.asarray(depth_mm)
    intrinsics, transform = np.asarray(intrinsics, float), np.asarray(camera_to_arm, float)
    if (bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8
            or depth.shape != bgr.shape[:2] or not np.issubdtype(depth.dtype, np.number)):
        raise PerceptionError('Expected BGR8 and aligned depth in millimeters')
    if (intrinsics.shape != (4,) or not np.isfinite(intrinsics).all()
            or np.any(intrinsics[:2] <= 0) or transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
        raise PerceptionError('Invalid intrinsics or rigid camera transform')
    if not (.015 <= radius <= .04 and .08 <= height <= .20):
        raise PerceptionError('Can dimensions are outside the supported range')
    if support_z is not None and not np.isfinite(support_z):
        raise PerceptionError('Invalid support height')
    if expected_xy is not None:
        expected_xy = np.asarray(expected_xy, float)
        if expected_xy.shape != (2,) or not np.isfinite(expected_xy).all():
            raise PerceptionError('Invalid expected can position')

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    warm = (((hue < 13) | (hue > 168) | ((hue > 15) & (hue < 40)))
            & (saturation > 95) & (value > 35)).astype(np.uint8)
    grouped = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    valid_depth = np.isfinite(depth) & (depth >= 80) & (depth <= 1500)
    if np.count_nonzero(warm) >= 80 and np.count_nonzero((warm > 0) & valid_depth) < 80:
        raise PerceptionError('Can body has inadequate depth')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        grouped * valid_depth.astype(np.uint8))
    v_all, u_all = np.nonzero(valid_depth)
    camera_all = _camera_points(u_all, v_all, depth, intrinsics)
    arm_all = camera_all @ transform[:3, :3].T + transform[:3, 3]
    candidates = []
    failures = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 80:
            continue
        v, u = np.nonzero((labels == index) & (warm > 0) & valid_depth)
        if len(u) < 80:
            failures.append('Can body has inadequate depth')
            continue
        points = _camera_points(u, v, depth, intrinsics) @ transform[:3, :3].T + transform[:3, 3]
        points = points[::max(1, len(points) // 2500)]
        xy = points[:, :2]
        median = np.median(xy, axis=0)
        direction = median - transform[:2, 3]
        direction /= max(np.linalg.norm(direction), 1e-9)
        fit = least_squares(lambda c: np.linalg.norm(xy - c, axis=1) - radius,
                            median + radius * direction, loss='soft_l1', f_scale=.002)
        residuals = abs(fit.fun)
        inliers = points[residuals < .004]
        if len(inliers) < 80 or len(inliers) < .7 * len(points):
            failures.append('Warm object does not fit the can cylinder')
            continue
        fit = least_squares(lambda c: np.linalg.norm(inliers[:, :2] - c, axis=1) - radius,
                            fit.x, loss='soft_l1', f_scale=.001)
        center = fit.x
        if expected_xy is not None and np.linalg.norm(center - expected_xy) > .08:
            continue
        angles = np.arctan2(inliers[:, 1] - center[1], inliers[:, 0] - center[0])
        wrapped = np.r_[np.sort(angles), np.min(angles) + 2 * np.pi]
        angular_span = 2 * np.pi - np.max(np.diff(wrapped))
        if angular_span < np.deg2rad(40) or np.ptp(inliers[:, 2]) < .015:
            failures.append('Can body has inadequate geometric coverage')
            continue
        error = float(np.sqrt(np.mean(fit.fun ** 2)))
        if error > .0025:
            failures.append('Can cylinder fit is inaccurate')
            continue
        try:
            body_bottom, body_top = np.percentile(inliers[:, 2], [1, 99])
            support = (_support(arm_all, center, body_bottom) if support_z is None
                       else float(support_z))
            top = _rim_height(bgr, center, body_top, radius, intrinsics, transform, support)
            measured_height = top - support if support_z is None else height
            bottom = top - measured_height
            if not .08 <= measured_height <= .20 or bottom < support - .012:
                raise PerceptionError('Can height does not agree with the support plane')
        except PerceptionError as exc:
            failures.append(str(exc))
            continue
        position = np.r_[center, top - measured_height * .5]
        local = (position - transform[:3, 3]) @ transform[:3, :3]
        x, y, width, bbox_height = stats[index, :4]
        candidates.append(dict(position=position.tolist(), top_z=top, bottom_z=bottom,
                               radius=float(radius), height=float(measured_height),
                               support_z=support,
                               depth_sample_count=len(inliers), error_m=error,
                               bbox=[int(x), int(y), int(width), int(bbox_height)],
                               camera_position=local.tolist()))
    if len(candidates) != 1:
        if len(candidates) > 1:
            raise PerceptionError('Ambiguous image: multiple can candidates')
        raise PerceptionError('; '.join(dict.fromkeys(failures)) or 'No can body found')
    return candidates[0]


def verify_visual_lift(before_bgr, after_bgr, intrinsics, before_camera_to_arm,
                       after_camera_to_arm, can, tool_delta, minimum_clearance=.015):
    """Measure can motion from tracked body texture when held inside depth range."""
    before, after = np.asarray(before_bgr), np.asarray(after_bgr)
    intrinsics = np.asarray(intrinsics, float)
    first, second = np.asarray(before_camera_to_arm, float), np.asarray(after_camera_to_arm, float)
    try:
        center = np.asarray(can['position'], float)
        radius, bottom, top = float(can['radius']), float(can['bottom_z']), float(can['top_z'])
        support = float(can['support_z'])
        quality = float(can['error_m'])
        sample_count = float(can['depth_sample_count'])
        measured_height = float(can['height'])
        tool = np.asarray(tool_delta, float)
    except (KeyError, TypeError, ValueError) as exc:
        raise PerceptionError('Visual lift needs a measured can pose') from exc
    if (before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3
            or before.dtype != np.uint8 or after.dtype != np.uint8
            or intrinsics.shape != (4,) or np.any(intrinsics[:2] <= 0)
            or center.shape != (3,) or tool.shape != (3,) or first.shape != (4, 4)
            or second.shape != (4, 4)
            or not np.isfinite(np.r_[center, tool, radius, bottom, top, support, quality,
                                     sample_count, measured_height, intrinsics, first.ravel(),
                                     second.ravel(), minimum_clearance]).all()
            or not .015 <= radius <= .04 or not .08 <= top - bottom <= .20
            or minimum_clearance <= 0 or sample_count < 80 or not 0 <= quality <= .0025
            or abs(measured_height - (top - bottom)) > .001
            or abs(center[2] - (top + bottom) / 2) > .002):
        raise PerceptionError('Invalid visual lift inputs')
    for transform in (first, second):
        if (not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
            raise PerceptionError('Invalid visual lift camera transform')
    hsv = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
    mask = ((((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 168)
              | ((hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40)))
             & (hsv[:, :, 1] > 95) & (hsv[:, :, 2] > 35)).astype(np.uint8) * 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    gray_first = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_second = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray_first, maxCorners=160, qualityLevel=.015,
                                       minDistance=5, mask=mask, blockSize=5)
    if features is None or len(features) < 8:
        raise PerceptionError('Insufficient can body texture for visual lift')
    pixels = features[:, 0, :].astype(float)
    fx, fy, cx, cy = intrinsics
    rays = np.column_stack(((pixels[:, 0] - cx) / fx,
                            (pixels[:, 1] - cy) / fy, np.ones(len(pixels)))) @ first[:3, :3].T
    origin = first[:3, 3]
    offset = origin[:2] - center[:2]
    a = np.sum(rays[:, :2] ** 2, axis=1)
    b = 2 * rays[:, :2] @ offset
    c = np.sum(offset ** 2) - radius ** 2
    discriminant = b * b - 4 * a * c
    distance = (-b - np.sqrt(np.maximum(discriminant, 0))) / np.maximum(2 * a, 1e-12)
    anchors = origin + distance[:, None] * rays
    valid = ((discriminant > 0) & (distance > .02) & (anchors[:, 2] > bottom + .004)
             & (anchors[:, 2] < top - .004))
    features, anchors = features[valid], anchors[valid]
    if len(features) < 8:
        raise PerceptionError('Insufficient can cylinder feature anchors')
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_first, gray_second, features, None, winSize=(31, 31), maxLevel=4)
    if tracked is None:
        raise PerceptionError('Can body features could not be tracked')
    returned, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_second, gray_first, tracked, None, winSize=(31, 31), maxLevel=4)
    if returned is None:
        raise PerceptionError('Can body feature tracking is inconsistent')
    valid = ((status[:, 0] != 0) & (back_status[:, 0] != 0)
             & (np.linalg.norm(returned[:, 0] - features[:, 0], axis=1) < 1.))
    anchors, observed = anchors[valid], tracked[valid, 0].astype(float)
    original_pixels = features[valid, 0]
    if len(anchors) < 8 or np.min(np.ptp(original_pixels, axis=0)) < 12:
        raise PerceptionError('Insufficient distinct tracked can body features')

    def residual(delta):
        projected, z = _project(anchors + delta, intrinsics, second)
        if np.any(z < .02):
            return np.full(observed.size, 10000.)
        return (projected - observed).ravel()

    fit = least_squares(residual, [0., 0., 0.], bounds=(-.12, .12),
                        loss='soft_l1', f_scale=1.)
    errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    inliers = errors < 2.
    if np.count_nonzero(inliers) < 8 or np.mean(inliers) < .7:
        raise PerceptionError('Can visual motion has inadequate reprojection agreement')
    anchors, observed = anchors[inliers], observed[inliers]
    fit = least_squares(residual, fit.x, bounds=(-.12, .12), loss='soft_l1', f_scale=.5)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 150:
        raise PerceptionError('Can visual translation is ambiguous')
    displacement = fit.x
    mismatch = float(np.linalg.norm(displacement - tool))
    clearance = bottom + displacement[2] - support
    error = float(np.sqrt(np.mean(residual(displacement) ** 2)))
    if (error > 1.5 or tool[2] < minimum_clearance or displacement[2] < minimum_clearance
            or displacement[2] < .7 * tool[2] or clearance < minimum_clearance
            or mismatch > .010 or np.linalg.norm(displacement[:2] - tool[:2]) > .008):
        raise PerceptionError('Can visual motion does not confirm support clearance')
    return dict(verified=True, displacement=displacement.tolist(), clearance_m=float(clearance),
                tool_error_m=mismatch, feature_count=len(anchors), reprojection_error_px=error)
