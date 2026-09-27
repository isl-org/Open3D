"""Ground-truth-free RGB-D loop proposal, estimation, and verification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


@dataclass
class LoopKeyframe:
    """Appearance and metric geometry retained for autonomous loop search."""

    node_id: int
    image_points: np.ndarray
    object_points: np.ndarray
    descriptors: np.ndarray
    cloud: o3d.geometry.PointCloud


@dataclass
class LoopResult:
    """A geometrically verified source-keyframe to current-frame constraint."""

    source_id: int
    transformation: np.ndarray
    information: np.ndarray
    matches: int
    pnp_inliers: int
    icp_fitness: float
    icp_rmse: float


@dataclass
class LoopSearchStats:
    """Counts used to explain autonomous place-recognition selectivity."""

    compared: int = 0
    appearance_candidates: int = 0
    geometric_checks: int = 0


def make_keyframe(node_id, color, depth, points, camera, settings):
    """Extract depth-backed ORB features and retain the local frame cloud."""
    gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=settings["orb_features"])
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    empty_descriptors = np.empty((0, 32), dtype=np.uint8)
    if descriptors is None:
        descriptors = empty_descriptors

    image_points, object_points, valid_descriptors = [], [], []
    height, width = depth.shape
    for keypoint, descriptor in zip(keypoints, descriptors):
        u, v = keypoint.pt
        column, row = int(round(u)), int(round(v))
        if not (0 <= column < width and 0 <= row < height):
            continue
        z = float(depth[row, column])
        if not (0.0 < z < settings["max_feature_depth_m"]):
            continue
        x = (u - camera["cx"]) * z / camera["fx"]
        y = (v - camera["cy"]) * z / camera["fy"]
        image_points.append((u, v))
        object_points.append((x, y, z))
        valid_descriptors.append(descriptor)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return LoopKeyframe(
        node_id=node_id,
        image_points=np.asarray(image_points, dtype=np.float32).reshape(-1, 2),
        object_points=np.asarray(object_points, dtype=np.float32).reshape(-1, 3),
        descriptors=(np.asarray(valid_descriptors, dtype=np.uint8).reshape(-1, 32)
                     if valid_descriptors else empty_descriptors),
        cloud=cloud)


def _ratio_matches(candidate, current, ratio):
    if len(candidate.descriptors) < 2 or len(current.descriptors) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(candidate.descriptors, current.descriptors, k=2)
    return [first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < ratio * second.distance]


def _verify_candidate(candidate, current, matches, camera, settings):
    """Estimate candidate-to-current motion with PnP and verify it using ICP."""
    object_points = np.asarray(
        [candidate.object_points[match.queryIdx] for match in matches], dtype=np.float32)
    image_points = np.asarray(
        [current.image_points[match.trainIdx] for match in matches], dtype=np.float32)
    intrinsic = np.asarray([[camera["fx"], 0.0, camera["cx"]],
                            [0.0, camera["fy"], camera["cy"]],
                            [0.0, 0.0, 1.0]], dtype=np.float64)
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points, image_points, intrinsic, None,
        iterationsCount=settings["pnp_iterations"],
        reprojectionError=settings["pnp_reprojection_error_px"],
        confidence=settings["pnp_confidence"], flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None:
        return None
    inlier_count = len(inliers)
    if (inlier_count < settings["min_pnp_inliers"] or
            inlier_count / len(matches) < settings["min_pnp_inlier_ratio"]):
        return None

    initial = np.eye(4)
    initial[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    initial[:3, 3] = translation[:, 0]
    threshold = settings["icp_max_correspondence_distance_m"]
    result = o3d.pipelines.registration.registration_icp(
        candidate.cloud, current.cloud, threshold, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=settings["icp_max_iterations"]))
    if (result.fitness < settings["min_icp_fitness"] or
            result.inlier_rmse > settings["max_icp_rmse_m"]):
        return None

    correction = result.transformation @ np.linalg.inv(initial)
    translation_correction = np.linalg.norm(correction[:3, 3])
    rotation_correction = math_degrees(Rotation.from_matrix(
        correction[:3, :3]).magnitude())
    if (translation_correction > settings["max_icp_translation_correction_m"] or
            rotation_correction > settings["max_icp_rotation_correction_deg"]):
        return None

    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        candidate.cloud, current.cloud, threshold, result.transformation)
    return LoopResult(candidate.node_id, result.transformation.copy(), information,
                      len(matches), inlier_count, float(result.fitness),
                      float(result.inlier_rmse))


def math_degrees(radians):
    return float(np.degrees(radians))


def detect_loop(current, database, camera, settings, last_loop_frame):
    """Rank old keyframes by ORB similarity and verify the strongest candidates."""
    stats = LoopSearchStats()
    if current.node_id - last_loop_frame < settings["cooldown_frames"]:
        return None, stats

    ranked = []
    for candidate in database:
        if current.node_id - candidate.node_id < settings["min_frame_separation"]:
            continue
        stats.compared += 1
        matches = _ratio_matches(candidate, current, settings["ratio_test"])
        denominator = max(1, min(len(candidate.descriptors), len(current.descriptors)))
        score = len(matches) / denominator
        if len(matches) >= settings["min_matches"] and score >= settings["min_match_score"]:
            ranked.append((len(matches), candidate, matches))
    stats.appearance_candidates = len(ranked)

    ranked.sort(key=lambda item: item[0], reverse=True)
    for _, candidate, matches in ranked[:settings["max_candidates"]]:
        stats.geometric_checks += 1
        result = _verify_candidate(candidate, current, matches, camera, settings)
        if result is not None:
            return result, stats
    return None, stats
