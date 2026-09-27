#!/usr/bin/env python3
"""Educational RGB-D localization built on Open3D's official odometry example.

The odometry call and pose-graph convention follow:
examples/python/pipelines/rgbd_odometry.py and
examples/python/reconstruction_system/make_fragments.py in this clone.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

from autonomous_loop import LoopResult, detect_loop, make_keyframe


@dataclass(frozen=True)
class FrameRecord:
    timestamp: float
    color: Path
    depth: Path
    gt_pose: np.ndarray | None


@dataclass
class FrameState:
    index: int
    timestamp: float
    estimated: np.ndarray
    local_odometry: np.ndarray
    corrected: np.ndarray
    optimized_poses: np.ndarray
    ground_truth: np.ndarray | None
    covariance: np.ndarray
    map_points: np.ndarray
    map_colors: np.ndarray
    replace_map: bool
    current_scan: np.ndarray
    color_frame: np.ndarray
    depth_frame: np.ndarray
    timings_ms: dict[str, float]
    odometry_ok: bool
    closure: bool
    loop_source: int | None
    loop_compared: int
    loop_appearance_candidates: int
    loop_geometric_checks: int
    loop_matches: int
    pnp_inliers: int
    icp_fitness: float
    icp_rmse: float
    correction_jump_m: float
    produced_at: float


def read_tum_list(path: Path, columns: int) -> list[tuple[float, list[str]]]:
    """Read a timestamped TUM text file while ignoring comments."""
    rows: list[tuple[float, list[str]]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#") or len(fields) < columns:
                continue
            rows.append((float(fields[0]), fields[1:]))
    return rows


def nearest(rows: list[tuple[float, list[str]]], stamp: float):
    """Return the nearest timestamped row in O(log n)."""
    stamps = [row[0] for row in rows]
    index = bisect.bisect_left(stamps, stamp)
    candidates = rows[max(0, index - 1):min(len(rows), index + 1)]
    return min(candidates, key=lambda row: abs(row[0] - stamp))


def pose_from_tum(values: list[str]) -> np.ndarray:
    """Convert TUM tx ty tz qx qy qz qw into a camera-to-world matrix."""
    numbers = np.asarray(values[:7], dtype=float)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(numbers[3:7]).as_matrix()
    pose[:3, 3] = numbers[:3]
    return pose


def load_records(dataset: Path, max_dt: float) -> list[FrameRecord]:
    """Associate RGB/depth and optionally attach nearest ground-truth poses."""
    rgb = read_tum_list(dataset / "rgb.txt", 2)
    depth = read_tum_list(dataset / "depth.txt", 2)
    ground_truth_path = dataset / "groundtruth.txt"
    ground_truth = (read_tum_list(ground_truth_path, 8)
                    if ground_truth_path.exists() else [])
    records: list[FrameRecord] = []
    for stamp, color_fields in rgb:
        depth_row = nearest(depth, stamp)
        if abs(depth_row[0] - stamp) > max_dt:
            continue
        gt_pose = None
        if ground_truth:
            gt_row = nearest(ground_truth, stamp)
            if abs(gt_row[0] - stamp) <= max_dt:
                gt_pose = pose_from_tum(gt_row[1])
        records.append(FrameRecord(stamp, dataset / color_fields[0],
                                   dataset / depth_row[1][0], gt_pose))
    if not records:
        raise RuntimeError(f"No associated frames found under {dataset}")

    # GT is evaluation-only outside gt-assisted mode; normalize it when available.
    origin = next((record.gt_pose for record in records
                   if record.gt_pose is not None), None)
    if origin is None:
        return records
    origin_inverse = np.linalg.inv(origin)
    return [FrameRecord(r.timestamp, r.color, r.depth,
                        origin_inverse @ r.gt_pose if r.gt_pose is not None else None)
            for r in records]


def make_intrinsic(config: dict) -> o3d.camera.PinholeCameraIntrinsic:
    camera = config["camera"]
    return o3d.camera.PinholeCameraIntrinsic(
        camera["width"], camera["height"], camera["fx"], camera["fy"],
        camera["cx"], camera["cy"])


def read_rgbd(record: FrameRecord, config: dict) -> o3d.geometry.RGBDImage:
    """Load one synchronized pair using TUM's millimeter depth convention."""
    color = o3d.io.read_image(str(record.color))
    depth = o3d.io.read_image(str(record.depth))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=config["dataset"]["depth_scale"],
        depth_trunc=config["dataset"]["depth_trunc_m"],
        convert_rgb_to_intensity=False)


def perturbation(rng: np.random.Generator, config: dict) -> np.ndarray:
    """Draw a small SE(3) increment that accumulates into visible drift."""
    noise = config["noise"]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rng.normal(
        0.0, math.radians(noise["rotation_std_deg_per_frame"]), 3)).as_matrix()
    transform[:3, 3] = rng.normal(0.0, noise["translation_std_m_per_frame"], 3)
    return transform


def initial_covariance(config: dict) -> np.ndarray:
    covariance = config["covariance"]
    translation = covariance["initial_translation_std_m"] ** 2
    rotation = math.radians(covariance["initial_rotation_std_deg"]) ** 2
    return np.diag([translation] * 3 + [rotation] * 3)


def process_noise(config: dict) -> np.ndarray:
    covariance = config["covariance"]
    translation = covariance["process_translation_std_m"] ** 2
    rotation = math.radians(covariance["process_rotation_std_deg"]) ** 2
    return np.diag([translation] * 3 + [rotation] * 3)


def point_chunk(rgbd, intrinsic, config, rng):
    """Create a bounded camera-frame cloud for mapping and live alignment."""
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    cloud = cloud.voxel_down_sample(config["processing"]["voxel_size_m"])
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    limit = config["processing"]["points_per_keyframe"]
    if len(points) > limit:
        selection = rng.choice(len(points), limit, replace=False)
        points, colors = points[selection], colors[selection]
    return points.copy(), colors.copy()


def transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Move camera-frame points into the fixed world/map frame."""
    return points @ pose[:3, :3].T + pose[:3, 3]


def optimize_pose_graph(pose_graph, config):
    """Run Open3D global pose-graph optimization after a loop constraint."""
    settings = config["optimization"]
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=settings["max_correspondence_distance_m"],
        edge_prune_threshold=settings["edge_prune_threshold"],
        preference_loop_closure=settings["preference_loop_closure"],
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)


def processing_loop(records, config, output, stop, pause, seed):
    """Run odometry, optional autonomous/GT graph correction, and mapping."""
    try:
        rng = np.random.default_rng(seed)
        intrinsic = make_intrinsic(config)
        mode = config["slam"]["mode"]
        option = o3d.pipelines.odometry.OdometryOption()
        option.depth_diff_max = config["processing"]["depth_diff_max_m"]
        jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        estimated = np.eye(4)
        local_odometry = np.eye(4)
        corrected = np.eye(4)
        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        keyframes: list[tuple[int, np.ndarray, np.ndarray]] = []
        loop_database = []
        last_loop_frame = -config["autonomous_loop"]["cooldown_frames"]
        covariance = initial_covariance(config)
        process_covariance = process_noise(config)
        previous = read_rgbd(records[0], config)
        fieldnames = ["frame", "timestamp", "io_ms", "odometry_ms", "map_ms",
                      "total_ms", "processing_hz", "odometry_ok", "closure",
                      "loop_source", "loop_compared", "loop_appearance_candidates",
                      "loop_geometric_checks", "loop_matches", "pnp_inliers",
                      "icp_fitness", "icp_rmse", "correction_jump_m", "position_std_m"]

        telemetry_path = Path(__file__).parent / "telemetry.csv"
        with telemetry_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for index, record in enumerate(records):
                while pause.is_set() and not stop.is_set():
                    time.sleep(0.02)
                if stop.is_set():
                    break
                started = time.perf_counter()
                if index == 0:
                    current, io_ms, odometry_ms, odometry_ok = previous, 0.0, 0.0, True
                else:
                    stage = time.perf_counter()
                    current = read_rgbd(record, config)
                    io_ms = (time.perf_counter() - stage) * 1000.0
                    stage = time.perf_counter()
                    odometry_ok, source_to_target, information = (
                        o3d.pipelines.odometry.compute_rgbd_odometry(
                            previous, current, intrinsic, np.eye(4), jacobian, option))
                    odometry_ms = (time.perf_counter() - stage) * 1000.0
                    if odometry_ok:
                        camera_increment = np.linalg.inv(source_to_target)
                        local_odometry = local_odometry @ camera_increment
                        corrected = corrected @ camera_increment
                        estimated = estimated @ camera_increment @ perturbation(rng, config)
                    else:
                        source_to_target = np.eye(4)
                        information = np.eye(6) * 1e-3
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(corrected.copy()))
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        index - 1, index, source_to_target, information, uncertain=False))
                    covariance = covariance + process_covariance

                stage = time.perf_counter()
                local_points, local_colors = point_chunk(current, intrinsic, config, rng)
                is_keyframe = index % config["processing"]["map_every_n_frames"] == 0
                loop_result: LoopResult | None = None
                closure = False
                jump = 0.0
                loop_source = None
                loop_compared = 0
                loop_appearance_candidates = 0
                loop_geometric_checks = 0

                closure_period = config["processing"]["loop_closure_every_n_frames"]
                gt_constraint = (mode == "gt-assisted" and index > 0 and
                                 closure_period > 0 and index % closure_period == 0)
                if gt_constraint:
                    if record.gt_pose is None or records[0].gt_pose is None:
                        raise RuntimeError("GT-assisted mode requires associated ground truth")
                    before = corrected[:3, 3].copy()
                    loop_transform = np.linalg.inv(record.gt_pose) @ records[0].gt_pose
                    loop_information = np.eye(6) * config["optimization"]["loop_information"]
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        0, index, loop_transform, loop_information, uncertain=True))
                    optimize_pose_graph(pose_graph, config)
                    corrected = np.asarray(pose_graph.nodes[-1].pose).copy()
                    jump = float(np.linalg.norm(corrected[:3, 3] - before))
                    closure, loop_source = True, 0
                    covariance *= config["covariance"]["loop_closure_shrink"]

                current_loop_keyframe = None
                if mode == "autonomous" and is_keyframe:
                    current_loop_keyframe = make_keyframe(
                        index, np.asarray(current.color), np.asarray(current.depth),
                        local_points, config["camera"], config["autonomous_loop"])
                    loop_result, search_stats = detect_loop(
                        current_loop_keyframe, loop_database, config["camera"],
                        config["autonomous_loop"], last_loop_frame)
                    loop_compared = search_stats.compared
                    loop_appearance_candidates = search_stats.appearance_candidates
                    loop_geometric_checks = search_stats.geometric_checks
                    if loop_result is not None:
                        before = corrected[:3, 3].copy()
                        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                            loop_result.source_id, index, loop_result.transformation,
                            loop_result.information, uncertain=True))
                        optimize_pose_graph(pose_graph, config)
                        corrected = np.asarray(pose_graph.nodes[-1].pose).copy()
                        jump = float(np.linalg.norm(corrected[:3, 3] - before))
                        closure, loop_source = True, loop_result.source_id
                        last_loop_frame = index
                        covariance *= config["covariance"]["loop_closure_shrink"]
                        print(f"\nAUTONOMOUS LOOP {loop_result.source_id}->{index} | "
                              f"matches {loop_result.matches} | "
                              f"PnP {loop_result.pnp_inliers} | "
                              f"ICP fitness {loop_result.icp_fitness:.3f} | "
                              f"RMSE {loop_result.icp_rmse:.3f} m", flush=True)
                    loop_database.append(current_loop_keyframe)

                current_scan = transform_points(local_points, corrected)
                if is_keyframe:
                    keyframes.append((index, local_points, local_colors))
                if closure:
                    rebuilt = [(transform_points(p, np.asarray(pose_graph.nodes[i].pose)), c)
                               for i, p, c in keyframes]
                    points = np.vstack([item[0] for item in rebuilt])
                    colors = np.vstack([item[1] for item in rebuilt])
                elif is_keyframe:
                    points, colors = current_scan, local_colors
                else:
                    points, colors = np.empty((0, 3)), np.empty((0, 3))
                map_ms = (time.perf_counter() - stage) * 1000.0
                total_ms = (time.perf_counter() - started) * 1000.0
                timings = {"io": io_ms, "odometry": odometry_ms, "map": map_ms,
                           "total": total_ms, "hz": 1000.0 / max(total_ms, 1e-6)}
                position_std = float(np.sqrt(np.max(np.linalg.eigvalsh(covariance[:3, :3]))))
                writer.writerow({
                    "frame": index, "timestamp": record.timestamp, "io_ms": f"{io_ms:.2f}",
                    "odometry_ms": f"{odometry_ms:.2f}", "map_ms": f"{map_ms:.2f}",
                    "total_ms": f"{total_ms:.2f}", "processing_hz": f"{timings['hz']:.2f}",
                    "odometry_ok": int(odometry_ok), "closure": int(closure),
                    "loop_source": "" if loop_source is None else loop_source,
                    "loop_compared": loop_compared,
                    "loop_appearance_candidates": loop_appearance_candidates,
                    "loop_geometric_checks": loop_geometric_checks,
                    "loop_matches": 0 if loop_result is None else loop_result.matches,
                    "pnp_inliers": 0 if loop_result is None else loop_result.pnp_inliers,
                    "icp_fitness": "" if loop_result is None else f"{loop_result.icp_fitness:.4f}",
                    "icp_rmse": "" if loop_result is None else f"{loop_result.icp_rmse:.4f}",
                    "correction_jump_m": f"{jump:.4f}", "position_std_m": f"{position_std:.4f}"})
                csv_file.flush()
                optimized_poses = np.asarray([node.pose for node in pose_graph.nodes])
                ground_truth = (record.gt_pose.copy()
                                if record.gt_pose is not None else None)
                state = FrameState(
                    index=index, timestamp=record.timestamp, estimated=estimated.copy(),
                    local_odometry=local_odometry.copy(), corrected=corrected.copy(),
                    optimized_poses=optimized_poses, ground_truth=ground_truth,
                    covariance=covariance.copy(), map_points=points, map_colors=colors,
                    replace_map=closure, current_scan=current_scan,
                    color_frame=np.asarray(current.color).copy(),
                    depth_frame=np.asarray(current.depth).copy(), timings_ms=timings,
                    odometry_ok=odometry_ok, closure=closure, loop_source=loop_source,
                    loop_compared=loop_compared,
                    loop_appearance_candidates=loop_appearance_candidates,
                    loop_geometric_checks=loop_geometric_checks,
                    loop_matches=0 if loop_result is None else loop_result.matches,
                    pnp_inliers=0 if loop_result is None else loop_result.pnp_inliers,
                    icp_fitness=0.0 if loop_result is None else loop_result.icp_fitness,
                    icp_rmse=0.0 if loop_result is None else loop_result.icp_rmse,
                    correction_jump_m=jump, produced_at=time.perf_counter())
                while not stop.is_set():
                    try:
                        output.put(state, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                previous = current
        pose_graph_path = Path(config["_output_dir"]) / config["output"]["pose_graph_file"]
        pose_graph_path.unlink(missing_ok=True)
        o3d.io.write_pose_graph(str(pose_graph_path), pose_graph)
        if not pose_graph_path.is_file():
            raise RuntimeError(f"Could not save pose graph: {pose_graph_path}")
    except Exception as error:  # Propagate worker failures to the main thread.
        output.put(error)
    finally:
        output.put(None)


def line_set(points, color):
    """Build a trajectory LineSet, including a harmless initial zero segment."""
    shown = points if len(points) > 1 else [points[0], points[0]]
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(np.asarray(shown))
    geometry.lines = o3d.utility.Vector2iVector(
        np.column_stack((np.arange(len(shown) - 1), np.arange(1, len(shown)))))
    geometry.colors = o3d.utility.Vector3dVector(
        np.tile(color, (len(shown) - 1, 1)))
    return geometry


def replace_line_set(target, source):
    target.points, target.lines, target.colors = source.points, source.lines, source.colors


def covariance_ellipsoid(pose, covariance, sigma):
    """Create a wire ellipsoid from the positional 3x3 covariance block."""
    values, vectors = np.linalg.eigh(covariance[:3, :3])
    radii = sigma * np.sqrt(np.maximum(values, 1e-10))
    points, lines = [], []
    samples = 32
    for plane in range(3):
        start = len(points)
        for sample in range(samples):
            angle = 2.0 * math.pi * sample / samples
            unit = np.zeros(3)
            unit[(plane + 1) % 3] = math.cos(angle)
            unit[(plane + 2) % 3] = math.sin(angle)
            points.append(pose[:3, 3] + pose[:3, :3] @ vectors @ (radii * unit))
            lines.append([start + sample, start + (sample + 1) % samples])
    ellipsoid = o3d.geometry.LineSet()
    ellipsoid.points = o3d.utility.Vector3dVector(np.asarray(points))
    ellipsoid.lines = o3d.utility.Vector2iVector(np.asarray(lines))
    ellipsoid.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.75, 0.0], (len(lines), 1)))
    return ellipsoid


def make_frustum(pose, intrinsic, scale):
    camera = intrinsic.intrinsic_matrix
    width, height = intrinsic.width, intrinsic.height
    frustum = o3d.geometry.LineSet.create_camera_visualization(
        width, height, camera, np.linalg.inv(pose), scale)
    frustum.paint_uniform_color([0.1, 1.0, 1.0])
    return frustum


def print_telemetry(state, end_to_end_ms, viewer_hz):
    status = "LOOP" if state.closure else ("OK" if state.odometry_ok else "ODOM FAIL")
    print(f"\rframe {state.index:03d} | {status:9s} | process {state.timings_ms['total']:6.1f} ms "
          f"({state.timings_ms['hz']:5.1f} Hz) | odom {state.timings_ms['odometry']:6.1f} ms | "
          f"map {state.timings_ms['map']:5.1f} ms | e2e {end_to_end_ms:5.1f} ms | "
          f"viewer {viewer_hz:5.1f} Hz | jump {state.correction_jump_m:.3f} m", end="", flush=True)


def configure_window_backend(viewer):
    """Use XWayland when Open3D's legacy GLFW viewer cannot initialize on Wayland."""
    if (viewer and os.environ.get("XDG_SESSION_TYPE") == "wayland" and
            os.environ.get("DISPLAY")):
        os.environ["XDG_SESSION_TYPE"] = "x11"
        print("Open3D viewer: using XWayland compatibility backend")


def make_image_panel(state, config, paused=False):
    """Show the exact RGB and metric depth pair used by dense odometry."""
    color = cv2.cvtColor(state.color_frame, cv2.COLOR_RGB2BGR)
    depth_limit = config["dataset"]["depth_trunc_m"]
    depth_8bit = np.clip(state.depth_frame / depth_limit * 255.0, 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
    depth_color[state.depth_frame <= 0] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(color, "RGB frame used by odometry", (15, 30), font, 0.7,
                (40, 255, 40), 2, cv2.LINE_AA)
    cv2.putText(color, "SPACE pause/resume | Q or Esc quit", (15, 60), font,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(depth_color, f"Depth: 0-{depth_limit:.1f} m (dense pixels)",
                (15, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    status = "LOOP CLOSURE" if state.closure else (
        "ODOMETRY OK" if state.odometry_ok else "ODOMETRY FAILED")
    caption = (f"frame {state.index} | {status} | odom {state.timings_ms['odometry']:.0f} ms | "
               f"{state.timings_ms['hz']:.1f} Hz | map xyz "
               f"[{state.corrected[0, 3]:+.2f}, {state.corrected[1, 3]:+.2f}, "
               f"{state.corrected[2, 3]:+.2f}] m")
    panel = np.hstack((color, depth_color))
    cv2.rectangle(panel, (0, panel.shape[0] - 38),
                  (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
    cv2.putText(panel, caption, (15, panel.shape[0] - 12), font, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    if paused:
        cv2.rectangle(panel, (panel.shape[1] // 2 - 145, 190),
                      (panel.shape[1] // 2 + 145, 285), (0, 0, 0), -1)
        cv2.putText(panel, "PAUSED", (panel.shape[1] // 2 - 105, 240), font,
                    1.5, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, "SPACE to resume", (panel.shape[1] // 2 - 105, 270),
                    font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_outputs(map_points, map_colors, trajectories, config):
    """Persist the final displayed map and trajectory arrays for later use."""
    output_dir = Path(config["_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    map_summary = {"saved_points": 0, "extent_m": [0.0, 0.0, 0.0]}
    if len(map_points):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(map_points)
        cloud.colors = o3d.utility.Vector3dVector(map_colors)
        voxel_size = config["output"]["map_voxel_size_m"]
        if voxel_size > 0:
            cloud = cloud.voxel_down_sample(voxel_size)
        map_path = output_dir / config["output"]["map_file"]
        if not o3d.io.write_point_cloud(str(map_path), cloud, compressed=True):
            raise RuntimeError(f"Could not save map: {map_path}")
        print(f"Saved map: {map_path} ({len(cloud.points)} points)")
        saved_points = np.asarray(cloud.points)
        map_summary = {
            "saved_points": len(saved_points),
            "extent_m": np.ptp(saved_points, axis=0).round(6).tolist(),
        }

    trajectory_path = output_dir / config["output"]["trajectory_file"]
    np.savez_compressed(
        trajectory_path,
        noisy=np.asarray(trajectories["estimated"]),
        local_odometry=np.asarray(trajectories["local_odometry"]),
        optimized=np.asarray(trajectories["corrected"]),
        ground_truth=np.asarray(trajectories["ground_truth"]))
    print(f"Saved trajectories: {trajectory_path}")
    return map_summary


def path_length(poses):
    """Return translation accumulated along a camera-pose sequence."""
    if len(poses) < 2:
        return 0.0
    positions = np.asarray(poses)[:, :3, 3]
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def accuracy_kpis(poses, ground_truth):
    """Compute absolute and relative pose errors on GT-associated frames."""
    indices = [index for index in sorted(ground_truth) if index < len(poses)]
    if not indices:
        return None
    absolute = [np.linalg.norm(poses[index][:3, 3] - ground_truth[index][:3, 3])
                for index in indices]
    relative_translation = []
    relative_rotation = []
    for first, second in zip(indices, indices[1:]):
        estimated_delta = np.linalg.inv(poses[first]) @ poses[second]
        truth_delta = np.linalg.inv(ground_truth[first]) @ ground_truth[second]
        error = np.linalg.inv(truth_delta) @ estimated_delta
        relative_translation.append(np.linalg.norm(error[:3, 3]))
        relative_rotation.append(math.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))

    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    return {
        "associated_frames": len(indices),
        "ate_rmse_m": rmse(absolute),
        "ate_median_m": float(np.median(absolute)),
        "ate_max_m": float(np.max(absolute)),
        "rpe_translation_rmse_m": rmse(relative_translation),
        "rpe_rotation_rmse_deg": rmse(relative_rotation),
        "final_position_error_m": float(absolute[-1]),
    }


def build_run_summary(samples, optimized_poses, ground_truth, map_summary,
                      elapsed_s, config):
    """Build the machine-readable KPI report from final graph and run samples."""
    timings = {name: np.asarray([sample["timings"][name] for sample in samples])
               for name in ("io", "odometry", "map", "total")}
    transitions = samples[1:]
    successful = sum(sample["odometry_ok"] for sample in transitions)
    closures = [sample for sample in samples if sample["closure"]]
    autonomous_closures = [sample for sample in closures if sample["loop_matches"] > 0]
    geometric_checks = sum(sample["loop_geometric_checks"] for sample in samples)
    local_poses = np.asarray([sample["local_pose"] for sample in samples])

    def latency(values):
        return {
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }

    summary = {
        "run": {
            "mode": config["slam"]["mode"],
            "frames_processed": len(samples),
            "elapsed_s": float(elapsed_s),
            "ground_truth_available": bool(ground_truth),
        },
        "tracking": {
            "transitions_attempted": len(transitions),
            "successful_transitions": successful,
            "success_rate_percent": (100.0 * successful / len(transitions)
                                     if transitions else 100.0),
        },
        "performance": {
            "effective_throughput_hz": len(samples) / max(elapsed_s, 1e-9),
            "compute_throughput_hz": 1000.0 / max(float(np.mean(timings["total"])), 1e-9),
            "latency": {name: latency(values) for name, values in timings.items()},
        },
        "loop_closure": {
            "accepted": len(closures),
            "database_comparisons": sum(sample["loop_compared"] for sample in samples),
            "appearance_candidates": sum(
                sample["loop_appearance_candidates"] for sample in samples),
            "geometric_checks": geometric_checks,
            "autonomous_acceptance_rate_percent": (
                100.0 * len(autonomous_closures) / geometric_checks
                if geometric_checks else None),
            "mean_matches": (float(np.mean([sample["loop_matches"]
                                             for sample in autonomous_closures]))
                             if autonomous_closures else None),
            "mean_pnp_inliers": (float(np.mean([sample["pnp_inliers"]
                                                 for sample in autonomous_closures]))
                                 if autonomous_closures else None),
            "mean_pnp_inlier_ratio": (float(np.mean([
                sample["pnp_inliers"] / sample["loop_matches"]
                for sample in autonomous_closures]))
                                      if autonomous_closures else None),
            "mean_icp_fitness": (float(np.mean([sample["icp_fitness"]
                                                 for sample in autonomous_closures]))
                                 if autonomous_closures else None),
            "mean_icp_rmse_m": (float(np.mean([sample["icp_rmse"]
                                                for sample in autonomous_closures]))
                                if autonomous_closures else None),
            "mean_correction_m": (float(np.mean([sample["correction_jump_m"]
                                                  for sample in closures]))
                                  if closures else 0.0),
            "max_correction_m": (max(sample["correction_jump_m"] for sample in closures)
                                 if closures else 0.0),
        },
        "trajectory": {
            "local_path_length_m": path_length(local_poses),
            "optimized_path_length_m": path_length(optimized_poses),
        },
        "map": map_summary,
    }
    if ground_truth:
        summary["accuracy"] = {
            "local_odometry": accuracy_kpis(local_poses, ground_truth),
            "optimized": accuracy_kpis(optimized_poses, ground_truth),
        }
    return summary


def print_and_save_summary(summary, config):
    """Print the high-value KPIs and persist the complete report as JSON."""
    def metric(value, precision):
        return "n/a" if value is None else f"{value:.{precision}f}"

    run = summary["run"]
    tracking = summary["tracking"]
    performance = summary["performance"]
    loops = summary["loop_closure"]
    trajectory = summary["trajectory"]
    map_kpis = summary["map"]
    total_latency = performance["latency"]["total"]
    print("\n=== Run KPI Summary ===")
    print(f"Mode / frames / elapsed: {run['mode']} / {run['frames_processed']} / "
          f"{run['elapsed_s']:.2f} s")
    print(f"Tracking success: {tracking['successful_transitions']}/"
          f"{tracking['transitions_attempted']} ({tracking['success_rate_percent']:.1f}%)")
    print(f"Total latency mean / p95: {total_latency['mean_ms']:.1f} / "
          f"{total_latency['p95_ms']:.1f} ms | compute throughput "
          f"{performance['compute_throughput_hz']:.2f} Hz")
    print(f"Loops accepted / geometric checks / DB comparisons: {loops['accepted']} / "
          f"{loops['geometric_checks']} / {loops['database_comparisons']}")
    if loops["mean_icp_fitness"] is not None:
        print(f"Autonomous acceptance / PnP inlier ratio / ICP fitness / RMSE: "
              f"{metric(loops['autonomous_acceptance_rate_percent'], 1)}% / "
              f"{metric(loops['mean_pnp_inlier_ratio'], 3)} / "
              f"{metric(loops['mean_icp_fitness'], 3)} / "
              f"{metric(loops['mean_icp_rmse_m'], 4)} m")
    print(f"Loop correction mean / max: {loops['mean_correction_m']:.4f} / "
          f"{loops['max_correction_m']:.4f} m")
    print(f"Path length local / optimized: {trajectory['local_path_length_m']:.3f} / "
          f"{trajectory['optimized_path_length_m']:.3f} m")
    print(f"Saved map points / extent XYZ: {map_kpis['saved_points']} / "
          f"{map_kpis['extent_m']} m")
    if "accuracy" in summary:
        for name, metrics in summary["accuracy"].items():
            print(f"{name.replace('_', ' ').title()} ATE RMSE / RPE trans / RPE rot: "
                  f"{metric(metrics['ate_rmse_m'], 4)} m / "
                  f"{metric(metrics['rpe_translation_rmse_m'], 4)} m / "
                  f"{metric(metrics['rpe_rotation_rmse_deg'], 3)} deg")
    else:
        print("GT accuracy: unavailable (no groundtruth.txt; estimation remained autonomous)")

    summary_path = (Path(config["_output_dir"]) /
                    config["output"].get("run_summary_file", "run_summary.json"))
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved run summary: {summary_path}")


def consume(output, stop, pause, config, intrinsic, viewer, show_images, realtime):
    """Consume worker states either headlessly or in Open3D's main-thread GUI."""
    run_started = time.perf_counter()
    visualizer = None
    map_geometry = o3d.geometry.PointCloud()
    scan_geometry = o3d.geometry.PointCloud()
    map_points = np.empty((0, 3))
    map_colors = np.empty((0, 3))
    trajectories = {"estimated": [], "local_odometry": [], "corrected": [],
                    "ground_truth": []}
    samples = []
    ground_truth_poses = {}
    optimized_poses = np.empty((0, 4, 4))
    last_view = time.perf_counter()
    wall_start = None
    view_fitted = False
    state = None
    pause_started = None

    def toggle_pause(_visualizer=None):
        """Pause computation/playback while leaving both GUI event loops active."""
        nonlocal pause_started, wall_start
        now = time.perf_counter()
        if pause.is_set():
            pause.clear()
            if wall_start is not None and pause_started is not None:
                wall_start += now - pause_started
            pause_started = None
            print("\nRESUMED | Space: pause", flush=True)
        else:
            pause.set()
            pause_started = now
            print("\nPAUSED | 3D controls remain active | Space: resume", flush=True)
        return False

    def request_exit(_visualizer=None):
        stop.set()
        return False

    if viewer:
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        visualizer.register_key_callback(32, toggle_pause)  # GLFW space.
        visualizer.register_key_callback(256, request_exit)  # GLFW escape.
        size = config["visualization"]
        if not visualizer.create_window("RGB-D SLAM | SPACE pause | H controls | R reset view",
                                        size["window_width"], size["window_height"]):
            stop.set()
            raise RuntimeError("Open3D could not create a window; use --no-viewer over SSH/headless")
        map_geometry.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        map_geometry.colors = o3d.utility.Vector3dVector(np.ones((1, 3)))
        visualizer.add_geometry(map_geometry)
        scan_geometry.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        scan_geometry.colors = o3d.utility.Vector3dVector(np.asarray([[0.0, 1.0, 1.0]]))
        visualizer.add_geometry(scan_geometry)
        visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))
        path_geometries = {
            "estimated": line_set([np.zeros(3)], [1.0, 0.1, 0.1]),
            "local_odometry": line_set([np.zeros(3)], [1.0, 0.55, 0.0]),
            "corrected": line_set([np.zeros(3)], [0.1, 1.0, 0.1]),
        }
        if config["_has_ground_truth"]:
            path_geometries["ground_truth"] = line_set(
                [np.zeros(3)], [0.2, 0.4, 1.0])
        for geometry in path_geometries.values():
            visualizer.add_geometry(geometry)
        frustum = make_frustum(np.eye(4), intrinsic, size["frustum_scale"])
        ellipsoid = covariance_ellipsoid(np.eye(4), initial_covariance(config),
                                         config["covariance"]["ellipsoid_sigma"])
        visualizer.add_geometry(frustum)
        visualizer.add_geometry(ellipsoid)
        camera_marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=size["camera_marker_radius_m"])
        camera_marker.compute_vertex_normals()
        camera_marker.paint_uniform_color([1.0, 0.0, 1.0])
        visualizer.add_geometry(camera_marker)
        camera_center = np.zeros(3)
        visualizer.get_render_option().point_size = size["point_size"]
        visualizer.get_render_option().line_width = size["trajectory_width"]
        print("3D controls | left-drag: orbit/tilt | Ctrl+left-drag or middle-drag: pan | "
              "wheel: zoom | R: reset | H: Open3D help | Space: pause", flush=True)

    while not stop.is_set():
        if visualizer:
            if not visualizer.poll_events():
                stop.set()
                break
            visualizer.update_renderer()

        if show_images and state is not None:
            cv2.imshow("TUM RGB-D input | Space: pause | Q or Esc: quit",
                       make_image_panel(state, config, pause.is_set()))
        key = cv2.waitKey(10) & 0xFF if show_images else -1
        if key == ord(" "):
            toggle_pause()
        elif key in (ord("q"), 27):
            stop.set()
            break

        if pause.is_set():
            time.sleep(0.01)
            continue

        try:
            item = output.get(timeout=0.02)
        except queue.Empty:
            continue
        received = time.perf_counter()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        state = item
        if wall_start is None:
            wall_start = received
        if realtime:
            target = wall_start + state.index / config["processing"]["playback_hz"]
            time.sleep(max(0.0, target - time.perf_counter()))

        for name in ("estimated", "local_odometry"):
            trajectories[name].append(getattr(state, name)[:3, 3].copy())
        if state.ground_truth is not None:
            trajectories["ground_truth"].append(state.ground_truth[:3, 3].copy())
            ground_truth_poses[state.index] = state.ground_truth.copy()
        optimized_poses = state.optimized_poses.copy()
        trajectories["corrected"] = list(optimized_poses[:, :3, 3])
        samples.append({
            "timings": state.timings_ms.copy(),
            "odometry_ok": state.odometry_ok,
            "closure": state.closure,
            "loop_compared": state.loop_compared,
            "loop_appearance_candidates": state.loop_appearance_candidates,
            "loop_geometric_checks": state.loop_geometric_checks,
            "loop_matches": state.loop_matches,
            "pnp_inliers": state.pnp_inliers,
            "icp_fitness": state.icp_fitness,
            "icp_rmse": state.icp_rmse,
            "correction_jump_m": state.correction_jump_m,
            "local_pose": state.local_odometry.copy(),
        })
        if state.replace_map:
            map_points, map_colors = state.map_points.copy(), state.map_colors.copy()
        elif len(state.map_points):
            map_points = np.vstack((map_points, state.map_points))
            map_colors = np.vstack((map_colors, state.map_colors))

        now = time.perf_counter()
        viewer_hz = 1.0 / max(now - last_view, 1e-6)
        end_to_end_ms = max(0.0, (now - state.produced_at) * 1000.0)
        print_telemetry(state, end_to_end_ms, viewer_hz)
        last_view = now

        if visualizer:
            map_geometry.points = o3d.utility.Vector3dVector(map_points)
            map_geometry.colors = o3d.utility.Vector3dVector(map_colors)
            visualizer.update_geometry(map_geometry)
            scan_geometry.points = o3d.utility.Vector3dVector(state.current_scan)
            scan_geometry.colors = o3d.utility.Vector3dVector(
                np.tile([0.0, 1.0, 1.0], (len(state.current_scan), 1)))
            visualizer.update_geometry(scan_geometry)
            colors = {"estimated": [1.0, 0.1, 0.1],
                      "local_odometry": [1.0, 0.55, 0.0],
                      "corrected": [0.1, 1.0, 0.1],
                      "ground_truth": [0.2, 0.4, 1.0]}
            for name, geometry in path_geometries.items():
                replace_line_set(geometry, line_set(trajectories[name], colors[name]))
                visualizer.update_geometry(geometry)
            replace_line_set(frustum, make_frustum(state.corrected, intrinsic, size["frustum_scale"]))
            replace_line_set(ellipsoid, covariance_ellipsoid(
                state.corrected, state.covariance, config["covariance"]["ellipsoid_sigma"]))
            visualizer.update_geometry(frustum)
            visualizer.update_geometry(ellipsoid)
            new_camera_center = state.corrected[:3, 3]
            camera_marker.translate(new_camera_center - camera_center)
            camera_center = new_camera_center.copy()
            visualizer.update_geometry(camera_marker)
            if not view_fitted and len(map_points):
                visualizer.reset_view_point(True)
                view_fitted = True
    print("\nTelemetry saved to telemetry.csv")
    if visualizer:
        visualizer.destroy_window()
    if show_images:
        cv2.destroyAllWindows()
    map_summary = save_outputs(map_points, map_colors, trajectories, config)
    if samples:
        summary = build_run_summary(
            samples, optimized_poses, ground_truth_poses, map_summary,
            time.perf_counter() - run_started, config)
        print_and_save_summary(summary, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--dataset", type=Path, help="Override dataset.path")
    parser.add_argument("--max-frames", type=int, help="Override processing.max_frames")
    parser.add_argument("--no-viewer", action="store_true", help="Run compute/telemetry only")
    parser.add_argument("--no-image-window", action="store_true", help="Show 3D only")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace visualization")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=("autonomous", "gt-assisted", "odometry-only"),
                        help="Override slam.mode from config.yaml")
    closure_group = parser.add_mutually_exclusive_group()
    closure_group.add_argument("--no-loop-closure", action="store_true",
                               help="Disable loop constraints/global optimization")
    closure_group.add_argument("--loop-closure-every", type=int, metavar="N",
                               help="Override the loop-closure interval")
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.mode:
        config["slam"]["mode"] = args.mode
    if args.no_loop_closure:
        config["slam"]["mode"] = "odometry-only"
    elif args.loop_closure_every is not None:
        config["slam"]["mode"] = "gt-assisted"
        config["processing"]["loop_closure_every_n_frames"] = args.loop_closure_every
    mode = config["slam"]["mode"]
    dataset = args.dataset.resolve() if args.dataset else (
        config_path.parent / config["dataset"]["path"]).resolve()
    if not (dataset / "rgb.txt").exists() or not (dataset / "depth.txt").exists():
        raise SystemExit(f"Dataset missing: {dataset}\nRun: python download_dataset.py")
    records = load_records(dataset, config["dataset"]["association_max_dt_s"])
    if mode == "gt-assisted":
        records = [record for record in records if record.gt_pose is not None]
        if not records:
            raise SystemExit("GT-assisted mode requires associated ground-truth poses")
    records = records[::config["processing"]["frame_stride"]]
    maximum = args.max_frames or config["processing"]["max_frames"]
    records = records[:maximum]
    config["_has_ground_truth"] = any(record.gt_pose is not None for record in records)
    output_dir = config_path.parent / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["_output_dir"] = str(output_dir.resolve())
    gt_status = "available for display" if config["_has_ground_truth"] else "not present"
    print(f"Associated {len(records)} frames | mode={mode} | "
          f"map origin=first camera | GT={gt_status}")

    output: queue.Queue = queue.Queue(maxsize=2)
    stop = threading.Event()
    pause = threading.Event()
    worker = threading.Thread(target=processing_loop, name="rgbd-odometry",
                              args=(records, config, output, stop, pause, args.seed), daemon=True)
    configure_window_backend(not args.no_viewer)
    worker.start()
    try:
        consume(output, stop, pause, config, make_intrinsic(config), not args.no_viewer,
                not args.no_viewer and not args.no_image_window, not args.no_realtime)
    finally:
        stop.set()
        worker.join(timeout=5.0)


if __name__ == "__main__":
    main()
