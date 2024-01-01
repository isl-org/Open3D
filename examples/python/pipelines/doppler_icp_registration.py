# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Example script to run Doppler ICP point cloud registration.

This script runs Doppler ICP and point-to-plane ICP on DemoDopplerICPSequence.

This is the implementation of the following paper:
B. Hexsel, H. Vhavle, Y. Chen,
DICP: Doppler Iterative Closest Point Algorithm, RSS 2022.

Usage:
python doppler_icp_registration.py [-h] \
    --source SOURCE --target TARGET [--device {cpu,cuda}]
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d
import open3d.t.pipelines.registration as o3d_reg
from pyquaternion import Quaternion


def translation_quaternion_to_transform(translation,
                                        quaternion,
                                        inverse=False,
                                        quat_xyzw=False):
    """Converts translation and WXYZ quaternion to a transformation matrix.

    Args:
        translation: (3,) ndarray representing the translation vector.
        quaternion: (4,) ndarray representing the quaternion.
        inverse: If True, returns the inverse transformation.
        quat_xyzw: If True, this indicates that quaternion is in XYZW format.

    Returns:
        (4, 4) ndarray representing the transformation matrix.
    """
    if quat_xyzw:
        quaternion = np.roll(quaternion, 1)
    transform = Quaternion(quaternion).transformation_matrix  # [w, x, y, z]
    transform[:3, -1] = translation  # [x, y, z]
    return np.linalg.inv(transform) if inverse else transform


def load_tum_file(filename):
    """Loads poses in TUM RGBD format: [timestamp, x, y, z, qx, qy, qz, qw].

    Args:
        filename (string): Path to the TUM poses file.

    Returns:
        A tuple containing an array of 4x4 poses and timestamps.
    """
    # Load the TUM text file.
    data = np.loadtxt(filename, delimiter=' ')
    print('Loaded %d poses from %s (%.2f secs)' %
          (len(data), os.path.basename(filename), data[-1][0] - data[0][0]))

    # Parse timestamps and poses.
    timestamps = data[:, 0]
    poses = np.array([
        translation_quaternion_to_transform(tq[:3], tq[3:], quat_xyzw=True)
        for tq in data[:, 1:]
    ])
    return poses, timestamps


def get_calibration(demo_sequence):
    """Returns the vehicle to sensor calibration transformation and the time
    period (in secs) between sequential point cloud scans.

    Args:
        demo_sequence (DemoDopplerICPSequence): Doppler ICP dataset.

    Returns:
        A tuple of 4x4 array representing the transform, and the period.
    """
    with open(demo_sequence.calibration_path) as f:
        data = json.load(f)

    transform_vehicle_to_sensor = np.array(
        data['transform_vehicle_to_sensor']).reshape(4, 4)
    period = data['period']

    return transform_vehicle_to_sensor, period


def get_trajectory(demo_sequence):
    """Returns the ground truth trajectory of the dataset.

    Args:
        demo_sequence (DemoDopplerICPSequence): Doppler ICP dataset.

    Returns:
        An array of 4x4 poses for this sequence.
    """
    return load_tum_file(demo_sequence.trajectory_path)[0]


def get_ground_truth_pose(demo_sequence, source_idx, target_idx):
    """Returns the ground truth poses from the dataset.

    Args:
        demo_sequence (DemoDopplerICPSequence): Doppler ICP dataset.
        source_idx (int): Index of the source point cloud pose.
        target_idx (int): Index of the target point cloud pose.

    Returns:
        4x4 array representing the transformation between target and source.
    """
    poses = get_trajectory(demo_sequence)
    return np.linalg.inv(poses[target_idx]) @ poses[source_idx]


def run_doppler_icp(args):
    """Runs Doppler ICP on a given pair of point clouds.

    Args:
        args: Command line arguments.
    """
    # Setup data type and device.
    dtype = o3d.core.float32
    device = o3d.core.Device('CUDA:0' if args.device == 'cuda' else 'CPU:0')

    # Load the point clouds.
    demo_sequence = o3d.data.DemoDopplerICPSequence()
    source = o3d.t.io.read_point_cloud(demo_sequence.paths[args.source])
    target = o3d.t.io.read_point_cloud(demo_sequence.paths[args.target])

    # Load the calibration parameters.
    transform_vehicle_to_sensor, period = get_calibration(demo_sequence)

    # Downsample the pointcloud.
    source_in_S = source.uniform_down_sample(5)
    target_in_S = target.uniform_down_sample(5)

    # Transform the Open3D point cloud from sensor to vehicle frame.
    source_in_V = source_in_S.to(device).transform(transform_vehicle_to_sensor)
    target_in_V = target_in_S.to(device).transform(transform_vehicle_to_sensor)

    # Move tensor to device.
    init_transform = o3d.core.Tensor(np.eye(4), device=device)
    transform_vehicle_to_sensor = o3d.core.Tensor(transform_vehicle_to_sensor,
                                                  device=device)

    # Compute normals for target.
    target_in_V.estimate_normals(radius=10.0, max_nn=30)

    # Compute direction vectors on source point cloud frame in sensor frame.
    directions = source_in_S.point.positions.numpy()
    norms = np.tile(np.linalg.norm(directions, axis=1), (3, 1)).T
    directions = directions / norms
    source_in_V.point['directions'] = o3d.core.Tensor(directions, dtype, device)

    # Setup robust kernels.
    kernel = o3d_reg.robust_kernel.RobustKernel(o3d_reg.robust_kernel.TukeyLoss,
                                                scaling_parameter=0.5)

    # Setup convergence criteria.
    criteria = o3d_reg.ICPConvergenceCriteria(relative_fitness=1e-6,
                                              relative_rmse=1e-6,
                                              max_iteration=200)

    # Setup transformation estimator.
    estimator_p2l = o3d_reg.TransformationEstimationPointToPlane(kernel)
    estimator_dicp = o3d_reg.TransformationEstimationForDopplerICP(
        period=period * (args.target - args.source),
        lambda_doppler=0.01,
        reject_dynamic_outliers=False,
        doppler_outlier_threshold=2.0,
        outlier_rejection_min_iteration=2,
        geometric_robust_loss_min_iteration=0,
        doppler_robust_loss_min_iteration=2,
        goemetric_kernel=kernel,
        doppler_kernel=kernel,
        transform_vehicle_to_sensor=transform_vehicle_to_sensor)

    # Run Doppler ICP and point-to-plane ICP registration for comparison.
    max_neighbor_distance = 0.3
    results = [
        o3d_reg.icp(source_in_V, target_in_V, max_neighbor_distance,
                    init_transform, estimator, criteria)
        for estimator in [estimator_p2l, estimator_dicp]
    ]

    # Display the poses.
    np.set_printoptions(suppress=True, precision=4)
    print('Estimated pose from Point-to-Plane ICP [%s iterations]:' %
          results[0].num_iterations)
    print(results[0].transformation.numpy())

    print('\nEstimated pose from Doppler ICP [%s iterations]:' %
          results[1].num_iterations)
    print(results[1].transformation.numpy())

    print('\nGround truth pose:')
    print(get_ground_truth_pose(demo_sequence, args.source, args.target))


def parse_args():
    """Parses the command line arguments.

    Returns:
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        '-s',
                        type=int,
                        required=True,
                        help='Source point cloud index')
    parser.add_argument('--target',
                        '-t',
                        type=int,
                        required=True,
                        help='Target point cloud index')
    parser.add_argument('--device',
                        '-d',
                        default='cpu',
                        help='Device backend for the tensor',
                        choices=['cpu', 'cuda'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_doppler_icp(args)
