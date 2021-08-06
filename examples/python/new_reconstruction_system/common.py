# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/common.py

import sys, os
import time
import numpy as np
import open3d as o3d
import argparse
import glob


def load_image_file_names(path_dataset, config):
    depth_folder = os.path.join(path_dataset, config.input.depth_folder)
    color_folder = os.path.join(path_dataset, config.input.color_folder)

    # Only 16-bit png depth is supported
    depth_file_names = glob.glob(os.path.join(depth_folder, '*.png'))
    n_depth = len(depth_file_names)
    if n_depth == 0:
        print('Depth image not found in {}, abort!'.format(depth_folder))
        return [], []

    # Try png
    extensions = ['*.png', '*.jpg']
    for ext in extensions:
        color_file_names = glob.glob(os.path.join(color_folder, ext))
        if len(color_file_names) == n_depth:
            return sorted(depth_file_names), sorted(color_file_names)

    print(
        'Found {} depth images, but cannot find matched number of color images with extensions {}, abort!'
        .format(n_depth, extensions))
    return [], []


def load_intrinsic(path_intrinsic, config):
    if path_intrinsic is None or path_intrinsic == '':
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    else:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(path_intrinsic)

    if config.engine == 'legacy':
        return intrinsic
    elif config.engine == 'tensor':
        return o3d.core.Tensor(intrinsic.intrinsic_matrix,
                               o3d.core.Dtype.Float32)
    else:
        print('Unsupported engine {}'.format(config.engine))


def load_extrinsics(path_trajectory, config):
    extrinsics = []

    # For either a fragment or a scene
    if path_trajectory.endswith('log'):
        data = o3d.io.read_pinhole_camera_trajectory(path_trajectory)
        for param in data.parameters:
            extrinsics.append(param.extrinsic)

    # Only for a fragment
    elif path_trajectory.endswith('json'):
        data = o3d.io.read_pose_graph(path_trajectory)
        for node in data.nodes:
            extrinsics.append(np.linalg.inv(node.pose))

    if config.engine == 'legacy':
        return extrinsics
    elif config.engine == 'tensor':
        return list(
            map(lambda x: o3d.core.Tensor(x, o3d.core.Dtype.Float64),
                extrinsics))
    else:
        print('Unsupported engine {}'.format(config.engine))


def save_poses(
    path_trajectory,
    poses,
    intrinsic=o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)):
    if path_trajectory.endswith('log'):
        traj = o3d.camera.PinholeCameraTrajectory()
        params = []
        for pose in poses:
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsic
            param.extrinsic = np.linalg.inv(pose)
            params.append(param)
        traj.parameters = params
        o3d.io.write_pinhole_camera_trajectory(path_trajectory, traj)

    elif path_trajectory.endswith('json'):
        pose_graph = o3d.pipelines.registration.PoseGraph()
        for pose in poses:
            node = o3d.pipelines.registration.PoseGraphNode()
            node.pose = pose
            pose_graph.nodes.append(node)
        o3d.io.write_pose_graph(path_trajectory, pose_graph)


def init_volume(mode, config):
    if config.engine == 'legacy':
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=config.integration.voxel_size,
            sdf_trunc=config.integration.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    elif config.engine == 'tensor':
        if mode == 'scene':
            block_count = config.integration.scene_block_count
        else:
            block_count = config.integration.fragment_block_count
        return o3d.t.geometry.TSDFVoxelGrid(
            {
                'tsdf': o3d.core.Dtype.Float32,
                'weight': o3d.core.Dtype.UInt16,
                'color': o3d.core.Dtype.UInt16
            },
            voxel_size=config.integration.voxel_size,
            sdf_trunc=config.integration.sdf_trunc,
            block_resolution=16,
            block_count=block_count,
            device=o3d.core.Device(config.device))
    else:
        print('Unsupported engine {}'.format(config.engine))


def extract_pointcloud(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()

        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors

        if file_name is not None:
            o3d.io.write_point_cloud(file_name, pcd)

    elif config.engine == 'tensor':
        pcd = volume.extract_surface_points(
            weight_threshold=config.integration.surface_weight_threshold)

        if file_name is not None:
            o3d.t.io.write_point_cloud(file_name, pcd)

    return pcd


def extract_trianglemesh(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        if file_name is not None:
            o3d.io.write_triangle_mesh(file_name, mesh)

    elif config.engine == 'tensor':
        mesh = volume.extract_surface_mesh(
            weight_threshold=config.integration.surface_weight_threshold)
        mesh = mesh.to_legacy_triangle_mesh()

        if file_name is not None:
            o3d.io.write_triangle_mesh(file_name, mesh)

    return mesh
