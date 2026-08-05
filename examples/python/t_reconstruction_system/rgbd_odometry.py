# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from tqdm import tqdm
import numpy as np
import open3d as o3d
import open3d.core as o3c
from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics, get_default_dataset


def read_legacy_rgbd_image(color_file, depth_file, convert_rgb_to_intensity):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=convert_rgb_to_intensity)
    return rgbd_image


def rgbd_loop_closure(depth_list, color_list, intrinsic, config):
    # TODO: load it from config
    device = o3c.Device('CUDA:0')

    interval = config.odometry_loop_interval
    n_files = len(depth_list)

    key_indices = list(range(0, n_files, interval))
    n_key_indices = len(key_indices)

    edges = []
    poses = []
    infos = []

    pairs = []

    criteria_list = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5)
    ]
    method = o3d.t.pipelines.odometry.Method.PointToPlane

    for i in range(n_key_indices - 1):
        key_i = key_indices[i]
        depth_curr = o3d.t.io.read_image(depth_list[key_i]).to(device)
        color_curr = o3d.t.io.read_image(color_list[key_i]).to(device)
        rgbd_curr = o3d.t.geometry.RGBDImage(color_curr, depth_curr)

        for j in range(i + 1, n_key_indices):
            key_j = key_indices[j]
            depth_next = o3d.t.io.read_image(depth_list[key_j]).to(device)
            color_next = o3d.t.io.read_image(color_list[key_j]).to(device)
            rgbd_next = o3d.t.geometry.RGBDImage(color_next, depth_next)

            # TODO: add OpenCV initialization if necessary
            # TODO: better failure check
            try:
                res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                    rgbd_curr, rgbd_next, intrinsic, o3c.Tensor(np.eye(4)),
                    1000.0, 3.0, criteria_list, method)
                info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
                    depth_curr, depth_next, intrinsic, res.transformation, 0.07,
                    1000.0, 3.0)
            except Exception as e:
                pass
            else:
                if info[5, 5] / (depth_curr.columns * depth_curr.rows) > 0.3:
                    edges.append((key_i, key_j))
                    poses.append(res.transformation.cpu().numpy())
                    infos.append(info.cpu().numpy())

                    # pcd_src = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    #     rgbd_curr, intrinsic)
                    # pcd_dst = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    #     rgbd_next, intrinsic)
                    # o3d.visualization.draw([pcd_src, pcd_dst])
                    # o3d.visualization.draw(
                    #     [pcd_src.transform(res.transformation), pcd_dst])

    return edges, poses, infos


def rgbd_odometry(depth_list, color_list, intrinsic, config):
    # TODO: load it from config
    device = o3c.Device('CUDA:0')

    n_files = len(depth_list)

    depth_curr = o3d.t.io.read_image(depth_list[0]).to(device)
    color_curr = o3d.t.io.read_image(color_list[0]).to(device)
    rgbd_curr = o3d.t.geometry.RGBDImage(color_curr, depth_curr)

    # TODO: load all params and scale/max factors from config
    edges = []
    poses = []
    infos = []

    criteria_list = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5)
    ]
    method = o3d.t.pipelines.odometry.Method.PointToPlane

    for i in tqdm(range(0, n_files - 1)):
        depth_next = o3d.t.io.read_image(depth_list[i + 1]).to(device)
        color_next = o3d.t.io.read_image(color_list[i + 1]).to(device)
        rgbd_next = o3d.t.geometry.RGBDImage(color_next, depth_next)

        res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            rgbd_curr, rgbd_next, intrinsic, o3c.Tensor(np.eye(4)), 1000.0, 3.0,
            criteria_list, method)
        info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
            depth_curr, depth_next, intrinsic, res.transformation, 0.07, 1000.0,
            3.0)

        edges.append((i, i + 1))
        poses.append(res.transformation.cpu().numpy())
        infos.append(info.cpu().numpy())

        color_curr = color_next
        depth_curr = depth_next
        rgbd_curr = rgbd_next

    return edges, poses, infos


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add('--config',
               is_config_file=True,
               help='YAML config file path.'
               'Please refer to config.py for the options,'
               'and default_config.yml for default settings '
               'It overrides the default config file, but will be '
               'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
               'Default dataset may be selected from the following options: '
               '[lounge, jack_jack]',
               default='lounge')
    config = parser.get_config()

    if config.path_dataset == '':
        config = get_default_dataset(config)

    depth_file_names, color_file_names = load_rgbd_file_names(config)

    intrinsic = load_intrinsic(config)

    i = 0
    j = 10

    depth_src = o3d.t.io.read_image(depth_file_names[i])
    color_src = o3d.t.io.read_image(color_file_names[i])

    depth_dst = o3d.t.io.read_image(depth_file_names[j])
    color_dst = o3d.t.io.read_image(color_file_names[j])

    rgbd_src = o3d.t.geometry.RGBDImage(color_src, depth_src)
    rgbd_dst = o3d.t.geometry.RGBDImage(color_dst, depth_dst)

    # RGBD odmetry and information matrix computation
    res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        rgbd_src, rgbd_dst, intrinsic)
    info = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
        depth_src, depth_dst, intrinsic, res.transformation, 0.07)
    print(res.transformation, info)
    print(info[5, 5] / (depth_src.columns * depth_src.rows))

    # Legacy for reference, can be a little bit different due to minor implementation discrepancies
    rgbd_src_legacy = read_legacy_rgbd_image(color_file_names[i],
                                             depth_file_names[i], True)
    rgbd_dst_legacy = read_legacy_rgbd_image(color_file_names[j],
                                             depth_file_names[j], True)
    intrinsic_legacy = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_src_legacy, rgbd_dst_legacy, intrinsic_legacy, np.eye(4),
        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm())
    print(trans, info)

    # Visualization
    pcd_src = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_src, intrinsic)
    pcd_dst = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_dst, intrinsic)
    o3d.visualization.draw([pcd_src, pcd_dst])
    o3d.visualization.draw([pcd_src.transform(res.transformation), pcd_dst])
