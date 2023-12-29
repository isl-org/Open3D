# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/integrate.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import time

import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm

from common import load_rgbd_file_names, load_depth_file_names, load_intrinsic, load_extrinsics, get_default_dataset
from config import ConfigParser


def integrate(depth_file_names, color_file_names, depth_intrinsic,
              color_intrinsic, extrinsics, config):
    n_files = len(depth_file_names)
    device = o3d.core.Device(config.device)

    if config.integrate_color:
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=3.0 / 512,
            block_resolution=16,
            block_count=50000,
            device=device)
    else:
        vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
                                            attr_dtypes=(o3c.float32,
                                                         o3c.float32),
                                            attr_channels=((1), (1)),
                                            voxel_size=3.0 / 512,
                                            block_resolution=16,
                                            block_count=50000,
                                            device=device)

    start = time.time()
    for i in tqdm(range(n_files)):
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        extrinsic = extrinsics[i]

        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, depth_intrinsic, extrinsic, config.depth_scale,
            config.depth_max)

        if config.integrate_color:
            color = o3d.t.io.read_image(color_file_names[i]).to(device)
            vbg.integrate(frustum_block_coords, depth, color, depth_intrinsic,
                          color_intrinsic, extrinsic, config.depth_scale,
                          config.depth_max)
        else:
            vbg.integrate(frustum_block_coords, depth, depth_intrinsic,
                          extrinsic, config.depth_scale, config.depth_max)
        dt = time.time() - start
    print('Finished integrating {} frames in {} seconds'.format(n_files, dt))

    return vbg


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
               'Default dataset may be selected from the following options: '
               '[lounge, jack_jack]',
               default='lounge')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        config = get_default_dataset(config)

    if config.integrate_color:
        depth_file_names, color_file_names = load_rgbd_file_names(config)
    else:
        depth_file_names = load_depth_file_names(config)
        color_file_names = None

    depth_intrinsic = load_intrinsic(config)
    color_intrinsic = load_intrinsic(config, 'color')

    extrinsics = load_extrinsics(config.path_trajectory, config)
    vbg = integrate(depth_file_names, color_file_names, depth_intrinsic,
                    color_intrinsic, extrinsics, config)

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])
