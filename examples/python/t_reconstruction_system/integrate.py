# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt

from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics


def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    if os.path.exists(config.path_npz):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.path_npz))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.path_npz))

        n_files = len(depth_file_names)
        device = o3d.core.Device(config.device)

        if config.integrate_color:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                ('tsdf', 'weight', 'color'),
                (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
                3.0 / 512, 8, 100000, o3d.core.Device('CUDA:0'))
        else:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                ('tsdf', 'weight'), (o3c.float32, o3c.float32), ((1), (1)),
                3.0 / 512, 8, 100000, o3d.core.Device('CUDA:0'))

        start = time.time()
        for i in range(n_files):
            print('Integrating frame {}/{}'.format(i, n_files))

            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            extrinsic = extrinsics[i]

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, config.depth_scale,
                config.depth_max)

            if config.integrate_color:
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                vbg.integrate(frustum_block_coords, depth, color, intrinsic,
                              extrinsic, config.depth_scale, config.depth_max)
            else:
                vbg.integrate(frustum_block_coords, depth, intrinsic, extrinsic,
                              config.depth_scale, config.depth_max)
            dt = time.time() - start
        print('Finished integrating {} frames in {} seconds'.format(
            n_files, dt))
        print('Saving to {}...'.format(config.path_npz))
        vbg.save(config.path_npz)
        print('Saving finished')

    return vbg


if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--integrate_color', action='store_true')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        print('Dataset not found, falling back to test examples.')
        example_path = os.path.abspath(
            os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))
        default_dataset_path = os.path.join(example_path, 'test_data', 'RGBD')
        config.path_dataset = default_dataset_path
        config.path_trajectory = os.path.join(default_dataset_path,
                                              'trajectory.log')

    if config.integrate_color:
        depth_file_names, color_file_names = load_rgbd_file_names(config)
    else:
        depth_file_names = load_depth_file_names(config)
        color_file_names = None

    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)
    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])
