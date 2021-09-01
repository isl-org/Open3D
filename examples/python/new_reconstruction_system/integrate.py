# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import os
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt

from config import ConfigParser
from common import load_image_file_names, save_poses, load_intrinsic, load_extrinsics
import imageio


def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    if os.path.exists(config.npz_file):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.npz_file))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.npz_file)
        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.npz_file))

        n_files = len(color_file_names)
        device = o3d.core.Device(config.device)

        vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3d.core.Dtype.Float32, o3d.core.Dtype.Float32,
             o3d.core.Dtype.Float32), ((1), (1), (3)), 3.0 / 512, 8, 100000,
            o3d.core.Device('CUDA:0'))

        start = time.time()
        for i in range(n_files):
            print('Integrating frame {}/{}'.format(i, n_files))

            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            color = o3d.t.io.read_image(color_file_names[i]).to(device)
            extrinsic = extrinsics[i]

            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, config.depth_scale,
                config.depth_max)

            vbg.integrate(frustum_block_coords, depth, color, intrinsic,
                          extrinsic, config.depth_scale, config.depth_max)

            if i % 10 == 0 and i > 0:
                pcd = vbg.extract_point_cloud()
                o3d.visualization.draw([pcd])
            dt = time.time() - start
        print('Finished integrating {} frames in {} seconds'.format(
            n_files, dt))
        print('Saving to {}...'.format(config.npz_file))
        vbg.save(config.npz_file)
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
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--npz_file',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    depth_file_names, color_file_names = load_image_file_names(config)
    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])
