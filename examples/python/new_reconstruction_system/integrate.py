# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import os
import configargparse
import numpy as np
import open3d as o3d
import argparse
import time

from config import add_arguments, get_config
from common import load_image_file_names, load_extrinsics, load_intrinsic, init_volume, extract_pointcloud, extract_trianglemesh


def integrate(depth_filenames,
              color_filenames,
              extrinsics,
              intrinsic,
              config,
              mode='scene'):

    n_files = len(color_file_names)
    n_extrinsics = len(extrinsics)

    if n_files == 0:
        print('No RGBD file found. Please check the data folder. Abort.')
        exit(-1)

    if n_extrinsics == 0:
        print('No extrinsics provided. Please check the trajectory provider. Abort.')
        exit(-1)

    n = n_files
    if n_files != n_extrinsics:
        print(
            'Number of RGBD images ({}) and length of trajectory ({}) mismatch, using the smaller one.'
            .format(n_files, n_extrinsics))
        n = min(n_files, n_extrinsics)

    volume = init_volume(mode, config)
    device = o3d.core.Device(config.device)

    def legacy_integrate(i):
        depth = o3d.io.read_image(depth_file_names[i])
        color = o3d.io.read_image(color_file_names[i])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=config.depth_scale,
            depth_trunc=config.depth_max,
            convert_rgb_to_intensity=False)

        volume.integrate(rgbd, intrinsic, extrinsics[i])

    def tensor_integrate(i):
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        volume.integrate(depth, color, intrinsic, extrinsics[i],
                         config.depth_scale, config.depth_max)

    if config.engine == 'legacy':
        fn_integrate = legacy_integrate
    elif config.engine == 'tensor':
        fn_integrate = tensor_integrate

    for i in range(n):
        start = time.time()
        fn_integrate(i)
        stop = time.time()
        print('{:04d}/{:04d} integration takes {:.4}s'.format(
            i, n, stop - start))

    return volume


if __name__ == '__main__':
    parser = configargparse.ArgParser(default_config_files=[
        os.path.join(os.path.dirname(__file__), 'default_config.yml')
    ],
                                      conflict_handler='resolve')
    parser.add('--config', is_config_file=True, help='Config file path.')
    parser.add('path_trajectory', help='Path to the trajectory.')
    parser = add_arguments(parser)

    args = parser.parse_args()
    config = get_config(args)

    depth_file_names, color_file_names = load_image_file_names(config)
    intrinsic = load_intrinsic(config)

    extrinsics = load_extrinsics(args.path_trajectory, config)

    volume = integrate(depth_file_names, color_file_names, extrinsics,
                       intrinsic, config)

    mesh = extract_trianglemesh(volume, config, 'output.ply')
    o3d.visualization.draw([mesh])
