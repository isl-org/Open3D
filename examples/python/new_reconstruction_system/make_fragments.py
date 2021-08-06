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
from common import load_image_file_names, save_poses, load_intrinsic, init_volume, extract_pointcloud, extract_trianglemesh

from dense_slam import voxelhashing

if __name__ == '__main__':
    parser = configargparse.ArgParser(default_config_files=[
        os.path.join(os.path.dirname(__file__), 'default_config.yml')
    ],
                                      conflict_handler='resolve')
    parser.add('--config', is_config_file=True, help='Config file path.')
    parser = add_arguments(parser)

    args = parser.parse_args()
    config = get_config(args)

    depth_file_names, color_file_names = load_image_file_names(config)
    intrinsic = load_intrinsic(config)

    path_fragments = os.path.join(args.path_dataset, 'fragments')
    os.makedirs(path_fragments, exist_ok=True)

    n_fragments = len(depth_file_names) // config.fragment_size
    for i in range(n_fragments):
        min_idx = i * config.fragment_size
        max_idx = (i + 1) * config.fragment_size
        if i == n_fragments - 1:
            max_idx = None

        print('Processing fragment {:03d}'.format(i))
        volume, poses = voxelhashing(depth_file_names[min_idx:max_idx],
                                     color_file_names[min_idx:max_idx],
                                     intrinsic, config)
        ply_fragment = os.path.join(path_fragments, '{:03d}.ply'.format(i))
        log_fragment = os.path.join(path_fragments, '{:03d}.log'.format(i))

        pcd = extract_pointcloud(volume, config, ply_fragment)

        T = np.eye(4)
        T[1, 1] = -1
        T[2, 2] = -1
        o3d.visualization.draw([pcd.to_legacy().transform(T)])
        save_poses(log_fragment, poses)
