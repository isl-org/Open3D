# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import os
import numpy as np
import open3d as o3d
import argparse
import time

from config import Config, recursive_print
from common import load_image_file_names, save_poses, load_intrinsic, init_volume, extract_pointcloud, extract_trianglemesh

from dense_slam import voxelhashing

if __name__ == '__main__':
    #yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset',
                        type=str,
                        help='path to the dataset.'
                        'It should contain 16bit depth images in a folder named depth/'
                        'and rgb images in a folder named color/ or rgb/')

    parser.add_argument('--config',
                        type=str,
                        help='path to the config json file.'
                        'If provided, all the following arguments will be overrided.')

    # Engine
    parser.add_argument('--engine',
                        type=str,
                        default='tensor',
                        choices=['tensor', 'legacy'],
                        help='Engine to choose from.')
    parser.add_argument('--device',
                        type=str,
                        default='CUDA:0',
                        help='Device to choose from. Only works for the tensor engine.')

    # RGBD
    parser.add_argument('--path_intrinsic',
                        type=str,
                        default='',
                        help='path to the intrinsic.json config file.'
                        'By default PrimeSense intrinsics is used.')
    parser.add_argument('--depth_folder', type=str,
                        default='depth',
                        help='subfolder name that contains depth files')
    parser.add_argument('--color_folder', type=str,
                        default='color',
                        help='subfolder name that contains color files')
    parser.add_argument('--depth_scale',
                        type=float,
                        default=1000.0,
                        help='depth factor. Converting from a uint16 depth image to meter.')
    parser.add_argument('--depth_max',
                        type=float,
                        default=3.0,
                        help='max range in the scene to integrate.')
    parser.add_argument('--fragment_size',
                        type=int,
                        default=100,
                        help='number of RGBD frames per fragment')

    # Volume
    parser.add_argument('--block_count',
                        type=int,
                        default=10000,
                        help='estimated number of 16x16x16 voxel blocks to represent a scene.'
                        'Typically with a 6mm resolution,'
                        'a lounge scene requires around 30K blocks,'
                        'while a large apartment requires 80K blocks.'
                        'Open3D will dynamically increase the block count on demand,'
                        'but a rough upper bound will be useful especially when memory is limited.')
    parser.add_argument('--voxel_size',
                        type=float,
                        default=3.0 / 512,
                        help='voxel resolution.'
                        'For small scenes, 6mm preserves fine details.'
                        'For large indoor scenes, 1cm or larger will be reasonable for limited memory.')
    parser.add_argument('--sdf_trunc',
                        type=float,
                        default=0.04,
                        help='SDF truncation threshold.')
    parser.add_argument('--surface_weight_threshold',
                        type=float,
                        default=3.0,
                        help='SDF weight truncation threshold during surface extraction.')
    args = parser.parse_args()
    #yapf: enable

    if args.config:
        config = Config(args.config)
        recursive_print(config)
        print('Config loaded from file {}'.format(args.config))
    else:
        config = Config()
        config.engine = args.engine
        config.device = args.device
        config.fragment_size = args.fragment_size
        config.input.depth_scale = args.depth_scale
        config.input.depth_max = args.depth_max
        config.input.depth_folder = args.depth_folder
        config.input.color_folder = args.color_folder
        config.integration.voxel_size = args.voxel_size
        config.integration.sdf_trunc = args.sdf_trunc
        config.integration.surface_weight_threshold = args.surface_weight_threshold
        recursive_print(config)
        print('Config loaded from args.')

    depth_file_names, color_file_names = load_image_file_names(
        args.path_dataset, config)

    intrinsic = load_intrinsic(args.path_intrinsic, config)

    path_fragments = os.path.join(args.path_dataset, 'fragments')
    os.makedirs(path_fragments, exist_ok=True)

    n_fragments = len(depth_file_names) // config.fragment_size
    for i in range(n_fragments):
        min_idx = i * config.fragment_size
        max_idx = (i + 1) * config.fragment_size
        if i == n_fragments - 1:
            max_idx = None
        volume, poses = voxelhashing(depth_file_names[min_idx:max_idx],
                                     color_file_names[min_idx:max_idx],
                                     intrinsic, config)
        ply_fragment = os.path.join(path_fragments, '{:03d}.ply'.format(i))
        log_fragment = os.path.join(path_fragments, '{:03d}.log'.format(i))

        pcd = extract_pointcloud(volume, config, ply_fragment)
        # o3d.visualization.draw([pcd])
        save_poses(log_fragment, poses)
