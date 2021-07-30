# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import sys, os
import time
import numpy as np
import open3d as o3d
import argparse
from config import Config, recursive_print
import glob


def odometry(src_depth,
             src_color,
             dst_depth,
             dst_color,
             config,
             init=np.identity(4)):
    pass

def frame_to_model_tracking(depths, colors, config, init=np.identity(4)):
    


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
    parser.add_argument('--depth_scale',
                        type=float,
                        default=1000.0,
                        help='depth factor. Converting from a uint16 depth image to meter.')
    parser.add_argument('--depth_max',
                        type=float,
                        default=3.0,
                        help='max range in the scene to integrate.')
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
        config.input.depth_scale = args.depth_scale
        config.input.depth_max = args.depth_max
        config.integration.voxel_size = args.voxel_size
        config.integration.sdf_trunc = args.sdf_trunc
        config.integration.surface_weight_threshold = args.surface_weight_threshold
        recursive_print(config)
        print('Config loaded from args.')

    depth_file_names, color_file_names = load_image_file_names(
        args.path_dataset, config)
    extrinsics = load_extrinsics(args.path_trajectory, config)
    intrinsic = load_intrinsic(args.path_intrinsic, config)
    volume = integrate(depth_file_names, color_file_names, extrinsics,
                       intrinsic, config)

    mesh = extract_trianglemesh(volume, config, 'output.ply')
    o3d.visualization.draw([mesh])
