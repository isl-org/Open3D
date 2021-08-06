# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import open3d as o3d
import argparse
import time

from config import Config, recursive_print
from common import load_image_file_names, save_poses, load_intrinsic, init_volume, extract_pointcloud, extract_trianglemesh


def voxelhashing(depth_file_names,
                 color_file_names,
                 intrinsic,
                 config,
                 mode='scene'):
    n_files = len(color_file_names)
    device = o3d.core.Device(config.device)

    T_frame_to_model = o3d.core.Tensor(np.identity(4))
    model = o3d.t.pipelines.voxelhashing.Model(
        config.integration.voxel_size, config.integration.sdf_trunc, 16,
        config.integration.scene_block_count, T_frame_to_model, device)
    depth_ref = o3d.t.io.read_image(depth_file_names[0])
    input_frame = o3d.t.pipelines.voxelhashing.Frame(depth_ref.rows,
                                                     depth_ref.columns,
                                                     intrinsic, device)
    raycast_frame = o3d.t.pipelines.voxelhashing.Frame(depth_ref.rows,
                                                       depth_ref.columns,
                                                       intrinsic, device)

    poses = []

    for i in range(n_files):
        start = time.time()

        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        input_frame.set_data_from_image('depth', depth)
        input_frame.set_data_from_image('color', color)

        if i > 0:
            result = model.track_frame_to_model(
                input_frame, raycast_frame, config.input.depth_scale,
                config.input.depth_max, config.odometry.corres_distance_trunc)
            T_frame_to_model = T_frame_to_model @ result.transformation

        poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, config.input.depth_scale,
                        config.input.depth_max)
        model.synthesize_model_frame(raycast_frame, config.input.depth_scale,
                                     config.input.depth_min,
                                     config.input.depth_max, False)
        stop = time.time()
        print('{:04d}/{:04d} voxelhashing takes {:.4}s'.format(
            i, n_files, stop - start))

    return model.voxel_grid, poses


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

    volume, poses = voxelhashing(depth_file_names[:100], color_file_names[:100], intrinsic,
                                 config)
    save_poses('output.log', poses)
    save_poses('output.json', poses)

    mesh = extract_trianglemesh(volume, config, 'output.ply')
    o3d.visualization.draw([mesh])
