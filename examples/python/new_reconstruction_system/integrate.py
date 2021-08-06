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


def integrate(depth_filenames,
              color_filenames,
              extrinsics,
              intrinsic,
              config,
              mode='scene'):

    n_files = len(color_file_names)
    n_extrinsics = len(extrinsics)
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
            depth_scale=config.input.depth_scale,
            depth_trunc=config.input.depth_max,
            convert_rgb_to_intensity=False)

        volume.integrate(rgbd, intrinsic, extrinsics[i])

    def tensor_integrate(i):
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        color = o3d.t.io.read_image(color_file_names[i]).to(device)

        volume.integrate(depth, color, intrinsic, extrinsics[i],
                         config.input.depth_scale, config.input.depth_max)

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


def voxelhashing(depth_filenames,
                 color_filenames,
                 intrinsic,
                 config,
                 mode='scene'):

    n_files = len(color_file_names)
    n_extrinsics = len(extrinsics)
    n = n_files
    if n_files != n_extrinsics:
        print(
            'Number of RGBD images ({}) and length of trajectory ({}) mismatch, using the smaller one.'
            .format(n_files, n_extrinsics))
        n = min(n_files, n_extrinsics)

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

    for i in range(n):
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

        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, config.input.depth_scale,
                        config.input.depth_max)
        model.synthesize_model_frame(raycast_frame, config.input.depth_scale,
                                     config.input.depth_min,
                                     config.input.depth_max, False)
        stop = time.time()
        print('{:04d}/{:04d} voxelhashing takes {:.4}s'.format(
            i, n, stop - start))

    pcd = model.extract_pointcloud()
    o3d.visualization.draw([pcd])


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


if __name__ == '__main__':
    #yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('path_dataset',
                        type=str,
                        help='path to the dataset.'
                        'It should contain 16bit depth images in a folder named depth/'
                        'and rgb images in a folder named color/ or rgb/')
    parser.add_argument('path_trajectory',
                        type=str,
                        help='path to the trajectory in .log|.json format')

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
    extrinsics = load_extrinsics(args.path_trajectory, config)
    intrinsic = load_intrinsic(args.path_intrinsic, config)

    volume = voxelhashing(depth_file_names, color_file_names, intrinsic, config)

    # volume = integrate(depth_file_names, color_file_names, extrinsics,
    #                    intrinsic, config)

    # mesh = extract_trianglemesh(volume, config, 'output.ply')
    # o3d.visualization.draw([mesh])
