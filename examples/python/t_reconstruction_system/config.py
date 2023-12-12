# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os

import configargparse
import open3d as o3d


class ConfigParser(configargparse.ArgParser):

    def __init__(self):
        super().__init__(default_config_files=[
            os.path.join(os.path.dirname(__file__), 'default_config.yml')
        ],
                         conflict_handler='resolve')

        # yapf:disable
        # Default arguments
        self.add(
            '--name', type=str,
            help='Name of the config for the offline reconstruction system.')
        self.add(
            '--fragment_size', type=int,
            help='Number of RGBD frames to construct a fragment.')
        self.add(
            '--device', type=str,
            help='Device to run the system.')
        self.add(
            '--engine', type=str,
            choices=['tensor', 'legacy'],
            help='Open3D engine to reconstruct.')
        self.add(
            '--multiprocessing', action='store_true',
            help='Use multiprocessing in operations. Only available for the legacy engine.')

        input_parser = self.add_argument_group('input')
        input_parser.add(
            '--path_dataset', type=str,
            help='Path to the dataset folder. It should contain a folder with depth and a folder with color images.')
        input_parser.add(
            '--depth_folder', type=str,
            help='Path that stores depth images.')
        input_parser.add(
            '--color_folder', type=str,
            help='Path that stores color images.')
        input_parser.add(
            '--path_intrinsic', type=str,
            help='Path to the intrinsic.json config file.'
            'If the intrinsic matrix for color image is different,'
            'specify it by --path_color_intrinsic.'
            'By default PrimeSense intrinsics is used.')
        input_parser.add(
            '--path_color_intrinsic', type=str,
            help='Path to the intrinsic.json config file.'
            'If the intrinsic matrix for color image is different,'
            'specify it by --path_color_intrinsic.'
            'By default PrimeSense intrinsics is used.')
        input_parser.add(
            '--depth_min', type=float,
            help='Min clipping distance (in meter) for input depth data.')
        input_parser.add(
            '--depth_max', type=float,
            help='Max clipping distance (in meter) for input depth data.')
        input_parser.add(
            '--depth_scale', type=float,
            help='Scale factor to convert raw input depth data to meters.')
        input_parser.add('--fragment_size', type=int, help='Number of RGBD frames per fragment')

        odometry_parser = self.add_argument_group('odometry')
        odometry_parser.add(
            '--odometry_method', type=str,
            choices=['point2plane', 'intensity', 'hybrid', 'frame2model'],
            help='Method used in pose estimation between RGBD images.'
            'Frame2model only available for the tensor engine.')
        odometry_parser.add(
            '--odometry_loop_interval', type=int,
            help='Intervals to check loop closures between RGBD images.')
        odometry_parser.add(
            '--odometry_loop_weight', type=float,
            help='Weight of loop closure edges when optimizing pose graphs for odometry.')
        odometry_parser.add(
            '--odometry_distance_thr', type=float,
            help='Default distance threshold to filter outliers in odometry correspondences.')

        registration_parser = self.add_argument_group('registration')
        registration_parser.add(
            '--icp_method', type=str,
            choices=['colored', 'point2point', 'point2plane', 'generalized'],
            help='Method used in registration between fragment point clouds with a good initial pose estimate.'
            'Generalized ICP only available for the tensor engine.')
        registration_parser.add(
            '--icp_voxelsize', type=float,
            help='Voxel size used to down sample point cloud for fast/multiscale ICP.')
        registration_parser.add(
            '--icp_distance_thr', type=float,
            help='Default distance threshold to filter outliers in ICP correspondences.')
        registration_parser.add(
            '--global_registration_method', type=str,
            choices=['fgr', 'ransac'],
            help='Method used in global registration of two fragment point clouds without an initial pose estimate.')
        registration_parser.add(
            '--registration_loop_weight', type=float,
            help='Weight of loop closure edges when optimizing pose graphs for registration.')

        integration_parser = self.add_argument_group('integration')
        integration_parser.add(
            '--integrate_color', action='store_true',
            default=False, help='Volumetric integration mode.')
        integration_parser.add(
            '--voxel_size', type=float,
            help='Voxel size in meter for volumetric integration.')
        integration_parser.add(
            '--trunc_voxel_multiplier', type=float,
            help='Truncation distance multiplier in voxel size for signed distance. For instance, --trunc_voxel_multiplier=8 with --voxel_size=0.006(m) creates a truncation distance of 0.048(m).')
        integration_parser.add(
            '--est_point_count', type=int,
            help='Estimated point cloud size for surface extraction.')
        integration_parser.add(
            '--block_count', type=int,
            help='Pre-allocated voxel block count for volumetric integration.')
        integration_parser.add(
            '--surface_weight_thr', type=float,
            help='Weight threshold to filter outliers during volumetric surface reconstruction.')
        # yapf:enable

    def get_config(self):
        config = self.parse_args()

        # Resolve conflicts
        if config.engine == 'legacy':
            if config.device.lower().startswith('cuda'):
                print('Legacy engine only supports CPU.', 'Fallback to CPU.')
                config.device = 'CPU:0'

            if config.odometry_method == 'frame2model':
                print('Legacy engine does not supports frame2model tracking.',
                      'Fallback to hybrid odometry.')
                config.odometry_method = 'hybrid'

        elif config.engine == 'tensor':
            if config.icp_method == 'generalized':
                print('Tensor engine does not support generalized ICP.',
                      'Fallback to colored ICP.')
                config.icp_method = 'colored'

            if config.multiprocessing:
                print('Tensor engine does not support multiprocessing.',
                      'Disabled.')
                config.multiprocessing = False

            if (config.device.lower().startswith('cuda') and
                (not o3d.core.cuda.is_available())):
                print(
                    'Open3d not built with cuda support or no cuda device available. ',
                    'Fallback to CPU.')
                config.device = 'CPU:0'

        return config


if __name__ == '__main__':
    # Priority: command line > custom config file > default config file
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    config = parser.get_config()
    print(config)
