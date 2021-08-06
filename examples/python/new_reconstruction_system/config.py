import os
import yaml
import configargparse
from easydict import EasyDict as edict


def add_arguments(parser):
    # yapf:disable
    # Default arguments
    parser.add(
        '--name', type=str,
        help='Name of the config for the offline reconsturction system.')
    parser.add(
        '--fragment_size', type=int,
        help='Number of RGBD frames to construct a fragment.')
    parser.add(
        '--device', type=str,
        help='Device to run the system.')
    parser.add(
        '--engine', type=str,
        choices=['tensor', 'legacy'],
        help='Open3D engine to reconstruct.')
    parser.add(
        '--multiprocessing', action='store_true',
        help='Use multiprocessing in operations. Only available for the legacy engine.')

    input_parser = parser.add_argument_group('input')
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

    odometry_parser = parser.add_argument_group('odometry')
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

    registration_parser = parser.add_argument_group('registration')
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

    integration_parser = parser.add_argument_group('integration')
    integration_parser.add(
        '--integration_mode',type=str,
        choices=['color', 'depth'],
        help='Volumetric integration mode.')
    integration_parser.add(
        '--voxel_size', type=float,
        help='Voxel size in meter for volumetric integration.')
    integration_parser.add(
        '--sdf_trunc', type=float,
        help='Truncation distance for signed distance.')
    integration_parser.add(
        '--block_count', type=int,
        help='Pre-allocated voxel block count for volumetric integration.')
    integration_parser.add(
        '--surface_weight_thr', type=float,
        help='Weight threshold to filter outliers during volumetric surface reconstruction.')
    #yapf:enable

    return parser


def get_config(args):
    d = edict(vars(args))

    # Resolve conflicts
    if d.engine == 'legacy':
        if d.device.lower().startswith('cuda'):
            print('Legacy engine only supports CPU.', 'Fallback to CPU.')
            d.device = 'CPU:0'

        if d.odometry_method == 'frame2model':
            print('Legacy engine does not supports frame2model tracking.',
                  'Fallback to hybrid odometry.')
            d.odometry_method = 'hybrid'

    if d.engine == 'tensor':
        if d.icp_method == 'generalized':
            print('Tensor engine does not support generalized ICP.',
                  'Fallback to colored ICP.')
            d.icp_method = 'colored'

        if d.multiprocessing:
            print('Tensor engine does not support multiprocessing.',
                  'Disabled.')
            d.multiprocessing = False

    return d


if __name__ == '__main__':
    # Priority: command line > loaded json > default config
    parser = configargparse.ArgParser(default_config_files=[
        os.path.join(os.path.dirname(__file__), 'default_config.yml')
    ])
    parser.add('--config', is_config_file=True, help='Config file path.')
    parser = add_arguments(parser)

    d = get_config(parser.parse_args())
    print(d)
