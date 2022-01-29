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

import argparse
import os
import sys
import json
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)
from open3d_example import *


def parse_keys(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
    if content is None:
        print('Unable to open {}'.format(filename))
        return []
    else:
        return sorted(list(map(int, ''.join(content).split())))


def collect_keyframe_rgbd(color_files, depth_files, keys):
    if len(keys) == 0:
        raise RuntimeError("No key frames selected")
    if keys[-1] >= len(color_files):
        raise ValueError("keys[-1]: {} index out of range".format(keys[-1]))

    selected_color_files = []
    selected_depth_files = []

    for key in keys:
        selected_color_files.append(color_files[key])
        selected_depth_files.append(depth_files[key])

    return selected_color_files, selected_depth_files


def collect_keyframe_pose(traj, intrinsic, keys):
    if len(keys) == 0:
        raise RuntimeError("No key frames selected")
    if keys[-1] >= len(traj.parameters):
        raise ValueError("keys[-1]: {} index out of range".format(keys[-1]))

    selectd_params = []
    for key in keys:
        param = traj.parameters[key]
        param.intrinsic = intrinsic
        selectd_params.append(param)
    traj.parameters = selectd_params
    return traj


def main(config, keys):
    path = config["path_dataset"]

    # Read RGBD images
    color_files, depth_files = get_rgbd_file_lists(path)
    if len(color_files) != len(depth_files):
        raise ValueError(
            "The number of color images {} must equal to the number of depth images {}."
            .format(len(color_files), len(depth_files)))

    camera = o3d.io.read_pinhole_camera_trajectory(
        os.path.join(path, config["template_global_traj"]))
    if len(color_files) != len(camera.parameters):
        raise ValueError(
            "The number of color images {} must equal to the number of camera parameters {}."
            .format(len(color_files), len(depth_files)))

    color_files, depth_files = collect_keyframe_rgbd(color_files, depth_files,
                                                     keys)

    # Read camera poses
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(
            config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    camera = collect_keyframe_pose(camera, intrinsic, keys)

    # Read mesh
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(path, config["template_global_mesh"]))

    # Load images
    rgbd_images = []
    for i in range(len(depth_files)):
        depth = o3d.io.read_image(os.path.join(depth_files[i]))
        color = o3d.io.read_image(os.path.join(color_files[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=config["depth_scale"],
            depth_trunc=config["max_depth"],
            convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    # Before full optimization, let's just visualize texture map
    # with given geometry, RGBD images, and camera poses.
    mesh, camera = o3d.pipelines.color_map.run_rigid_optimizer(
        mesh, rgbd_images, camera,
        o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0))
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(
        os.path.join(path, config["folder_scene"],
                     "color_map_before_optimization.ply"), mesh)

    # Optimize texture and save the mesh as texture_mapped.ply
    # This is implementation of following paper
    # Q.-Y. Zhou and V. Koltun,
    # Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
    # SIGGRAPH 2014
    mesh, camera = o3d.pipelines.color_map.run_non_rigid_optimizer(
        mesh, rgbd_images, camera,
        o3d.pipelines.color_map.NonRigidOptimizerOption(
            maximum_iteration=300, maximum_allowable_depth=config["max_depth"]))
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(
        os.path.join(path, config["folder_scene"],
                     "color_map_after_optimization.ply"), mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Color map optimizer for a reconstruction dataset')
    parser.add_argument('--config',
                        type=str,
                        required=True,
                        help='path to the config for the dataset '
                        'preprocessed by the Reconstruction System')
    parser.add_argument(
        '--keys',
        type=str,
        help='txt file that contains the indices of the keyframes')
    parser.add_argument('--sample_rate',
                        type=int,
                        default=10,
                        help='sampling rate that evenly sample key frames '
                        'if key.txt is not provided')
    args = parser.parse_args()

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
            initialize_config(config)
    assert config is not None

    keys = None
    if args.keys is not None:
        keys = parse_keys(args.keys)
    if keys is None:
        traj = o3d.io.read_pinhole_camera_trajectory(
            os.path.join(config["path_dataset"],
                         config["template_global_traj"]))
        keys = [i for i in range(0, len(traj.parameters), args.sample_rate)]

    main(config, keys)
