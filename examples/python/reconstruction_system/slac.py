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

# examples/python/reconstruction_system/slac.py

import numpy as np
import open3d as o3d
import os, sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import join, get_file_list, write_poses_to_log

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def run(config):
    print("slac non-rigid optimization.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    path_dataset = config['path_dataset']

    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")

    if (len(ply_file_names) == 0):
        raise RuntimeError(
            "No fragment found in {}, please make sure the reconstruction_system has finished running on the dataset."
            .format(join(config["path_dataset"], config["folder_fragment"])))

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    # SLAC optimizer parameters.
    slac_params = o3d.t.pipelines.slac.slac_optimizer_params(
        max_iterations=config["max_iterations"],
        voxel_size=config["voxel_size"],
        distance_threshold=config["distance_threshold"],
        fitness_threshold=config["fitness_threshold"],
        regularizer_weight=config["regularizer_weight"],
        device=o3d.core.Device(str(config["device"])),
        slac_folder=join(path_dataset, config["folder_slac"]))

    # SLAC debug option.
    debug_option = o3d.t.pipelines.slac.slac_debug_option(False, 0)

    # Run the system.
    pose_graph_updated = o3d.pipelines.registration.PoseGraph()

    # rigid optimization method.
    if (config["method"] == "rigid"):
        pose_graph_updated = o3d.t.pipelines.slac.run_rigid_optimizer_for_fragments(
            ply_file_names, pose_graph_fragment, slac_params, debug_option)
    elif (config["method"] == "slac"):
        pose_graph_updated, ctrl_grid = o3d.t.pipelines.slac.run_slac_optimizer_for_fragments(
            ply_file_names, pose_graph_fragment, slac_params, debug_option)

        hashmap = ctrl_grid.get_hashmap()
        active_buf_indices = hashmap.active_buf_indices().to(
            o3d.core.Dtype.Int64)

        key_tensor = hashmap.key_tensor()[active_buf_indices]
        key_tensor.save(
            join(slac_params.get_subfolder_name(), "ctr_grid_keys.npy"))

        value_tensor = hashmap.value_tensor()[active_buf_indices]
        value_tensor.save(
            join(slac_params.get_subfolder_name(), "ctr_grid_values.npy"))

    else:
        raise RuntimeError(
            "Requested optimization method {}, is not implemented. Implemented methods includes slac and rigid."
            .format(config["method"]))

    # Write updated pose graph.
    o3d.io.write_pose_graph(
        join(slac_params.get_subfolder_name(),
             config["template_optimized_posegraph_slac"]), pose_graph_updated)

    # Write trajectory for slac-integrate stage.
    fragment_folder = join(path_dataset, config["folder_fragment"])
    params = []
    for i in range(len(pose_graph_updated.nodes)):
        fragment_pose_graph = o3d.io.read_pose_graph(
            join(fragment_folder, "fragment_optimized_%03d.json" % i))
        for node in fragment_pose_graph.nodes:
            pose = np.dot(pose_graph_updated.nodes[i].pose, node.pose)
            param = o3d.camera.PinholeCameraParameters()
            param.extrinsic = np.linalg.inv(pose)
            params.append(param)

    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = params

    o3d.io.write_pinhole_camera_trajectory(
        slac_params.get_subfolder_name() + "/optimized_trajectory_" +
        str(config["method"]) + ".log", trajectory)
