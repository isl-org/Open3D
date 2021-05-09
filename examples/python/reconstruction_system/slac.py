# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/slac.py

import numpy as np
import open3d as o3d
import sys
sys.path.append("../utility")
from file import join, get_file_list, write_poses_to_log

sys.path.append(".")


def run(config):
    print("slac non-rigid optimisation.")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    path_dataset = config['path_dataset']

    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")

    pose_graph_fragment = o3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    slac_params = o3d.t.pipelines.slac.slac_optimizer_params(
        max_iterations=config["max_iterations"],
        voxel_size=config["voxel_size"],
        distance_threshold=config["distance_threshold"],
        fitness_threshold=config["fitness_threshold"],
        regularizer_weight=config["regularizer_weight"],
        device=o3d.core.Device(str(config["device"])),
        slac_folder=path_dataset + config["folder_slac"])

    debug_option = o3d.t.pipelines.slac.slac_debug_option(False, 0)

    pose_graph_updated = o3d.pipelines.registration.PoseGraph()
    pose_graph_updated, ctrl_grid = o3d.t.pipelines.slac.run_slac_optimizer_for_fragments(
        ply_file_names, pose_graph_fragment, slac_params, debug_option)

    hashmap = ctrl_grid.get_hashmap()
    active_addrs = hashmap.get_active_addrs().to(o3d.core.Dtype.Int64)
    key_tensor = hashmap.get_key_tensor()[active_addrs]
    key_tensor.save(slac_params.get_subfolder_name() + "/ctr_grid_keys.npy")
    value_tensor = hashmap.get_value_tensor()[active_addrs]
    value_tensor.save(slac_params.get_subfolder_name() + "/ctr_grid_values.npy")

    o3d.io.write_pose_graph(
        join(slac_params.get_subfolder_name(),
             config["template_optimized_posegraph_slac"]), pose_graph_updated)

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
        slac_params.get_subfolder_name() + "/optimized_trajectory_slac.log",
        trajectory)
