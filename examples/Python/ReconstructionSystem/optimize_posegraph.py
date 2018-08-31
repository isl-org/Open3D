# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/ReconstructionSystem/optimize_posegraph.py

import sys
sys.path.append("../Utility")
from open3d import *
from common import *

def run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
        max_correspondence_distance, preference_loop_closure):
    # to display messages from global_optimization
    set_verbosity_level(VerbosityLevel.Debug)
    method = GlobalOptimizationLevenbergMarquardt()
    criteria = GlobalOptimizationConvergenceCriteria()
    option = GlobalOptimizationOption(
            max_correspondence_distance = max_correspondence_distance,
            edge_prune_threshold = 0.25,
            preference_loop_closure = preference_loop_closure,
            reference_node = 0)
    pose_graph = read_pose_graph(pose_graph_name)
    global_optimization(pose_graph, method, criteria, option)
    write_pose_graph(pose_graph_optmized_name, pose_graph)
    set_verbosity_level(VerbosityLevel.Error)


def optimize_posegraph_for_fragment(path_dataset, fragment_id, config):
    pose_graph_name = os.path.join(
            path_dataset, template_fragment_posegraph % fragment_id)
    pose_graph_optmized_name = os.path.join(path_dataset,
            template_fragment_posegraph_optimized % fragment_id)
    run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
            max_correspondence_distance = config["max_depth_diff"],
            preference_loop_closure = 0.1)


def optimize_posegraph_for_scene(path_dataset, config):
    pose_graph_name = os.path.join(path_dataset, template_global_posegraph)
    pose_graph_optmized_name = os.path.join(path_dataset,
            template_global_posegraph_optimized)
    run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
            max_correspondence_distance = config["voxel_size"] * 1.4,
            preference_loop_closure = 2.0)
