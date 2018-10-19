# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import json
import argparse
import os
import sys
from open3d import *
sys.path.append("../Utility")
from file import *
from visualization import *
sys.path.append(".")
from initialize_config import *


def list_posegraph_files(folder_posegraph):
    pose_graph_paths = get_file_list(folder_posegraph, ".json")
    for pose_graph_path in pose_graph_paths:
        pose_graph = read_pose_graph(pose_graph_path)
        n_nodes = len(pose_graph.nodes)
        n_edges = len(pose_graph.edges)
        print("Fragment PoseGraph %s has %d nodes and %d edges" %
                (pose_graph_path, n_nodes, n_edges))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize pose graph")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--source_id", type=int, help="ID of source fragment")
    parser.add_argument("--target_id", type=int, help="ID of target fragment")
    parser.add_argument("--adjacent", help="visualize adjacent pairs",
            action="store_true")
    parser.add_argument("--all", help="visualize all pairs",
            action="store_true")
    parser.add_argument("--list_posegraphs",
            help="list number of node and edges of all pose graphs",
            action="store_true")
    parser.add_argument("--before_optimized",
            help="visualize posegraph edges that is not optimized",
            action="store_true")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)

        ply_file_names = get_file_list(join(config["path_dataset"],
                config["folder_fragment"]), ".ply")
        if (args.list_posegraphs):
            list_posegraph_files(join(config["path_dataset"],
                    config["folder_fragment"]))
            list_posegraph_files(join(config["path_dataset"],
                    config["folder_scene"]))

        if (args.before_optimized):
            global_pose_graph_name = join(config["path_dataset"],
                    config["template_global_posegraph"])
        else:
            global_pose_graph_name = join(config["path_dataset"],
                    config["template_refined_posegraph_optimized"])
        print("Reading posegraph")
        print(global_pose_graph_name)
        pose_graph = read_pose_graph(global_pose_graph_name)
        n_nodes = len(pose_graph.nodes)
        n_edges = len(pose_graph.edges)
        print("Global PoseGraph having %d nodes and %d edges" % \
                (n_nodes, n_edges))

        # visualize alignment of posegraph edges
        for edge in pose_graph.edges:
            print("PoseGraphEdge %d-%d" % \
                    (edge.source_node_id, edge.target_node_id))
            if ((args.adjacent and \
                    edge.target_node_id - edge.source_node_id == 1)) or \
                    (not args.adjacent and
                    (args.source_id == edge.source_node_id and \
                    args.target_id == edge.target_node_id)) or \
                    args.all:
                print("    confidence : %.3f" % edge.confidence)
                source = read_point_cloud(ply_file_names[edge.source_node_id])
                target = read_point_cloud(ply_file_names[edge.target_node_id])
                source_down = voxel_down_sample(source, config["voxel_size"])
                target_down = voxel_down_sample(target, config["voxel_size"])
                print("original registration")
                draw_registration_result(
                        source_down, target_down, edge.transformation)
                print("optimized registration")
                source_down.transform(
                        pose_graph.nodes[edge.source_node_id].pose)
                target_down.transform(
                        pose_graph.nodes[edge.target_node_id].pose)
                draw_registration_result(
                        source_down, target_down, np.identity(4))
