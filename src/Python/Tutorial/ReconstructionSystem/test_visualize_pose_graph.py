# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import argparse
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from optimize_posegraph import *


def list_posegraph_files(folder_posegraph):
	pose_graph_paths = get_file_list(folder_posegraph, ".json")
	for pose_graph_path in pose_graph_paths:
		pose_graph = read_pose_graph(pose_graph_path)
		n_nodes = len(pose_graph.nodes)
		n_edges = len(pose_graph.edges)
		print("Fragment PoseGraph %s has %d nodes and %d edges" %
				(pose_graph_path, n_nodes, n_edges))


# test wide baseline matching
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="visualize pose graph")
	parser.add_argument("path_dataset", help="path to the dataset")
	parser.add_argument("-source_id", type=int, help="ID of source fragment")
	parser.add_argument("-target_id", type=int, help="ID of target fragment")
	args = parser.parse_args()

	ply_file_names = get_file_list(args.path_dataset + folder_fragment, ".ply")
	list_posegraph_files(args.path_dataset + folder_fragment)
	list_posegraph_files(args.path_dataset + folder_scene)

	global_pose_graph_name = args.path_dataset + \
			template_global_posegraph_optimized
	print(global_pose_graph_name)
	pose_graph = read_pose_graph(global_pose_graph_name)
	n_nodes = len(pose_graph.nodes)
	n_edges = len(pose_graph.edges)
	print("Global PoseGraph having %d nodes and %d edges" % (n_nodes, n_edges))

	# visualize all the edges
	for edge in pose_graph.edges:
		print("%d-%d" % (edge.source_node_id, edge.target_node_id))
		if edge.source_node_id == args.source_id and \
		 		edge.target_node_id == args.target_id:
			print("PoseGraphEdge %d-%d" % \
					(edge.source_node_id, edge.target_node_id))
			source = read_point_cloud(ply_file_names[edge.source_node_id])
			source_down = voxel_down_sample(source, 0.05)
			target = read_point_cloud(ply_file_names[edge.target_node_id])
			target_down = voxel_down_sample(target, 0.05)
			draw_registration_result(source, target, edge.transformation)

	# visualize all the trajectories
	pcds = []
	for i in range(len(pose_graph.nodes)):
		pcd = read_point_cloud(ply_file_names[i])
		pcd_down = voxel_down_sample(pcd, 0.05)
		pcd.transform(pose_graph.nodes[i].pose)
		print(np.linalg.inv(pose_graph.nodes[i].pose))
		pcds.append(pcd)
	draw_geometries(pcds)
