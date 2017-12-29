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
from visualization import *
from optimize_posegraph import *


# test wide baseline matching
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='visualize pose graph')
	parser.add_argument('path_dataset', help='path to the dataset')
	parser.add_argument('-source_id', type=int, help='ID of source fragment')
	parser.add_argument('-target_id', type=int, help='ID of target fragment')
	args = parser.parse_args()

	path_fragment = args.path_dataset + 'fragments/'
	ply_file_names = get_file_list(path_fragment, ".ply")
	all_pose_graph_paths = get_file_list(path_fragment, ".json")
	for pose_graph_path in all_pose_graph_paths:
		pose_graph = read_pose_graph(pose_graph_path)
		n_nodes = len(pose_graph.nodes)
		n_edges = len(pose_graph.edges)
		print("PoseGraph %s has %d nodes and %d edges" %
				(pose_graph_path, n_nodes, n_edges))

	global_pose_graph_name = path_fragment + \
			"global_registration_optimized.json"
	pose_graph = read_pose_graph(global_pose_graph_name)
	n_nodes = len(pose_graph.nodes)
	n_edges = len(pose_graph.edges)
	print("Global PoseGraph having %d nodes and %d edges" % (n_nodes, n_edges))

	for edge in pose_graph.edges:
		print('%d-%d' % (edge.source_node_id, edge.target_node_id))
		if edge.source_node_id == args.source_id and \
				edge.target_node_id == args.target_id:
		# if True:
			print("PoseGraphEdge %d-%d" % \
					(edge.source_node_id, edge.target_node_id))
			source = read_point_cloud(ply_file_names[edge.source_node_id])
			source_down = voxel_down_sample(source, 0.05)
			target = read_point_cloud(ply_file_names[edge.target_node_id])
			target_down = voxel_down_sample(target, 0.05)
			draw_registration_result(source, target, edge.transformation)
