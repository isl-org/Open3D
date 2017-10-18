import numpy as np
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from visualization import *
from optimize_posegraph import *


# test wide baseline matching
if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset")
	source_id = parse_argument_int(sys.argv, "--source_id")
	target_id = parse_argument_int(sys.argv, "--target_id")

	if path_dataset:
		path_fragment = path_dataset + 'fragments/'
		ply_file_names = get_file_list(path_fragment, ".ply")

		pose_graph_name = path_fragment + \
				"global_registration_optimized.json"
		pose_graph = ReadPoseGraph(pose_graph_name)
		n_nodes = len(pose_graph.nodes)
		n_edges = len(pose_graph.edges)
		print("PoseGraph having %d nodes and %d edges" % (n_nodes, n_edges))

		for edge in pose_graph.edges:
			print("PoseGraphEdge %d-%d" % \
					(edge.source_node_id, edge.target_node_id))

			# if edge.source_node_id == source_id and \
			# 		edge.target_node_id == target_id:
			if True:
				source = ReadPointCloud(ply_file_names[edge.source_node_id])
				source_down = VoxelDownSample(source, 0.05)
				target = ReadPointCloud(ply_file_names[edge.target_node_id])
				target_down = VoxelDownSample(target, 0.05)
				DrawRegistrationResult(source, target, edge.transformation)
