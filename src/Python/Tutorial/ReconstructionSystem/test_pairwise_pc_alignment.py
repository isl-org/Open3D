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
from register_fragments import *


def register_point_cloud_pairwise(path_dataset, ply_file_names,
		source_id, target_id, transformation_init = np.identity(4),
		registration_type = "color", draw_result = True):

	source = read_point_cloud(ply_file_names[source_id])
	target = read_point_cloud(ply_file_names[target_id])
	(source_down, source_fpfh) = preprocess_point_cloud(source)
	(target_down, target_fpfh) = preprocess_point_cloud(target)

	if abs(source_id - target_id) != 1:
		print("Do feature matching")
		(success_global, transformation_init) = \
					compute_initial_registration(
					source_id, target_id, source_down, target_down,
					source_fpfh, target_fpfh, path_dataset)
		if not success_global:
			print("No resonable solution for initial pose.")
		else:
			print(transformation_init)
	if draw_result:
		draw_registration_result(source_down, target_down,
				transformation_init)

	(transformation_icp, information_icp) = \
			local_refinement(source, target,
			source_down, target_down, transformation_init,
			registration_type, draw_result)
	if draw_result:
		draw_registration_result_original_color(source_down, target_down,
				transformation_icp)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="mathching two point clouds")
	parser.add_argument("path_dataset", help="path to the dataset")
	parser.add_argument("source_id", type=int, help="ID of source point cloud")
	parser.add_argument("target_id", type=int, help="ID of target point cloud")
	parser.add_argument("-path_json", help="reading json file for initial pose")
	args = parser.parse_args()

	ply_file_names = get_file_list(args.path_dataset + "/fragments/", ".ply")
	if not args.path_json:
		register_point_cloud_pairwise(args.path_dataset, ply_file_names,
				args.source_id, args.target_id)
	else:
		pose_graph = read_pose_graph(args.path_json)
		transformation_init = np.eye(4)
		for i in range(len(pose_graph.edges)):
			if pose_graph.edges[i].source_node_id == args.source_id and \
					pose_graph.edges[i].target_node_id == args.target_id:
				transformation_init = np.linalg.inv(pose_graph.edges[i].transformation)
		print("using following matrix for initial transformation")
		print(transformation_init)
		register_point_cloud_pairwise(args.path_dataset, ply_file_names,
				args.source_id, args.target_id, transformation_init)
