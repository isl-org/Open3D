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


def preprocess_point_cloud(pcd):
	pcd_down = voxel_down_sample(pcd, 0.05)
	estimate_normals(pcd_down,
			KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
	pcd_fpfh = compute_fpfh_feature(pcd_down,
			KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
	return (pcd_down, pcd_fpfh)


def register_point_cloud_fpfh(source, target,
		source_fpfh, target_fpfh):
	result_ransac = registration_ransac_based_on_feature_matching(
			source, target, source_fpfh, target_fpfh, 0.075,
			TransformationEstimationPointToPoint(False), 4,
			[CorrespondenceCheckerBasedOnEdgeLength(0.9),
			CorrespondenceCheckerBasedOnDistance(0.075),
			CorrespondenceCheckerBasedOnNormal(0.52359878)],
			RANSACConvergenceCriteria(4000000, 2000))
	if (result_ransac.transformation.trace() == 4.0):
		return (False, np.identity(4))
	else:
		return (True, result_ransac)


def compute_initial_registration(s, t, source_down, target_down,
		source_fpfh, target_fpfh, path_dataset, draw_result = False):

	if t == s + 1: # odometry case
		print("Using RGBD odometry")
		pose_graph_frag = read_pose_graph(path_dataset +
				template_fragment_posegraph_optimized % s)
		n_nodes = len(pose_graph_frag.nodes)
		transformation = np.linalg.inv(
				pose_graph_frag.nodes[n_nodes-1].pose)
		print(pose_graph_frag.nodes[0].pose)
		print(transformation)
	else: # loop closure case
		print("register_point_cloud_fpfh")
		(success_ransac, result_ransac) = register_point_cloud_fpfh(
				source_down, target_down,
				source_fpfh, target_fpfh)
		if not success_ransac:
			print("No resonable solution. Skip this pair")
			return (False, np.identity(4))
		else:
			transformation = result_ransac.transformation
		print(transformation)

	if draw_result:
		draw_registration_result(source_down, target_down,
				transformation)
	return (True, transformation)


def register_point_cloud_icp(source, target,
		init_transformation = np.identity(4)):
	result_icp = registration_icp(source, target, 0.02,
			init_transformation,
			TransformationEstimationPointToPlane())
	print(result_icp)
	information_matrix = get_information_matrix_from_point_clouds(
			source, target, 0.03, result_icp.transformation)
	return (result_icp.transformation, information_matrix)


# colored pointcloud registration
# This is implementation of following paper
# J. Park, Q.-Y. Zhou, V. Koltun,
# Colored Point Cloud Registration Revisited, ICCV 2017
def register_colored_point_cloud_icp(source, target,
		init_transformation = np.identity(4), draw_result = False):
	voxel_radius = [ 0.05, 0.025, 0.0125 ]
	max_iter = [ 50, 30, 14 ]
	current_transformation = init_transformation
	for scale in range(3): # multi-scale approach
		iter = max_iter[scale]
		radius = voxel_radius[scale]
		print("radius %f" % radius)
		source_down = voxel_down_sample(source, radius)
		target_down = voxel_down_sample(target, radius)
		estimate_normals(source_down, KDTreeSearchParamHybrid(
				radius = radius * 2, max_nn = 30))
		print(np.asarray(source_down.normals))
		estimate_normals(target_down, KDTreeSearchParamHybrid(
				radius = radius * 2, max_nn = 30))
		result_icp = registration_colored_icp(source_down, target_down,
				radius, current_transformation,
				ICPConvergenceCriteria(relative_fitness = 1e-6,
				relative_rmse = 1e-6, max_iteration = iter))
		current_transformation = result_icp.transformation

	information_matrix = get_information_matrix_from_point_clouds(
			source, target, 0.03, result_icp.transformation)
	if draw_result:
		draw_registration_result_original_color(source, target,
				result_icp.transformation)
	return (result_icp.transformation, information_matrix)


def local_refinement(source, target, source_down, target_down,
		transformation_init, registration_type = "color",
		draw_result = False):

	if (registration_type == "color"):
		print("register_colored_point_cloud")
		(transformation, information) = \
				register_colored_point_cloud_icp(
				source, target, transformation_init)
	else:
		print("register_point_cloud_icp")
		(transformation, information) = \
				register_point_cloud_icp(
				source_down, target_down, transformation_init)

	if draw_result:
		draw_registration_result_original_color(
				source_down, target_down, transformation)
	return (transformation, information)


def update_odometry_posegrph(s, t, transformation, information,
		odometry, pose_graph):

	print("Update PoseGraph")
	if t == s + 1: # odometry case
		odometry = np.dot(transformation, odometry)
		odometry_inv = np.linalg.inv(odometry)
		pose_graph.nodes.append(PoseGraphNode(odometry_inv))
		pose_graph.edges.append(
				PoseGraphEdge(s, t, transformation,
				information, uncertain = False))
	else: # loop closure case
		pose_graph.edges.append(
				PoseGraphEdge(s, t, transformation,
				information, uncertain = True))
	return (odometry, pose_graph)


def register_point_cloud(path_dataset, ply_file_names,
		registration_type = "color", draw_result = False):
	pose_graph = PoseGraph()
	odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(odometry))
	info = np.identity(6)

	n_files = len(ply_file_names)
	for s in range(n_files):
		for t in range(s + 1, n_files):
			print("reading %s ..." % ply_file_names[s])
			source = read_point_cloud(ply_file_names[s])
			print("reading %s ..." % ply_file_names[t])
			target = read_point_cloud(ply_file_names[t])
			(source_down, source_fpfh) = preprocess_point_cloud(source)
			(target_down, target_fpfh) = preprocess_point_cloud(target)

			(success_global, transformation_init) = \
					compute_initial_registration(
					s, t, source_down, target_down,
					source_fpfh, target_fpfh, path_dataset)
			if not success_global:
				continue

			(transformation_icp, information_icp) = \
					local_refinement(source, target,
					source_down, target_down, transformation_init,
					registration_type, draw_result)

			(odometry, pose_graph) = update_odometry_posegrph(s, t,
					transformation_icp, information_icp,
					odometry, pose_graph)
			print(pose_graph)

	write_pose_graph(path_dataset + template_global_posegraph, pose_graph)


if __name__ == "__main__":
	set_verbosity_level(VerbosityLevel.Debug)
	parser = argparse.ArgumentParser(description="register fragments.")
	parser.add_argument("path_dataset", help="path to the dataset")
	args = parser.parse_args()

	ply_file_names = get_file_list(args.path_dataset + folder_fragment, ".ply")
	make_folder(args.path_dataset + folder_scene)
	register_point_cloud(args.path_dataset, ply_file_names)
	optimize_posegraph_for_scene(args.path_dataset)
