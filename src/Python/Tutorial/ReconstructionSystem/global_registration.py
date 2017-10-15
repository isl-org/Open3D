import numpy as np
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from opencv import *
from visualization import *
from optimize_posegraph import *


def preprocess_point_cloud(ply_file_name):
	print(ply_file_name)
	pcd = ReadPointCloud(ply_file_name)
	pcd_down = VoxelDownSample(pcd, 0.05)
	EstimateNormals(pcd_down,
			KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
	pcd_fpfh = ComputeFPFHFeature(pcd_down,
			KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
	return (pcd_down, pcd_fpfh)


def register_point_cloud_FPFH(source, target,
		source_fpfh, target_fpfh):
	result_ransac = RegistrationRANSACBasedOnFeatureMatching(
			source, target, source_fpfh, target_fpfh, 0.075,
			TransformationEstimationPointToPoint(False), 4,
			[CorrespondenceCheckerBasedOnEdgeLength(0.9),
			CorrespondenceCheckerBasedOnDistance(0.075),
			CorrespondenceCheckerBasedOnNormal(0.52359878)],
			RANSACConvergenceCriteria(4000000, 1000))
	return result_ransac


def register_point_cloud_ICP(source, target,
		init_transformation = np.identity(4)):
	result_icp = RegistrationICP(source, target, 0.02,
			init_transformation,
			TransformationEstimationPointToPlane())
	print(result_icp)
	information_matrix = GetInformationMatrixFromRegistrationResult(
			source_down, target_down, result_icp)
	return (result_icp.transformation, information_matrix)


# colored pointcloud registration
# This is implementation of following paper
# J. Park, Q.-Y. Zhou, V. Koltun,
# Colored Point Cloud Registration Revisited, ICCV 2017
def register_colored_point_cloud_ICP(source, target,
		init_transformation = np.identity(4), draw_result = False):
	voxel_radius = [ 0.05, 0.025, 0.0125 ]
	max_iter = [ 50, 30, 14 ]
	current_transformation = init_transformation
	for scale in range(3): # multi-scale approach
		iter = max_iter[scale]
		radius = voxel_radius[scale]
		source_down = VoxelDownSample(source, radius)
		target_down = VoxelDownSample(target, radius)
		EstimateNormals(source_down, KDTreeSearchParamHybrid(
				radius = scale, max_nn = 30))
		EstimateNormals(target_down, KDTreeSearchParamHybrid(
				radius = scale, max_nn = 30))
		result_icp = RegistrationColoredICP(source_down, target_down,
				radius, current_transformation,
				ICPConvergenceCriteria(relative_fitness = 1e-6,
				relative_rmse = 1e-6, max_iteration = iter))
		current_transformation = result_icp.transformation

	information_matrix = GetInformationMatrixFromRegistrationResult(
			source, target, result_icp)
	if draw_result:
		DrawRegistrationResultOriginalColor(source, target,
				result_icp.transformation)
	return (result_icp.transformation, information_matrix)


def register_point_cloud(ply_file_names,
		registration_type = "color", draw_reslt = False):
	pose_graph = PoseGraph()
	odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(odometry))
	info = np.identity(6)

	n_files = len(ply_file_names)
	for s in range(n_files):
		for t in range(s + 1, n_files):
			(source_down, source_fpfh) = preprocess_point_cloud(
					ply_file_names[s])
			(target_down, target_fpfh) = preprocess_point_cloud(
					ply_file_names[t])

			print("RegistrationRANSACBasedOnFeatureMatching")
			result_ransac = register_point_cloud_FPFH(source_down, target_down,
					source_fpfh, target_fpfh)
			if draw_reslt:
				DrawRegistrationResultOriginalColor(source_down, target_down,
						result_ransac.transformation)

			print("RegistrationPointCloud")
			if (registration_type == "color"):
				(transformation_matrix, information_matrix) = \
						register_colored_point_cloud_ICP(
						source_down, target_down, result_ransac.transformation)
			else:
				(transformation_matrix, information_matrix) = \
						register_point_cloud_ICP(
						source_down, target_down, result_ransac.transformation)
			if draw_reslt:
				DrawRegistrationResult(source_down, target_down,
						result_icp.transformation)

			if t == s + 1: # odometry case
				odometry = np.dot(transformation_matrix, odometry)
				odometry_inv = np.linalg.inv(odometry)
				pose_graph.nodes.append(PoseGraphNode(odometry_inv))
				pose_graph.edges.append(
						PoseGraphEdge(s, t, transformation_matrix,
						information_matrix, False))
			else: # edge case
				pose_graph.edges.append(
						PoseGraphEdge(s, t, transformation_matrix,
						information_matrix, True))
	return pose_graph


if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset") # todo use argparse
	if not path_dataset:
		print("usage : %s " % sys.argv[0])
		print("  --path_dataset [path]   : Path to rgbd_dataset. Mandatory.")
		sys.exit()

	ply_file_names = get_file_list(path_dataset + "/fragments/", ".ply")
	pose_graph = register_point_cloud(ply_file_names)
	pose_graph_name = path_dataset + "/fragments/global_registration.json"
	WritePoseGraph(pose_graph_name, pose_graph)
	pose_graph_optmized_name = path_dataset + "/fragments/" + \
			"global_registration_optimized.json"
	optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
