import numpy as np
import sys
sys.path.append("../..")
from py3d import *
from utility import *


def preprocess_point_cloud(ply_file_name):
	pcd = ReadPointCloud(ply_file_name)
	pcd_down = VoxelDownSample(pcd, 0.05)
	EstimateNormals(pcd_down,
			KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
	pcd_fpfh = ComputeFPFHFeature(pcd_down,
			KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
	return (pcd_down, pcd_fpfh)


def register_point_cloud(ply_file_names):
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
			print(ply_file_names[s])
			print(ply_file_names[t])

			result_ransac = RegistrationRANSACBasedOnFeatureMatching(
					source_down, target_down, source_fpfh, target_fpfh, 0.075,
					TransformationEstimationPointToPoint(False), 4,
					[CorrespondenceCheckerBasedOnEdgeLength(0.9),
					CorrespondenceCheckerBasedOnDistance(0.075)],
					RANSACConvergenceCriteria(40000, 500))
			print(result_ransac)
			# DrawRegistrationResult(source_down, target_down,
			# 		result_ransac.transformation)
			# can it output information file too?
			result_icp = RegistrationICP(source_down, target_down, 0.02,
					result_ransac.transformation,
					TransformationEstimationPointToPlane())
			# print(result_icp)
			# DrawRegistrationResult(source, target, result_icp.transformation)
			if t == s + 1:
				odometry = np.dot(result_icp.transformation, odometry)
				odometry_inv = np.linalg.inv(odometry)
				pose_graph.nodes.append(PoseGraphNode(odometry_inv))
				pose_graph.edges.append(
						PoseGraphEdge(s, t, result_icp.transformation, info, False))
			else:
				pose_graph.edges.append(
						PoseGraphEdge(s, t, result_icp.transformation, info, True))
	return pose_graph


if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset")
	if path_dataset:
		ply_file_names = get_file_list(path_dataset, '.ply')
		pose_graph = register_point_cloud(ply_file_names)
		pose_graph_name = path_dataset + "global_registration.json"
		WritePoseGraph(pose_graph_name, pose_graph)
		pose_graph_optmized_name = path_fragment + \
				"global_registration_optimized.json"
		optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
