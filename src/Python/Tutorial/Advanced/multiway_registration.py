# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":

	set_verbosity_level(VerbosityLevel.Debug)
	pcds = []
	for i in range(3):
		pcd = read_point_cloud(
				"../../TestData/ICP/cloud_bin_%d.pcd" % i)
		downpcd = voxel_down_sample(pcd, voxel_size = 0.02)
		pcds.append(downpcd)
	draw_geometries(pcds)

	pose_graph = PoseGraph()
	odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(odometry))

	n_pcds = len(pcds)
	for source_id in range(n_pcds):
		for target_id in range(source_id + 1, n_pcds):
			source = pcds[source_id]
			target = pcds[target_id]

			print("Apply point-to-plane ICP")
			icp_coarse = registration_icp(source, target, 0.3,
					np.identity(4),
					TransformationEstimationPointToPlane())
			icp_fine = registration_icp(source, target, 0.03,
					icp_coarse.transformation,
					TransformationEstimationPointToPlane())
			transformation_icp = icp_fine.transformation
			information_icp = get_information_matrix_from_point_clouds(
					source, target, 0.03, icp_fine.transformation)
			print(transformation_icp)

			# draw_registration_result(source, target, np.identity(4))
			print("Build PoseGraph")
			if target_id == source_id + 1: # odometry case
				odometry = np.dot(transformation_icp, odometry)
				pose_graph.nodes.append(
						PoseGraphNode(np.linalg.inv(odometry)))
				pose_graph.edges.append(
						PoseGraphEdge(source_id, target_id,
						transformation_icp, information_icp, uncertain = False))
			else: # loop closure case
				pose_graph.edges.append(
						PoseGraphEdge(source_id, target_id,
						transformation_icp, information_icp, uncertain = True))

	print("Optimizing PoseGraph ...")
	option = GlobalOptimizationOption(
			max_correspondence_distance = 0.03,
			edge_prune_threshold = 0.25,
			reference_node = 0)
	global_optimization(pose_graph,
			GlobalOptimizationLevenbergMarquardt(),
			GlobalOptimizationConvergenceCriteria(), option)

	print("Transform points and display")
	for point_id in range(n_pcds):
		print(pose_graph.nodes[point_id].pose)
		pcds[point_id].transform(pose_graph.nodes[point_id].pose)
	draw_geometries(pcds)
