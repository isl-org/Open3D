# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
from trajectory_io import *

if __name__ == "__main__":

	set_verbosity_level(VerbosityLevel.Debug)
	traj = read_trajectory("../../TestData/ICP/init.log")
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
			result_icp = registration_icp(source, target, 0.30,
					np.identity(4),
					TransformationEstimationPointToPlane())
			transformation_icp = result_icp.transformation
			information_icp = get_information_matrix_from_point_clouds(
					source, target, 0.30, result_icp.transformation)
			print(transformation_icp)

			print("Build PoseGraph")
			if target_id == source_id + 1: # odometry case
				odometry = np.dot(transformation_icp, odometry)
				pose_graph.nodes.append(
						PoseGraphNode(np.linalg.inv(odometry)))
				pose_graph.edges.append(
						PoseGraphEdge(source_id, target_id,
						transformation_icp, information_icp, False))
			else: # loop closure case
				pose_graph.edges.append(
						PoseGraphEdge(source_id, target_id,
						transformation_icp, information_icp, True))

	print("Optimizing PoseGraph ...")
	global_optimization(pose_graph,
			GlobalOptimizationLevenbergMarquardt(),
			GlobalOptimizationConvergenceCriteria(),
			GlobalOptimizationOption())

	print("Transform points and display")
	for point_id in range(n_pcds):
		print(pose_graph.nodes[point_id].pose)
		pcds[point_id].transform(pose_graph.nodes[point_id].pose)
	draw_geometries(pcds)
