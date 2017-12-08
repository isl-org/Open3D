# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

def draw_registration_result_original_color(source, target, transformation):
	source.transform(transformation)
	draw_geometries([source, target])

if __name__ == "__main__":

	print("1. Load two point clouds and show initial pose")
	source = read_point_cloud("../../TestData/ColoredICP/frag_115.ply")
	target = read_point_cloud("../../TestData/ColoredICP/frag_116.ply")

	current_transformation = np.identity(4)
	draw_registration_result_original_color(source, target, current_transformation)

	# colored pointcloud registration
 	# This is implementation of following paper
 	# J. Park, Q.-Y. Zhou, V. Koltun,
 	# Colored Point Cloud Registration Revisited, ICCV 2017
	voxel_radius = [ 0.04, 0.02, 0.01 ];
	max_iter = [ 50, 30, 14 ];
	current_transformation = np.identity(4)
	for scale in range(3):
		iter = max_iter[scale]
		radius = voxel_radius[scale]
		print([iter,radius,scale])

		print("2. Downsample with a voxel size %.2f" % radius)
		source_down = voxel_down_sample(source, radius)
		target_down = voxel_down_sample(target, radius)

		print("3. Estimate normal.")
		estimate_normals(source_down, KDTreeSearchParamHybrid(
				radius = radius * 2, max_nn = 30))
		estimate_normals(target_down, KDTreeSearchParamHybrid(
				radius = radius * 2, max_nn = 30))

		print("4. Colored point cloud registration is applied on original point")
		print("   clouds to refine the alignment. This time we use a strict")
		print("   distance threshold %.2f" % radius)
		result_icp = registration_colored_icp(source_down, target_down,
				radius, current_transformation,
				ICPConvergenceCriteria(relative_fitness = 1e-6,
				relative_rmse = 1e-6, max_iteration = iter))
		current_transformation = result_icp.transformation
		print(result_icp)
	draw_registration_result_original_color(source, target, result_icp.transformation)

	# point to plane ICP
	current_transformation = np.identity(4);
	print("5. Point-to-plane ICP registration is applied on original point")
	print("   clouds to refine the alignment. Distance threshold 0.02.")
	result_icp = registration_icp(source, target, 0.02,
			current_transformation, TransformationEstimationPointToPlane())
	print(result_icp)
	draw_registration_result_original_color(source, target, result_icp.transformation)
