# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

def DrawRegistrationResultUniformColor(source, target, transformation):
	source.PaintUniformColor([1, 0.706, 0])
	target.PaintUniformColor([0, 0.651, 0.929])
	source.Transform(transformation)
	DrawGeometries([source, target])

def DrawRegistrationResultOriginalColor(source, target, transformation):
	source.Transform(transformation)
	DrawGeometries([source, target])

if __name__ == "__main__":

	print("1. Load two point clouds.")
	source = ReadPointCloud("../../TestData/frag_115.ply")
	target = ReadPointCloud("../../TestData/frag_116.ply")


	voxel_radius = [ 0.04, 0.02, 0.01 ];
	reg_num_scale = [ 400, 3000, 20000 ];
	max_iter = [ 50, 30, 14 ];
	current = []
	current_transformation = np.identity(4);

	for scale in range(3):
		iter = max_iter[scale]
		radius = voxel_radius[scale]
		scale = reg_num_scale[scale]
		print([iter,radius,scale])

		print("2. Downsample with a voxel size %.2f" % radius)
		source_down = VoxelDownSample(source, scale)
		target_down = VoxelDownSample(target, scale)

		print("3. Estimate normal with search radius 0.1.")
		EstimateNormals(source_down, KDTreeSearchParamHybrid(
				radius = scale, max_nn = 30))
		EstimateNormals(target_down, KDTreeSearchParamHybrid(
				radius = scale, max_nn = 30))

		print("4. Colored point cloud registration is applied on original point")
		print("   clouds to refine the alignment. This time we use a strict")
		print("   distance threshold %.2f" % radius)
		result_icp = RegistrationColoredICP(source, target,
				radius, current_transformation,
				ICPConvergenceCriteria(relative_fitness = 1e-6,
				relative_rmse = 1e-6, max_iteration = iter))
		current_transformation = result_icp.transformation
		print(result_icp)
	DrawRegistrationResultOriginalColor(source, target, result_icp.transformation)
