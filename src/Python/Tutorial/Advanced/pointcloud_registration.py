import sys
sys.path.append("../..")
from py3d import *
import numpy as np

def DrawRegistrationResult(source, target, transformation):
	source.PaintUniformColor([1, 0.706, 0])
	target.PaintUniformColor([0, 0.651, 0.929])
	source.Transform(transformation)
	DrawGeometries([source, target])

if __name__ == "__main__":

	print("1. Load two point clouds.")
	source = ReadPointCloud("../../TestData/ICP/cloud_bin_0.pcd")
	target = ReadPointCloud("../../TestData/ICP/cloud_bin_1.pcd")

	print("2. Downsample with a voxel size 0.05.")
	source_down = VoxelDownSample(source, 0.05)
	target_down = VoxelDownSample(target, 0.05)

	print("3. Estimate normal with search radius 0.1.")
	EstimateNormals(source_down, KDTreeSearchParamHybrid(
			radius = 0.1, max_nn = 30))
	EstimateNormals(target_down, KDTreeSearchParamHybrid(
			radius = 0.1, max_nn = 30))

	print("4. Compute FPFH feature with search radius 0.25")
	source_fpfh = ComputeFPFHFeature(source_down,
			KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
	target_fpfh = ComputeFPFHFeature(target_down,
			KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))

	print("5. RANSAC registration on downsampled point clouds.")
	print("   Since the downsampling voxel size is 0.05, we use a liberal")
	print("   distance threshold 0.075.")
	result_ransac = RegistrationRANSACBasedOnFeatureMatching(
			source_down, target_down, source_fpfh, target_fpfh, 0.075,
			TransformationEstimationPointToPoint(False), 4,
			[CorrespondenceCheckerBasedOnEdgeLength(0.9),
			CorrespondenceCheckerBasedOnDistance(0.075)],
			RANSACConvergenceCriteria(40000000, 500))
	print(result_ransac)
	DrawRegistrationResult(source_down, target_down,
			result_ransac.transformation)

	print("6. Point-to-plane ICP registration is applied on original point")
	print("   clouds to refine the alignment. This time we use a strict")
	print("   distance threshold 0.02.")
	result_icp = RegistrationICP(source, target, 0.02,
			result_ransac.transformation,
			TransformationEstimationPointToPlane())
	print(result_icp)
	DrawRegistrationResult(source, target, result_icp.transformation)
