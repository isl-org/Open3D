# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import numpy as np
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	print("Testing kdtree in py3d ...")
	print("Load a point cloud and paint it black.")
	pcd = read_point_cloud("../../TestData/Feature/cloud_bin_0.pcd")
	pcd.paint_uniform_color([0, 0, 0])
	pcd_tree = KDTreeFlann(pcd)
	print("Paint the 1500th point red.")
	pcd.colors[1500] = [1, 0, 0]
	print("Find its 200 nearest neighbors, paint blue.")
	[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
	np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
	print("Find its neighbors with distance less than 0.2, paint green.")
	[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
	np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
	print("Visualize the point cloud.")
	draw_geometries([pcd])
	print("")

	print("Load two aligned point clouds.")
	pcd0 = read_point_cloud("../../TestData/Feature/cloud_bin_0.pcd")
	pcd1 = read_point_cloud("../../TestData/Feature/cloud_bin_1.pcd")
	pcd0.paint_uniform_color([1, 0.706, 0])
	pcd1.paint_uniform_color([0, 0.651, 0.929])
	draw_geometries([pcd0, pcd1])
	print("Load their FPFH feature and evaluate.")
	print("Black : matching distance > 0.2")
	print("White : matching distance = 0")
	feature0 = read_feature("../../TestData/Feature/cloud_bin_0.fpfh.bin")
	feature1 = read_feature("../../TestData/Feature/cloud_bin_1.fpfh.bin")
	fpfh_tree = KDTreeFlann(feature1)
	for i in range(len(pcd0.points)):
		[_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
		dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
		c = (0.2 - np.fmin(dis, 0.2)) / 0.2
		pcd0.colors[i] = [c, c, c]
	draw_geometries([pcd0])
	print("")

	print("Load their L32D feature and evaluate.")
	print("Black : matching distance > 0.2")
	print("White : matching distance = 0")
	feature0 = read_feature("../../TestData/Feature/cloud_bin_0.d32.bin")
	feature1 = read_feature("../../TestData/Feature/cloud_bin_1.d32.bin")
	fpfh_tree = KDTreeFlann(feature1)
	for i in range(len(pcd0.points)):
		[_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
		dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
		c = (0.2 - np.fmin(dis, 0.2)) / 0.2
		pcd0.colors[i] = [c, c, c]
	draw_geometries([pcd0])
	print("")
