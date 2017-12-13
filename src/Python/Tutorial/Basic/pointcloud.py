# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import numpy as np
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	print("Load a ply point cloud, print it, and render it")
	pcd = read_point_cloud("../../TestData/fragment.ply")
	print(pcd)
	print(np.asarray(pcd.points))
	draw_geometries([pcd])

	print("Downsample the point cloud with a voxel of 0.05")
	downpcd = voxel_down_sample(pcd, voxel_size = 0.05)
	draw_geometries([downpcd])

	print("Recompute the normal of the downsampled point cloud")
	estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(
			radius = 0.1, max_nn = 30))
	draw_geometries([downpcd])
	print("")

	print("Load a polygon volume and use it to crop the original point cloud")
	vol = read_selection_polygon_volume("../../TestData/Crop/cropped.json")
	chair = vol.crop_point_cloud(pcd)
	draw_geometries([chair])
	print("")

	print("Paint chair")
	chair.paint_uniform_color([1, 0.706, 0])
	draw_geometries([chair])
	print("")
