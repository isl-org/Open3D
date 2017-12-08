.. _pointcloud:

Reading and visualizing pointcloud
-------------------------------------

Let's get started Open3D with the following example.

.. code-block:: python

	import sys
	import numpy as np
	sys.path.append("../..")
	from py3d import *

	if __name__ == "__main__":

		print("Testing point cloud in py3d ...")
		print("Load a pcd point cloud, print it, and render it")
		pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
		print(pcd)
		print(np.asarray(pcd.points))
		draw_geometries([pcd])

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

		print("We load a polygon volume and use it to crop the original point cloud")
		vol = read_selection_polygon_volume("../../TestData/Crop/cropped.json")
		chair = vol.crop_point_cloud(pcd)
		draw_geometries([chair])
		print("")

.. image:: ../../_static/pointcloud.png
	:width: 400px

.. image:: ../../_static/pointcloud_small.png
	:width: 400px

.. image:: ../../_static/pointcloud_downsample.png
	:width: 400px

.. image:: ../../_static/pointcloud_downsample_normal.png
	:width: 400px

.. image:: ../../_static/pointcloud_crop.png
	:width: 400px
