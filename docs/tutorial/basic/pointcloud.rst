.. _pointcloud:

Visualize pointcloud
-------------------------------------

This tutorial address basic functions you can try with pointcloud.
Consider the Python code below:

.. code-block:: python

	# src/Python/Tutorial/Basic/pointcloud.py

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

This script addresses basic operations for point cloud: voxel downsampling, point normal estimation, and cropping.
Let's take a look one by one.

.. code-block:: python

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

This script will read pcd file and ply file. It visualizes the pointcloud.
You will see below window twice:

.. image:: ../../_static/pointcloud.png
	:width: 400px

It looks like dense surface, but it is not. It is point cloud.
It is because each point is large enough to make no empty space.

Press :kbd:`-` key for several times. You will see:

.. image:: ../../_static/pointcloud_small.png
	:width: 400px

:kbd:`-` key is a helpful friend for analyzing your point cloud.

One of the most basic geometric operation with point cloud is voxel downsampling.
It reduces number of points by using regular voxel grid.
For example, if a voxel has multiple points, voxel downsampling outputs averaged points.

Voxel downsampling is very important and useful tool for point cloud preprocessing.
Below script performs voxel downsampling.

.. code-block:: python

	print("Downsample the point cloud with a voxel of 0.05")
	downpcd = voxel_down_sample(pcd, voxel_size = 0.05)
	draw_geometries([downpcd])

For ``voxel_down_sample``, you need to specify the unit voxel size using ``voxel_size = 0.05``.
Our example point cloud has metric unit: 1 means 1 meter, and 0.05 means 5cm.
As a result, ``downpcd`` has sparser point cloud. Each point is distant away approximately 5cm.

This is a downsampled point cloud you will see:

.. image:: ../../_static/pointcloud_downsample.png
	:width: 400px

Another operation is computing point normal. Take a look at this script:

.. code-block:: python

	print("Recompute the normal of the downsampled point cloud")
	estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(
			radius = 0.1, max_nn = 30))
	draw_geometries([downpcd])
	print("")

It computes normal for every points. ``estimate_normals`` takes ``KDTreeSearchParamHybrid`` class.
Detailed explaination about KDtree can be found [here].

Normal estimation uses covariance analysis. Each point uses a set of adjacent points.
The two key arguments ``radius = 0.1`` and ``max_nn = 30`` specifies search radius and maximum nearest neighbor.
It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.

The point cloud has normal direction now. Press :kbd:`n` key to see point normal.

.. image:: ../../_static/pointcloud_downsample_normal.png
	:width: 400px

You can use :kbd:`-` or :kbd:`+` key to increase or decrease length of black needles representing normal direction.

Another example is point cloud cropping. See this script:

.. code-block:: python

	print("We load a polygon volume and use it to crop the original point cloud")
	vol = read_selection_polygon_volume("../../TestData/Crop/cropped.json")
	chair = vol.crop_point_cloud(pcd)
	draw_geometries([chair])
	print("")

``read_selection_polygon_volume`` reads a json file that specifies polygon selection area.
``vol.crop_point_cloud(pcd)`` filters out points.

This will remain only the chair in the scene.

.. image:: ../../_static/pointcloud_crop.png
	:width: 400px
