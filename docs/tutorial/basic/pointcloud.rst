.. _pointcloud:

Point Cloud
-------------------------------------

This tutorial address basic usage regarding point cloud.
Consider the Python code below:

.. code-block:: python

    # src/Python/Tutorial/Basic/pointcloud.py

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

This script addresses basic operations for point cloud: voxel downsampling, point normal estimation, and cropping.
Let's take a look one by one.


.. _visualize_point_cloud:

Visualize point cloud
=====================================

.. code-block:: python

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("../../TestData/fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    draw_geometries([pcd])

Function ``read_point_cloud`` reads a point cloud from a file. This function detects the extension name of the file and tries its best to decode the file into a point cloud. Current supported extension names include: pcd, ply, xyz, xyzrgb, xyzn, pts.

``draw_geometries`` visualizes the point cloud.
Use mouse/trackpad to see the geometry from different view point.
Below window will appear twice:

.. image:: ../../_static/Basic/pointcloud/scene.png
    :width: 400px

It looks like dense surface, but it is point cloud.
Press :kbd:`-` key for several times. It becomes:

.. image:: ../../_static/Basic/pointcloud/scene_small.png
    :width: 400px

:kbd:`-` key is a helpful friend for decreasing the size of visualized points.


.. _voxel_downsampling:

Voxel downsampling
=====================================

One of the most basic geometric operation with point cloud is voxel downsampling.
It can reduce number of points by using a regular voxel grid. The pseudo algorithm is:

1. Points are assigned for corresponding voxel grid.
2. Voxel downsampling outputs a averaged point for each voxel.

Voxel downsampling is very important and useful tool for point cloud pre-processing.
Consider aligned point clouds. The points are dense for overlapping part and sparse for the non-overlapping part.
Voxel downsampling helps points to be evenly distributed as it produces only a single point from a single voxel.

Below script performs voxel downsampling for point cloud.

.. code-block:: python

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = voxel_down_sample(pcd, voxel_size = 0.05)
    draw_geometries([downpcd])

For ``voxel_down_sample``, it is necessary to specify the unit voxel size with ``voxel_size = 0.05``.
Our example point cloud has metric unit. 0.05 means 5cm.
As a result, ``downpcd`` has sparser point cloud than original point cloud.

This is a downsampled point cloud:

.. image:: ../../_static/Basic/pointcloud/downsampled.png
    :width: 400px


.. _vertex_normal_estimation:

Vertex normal estimation
=====================================

Another basic operation for point cloud is computing point normal. Take a look at this script:

.. code-block:: python

    print("Recompute the normal of the downsampled point cloud")
    estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(
            radius = 0.1, max_nn = 30))
    draw_geometries([downpcd])
    print("")

``estimate_normals`` computes normal for every points.
The function finds adjacent points and calculate the principal axis of points using covariance analysis.

The function takes an instance of ``KDTreeSearchParamHybrid`` class as an arguement.
The two key arguments ``radius = 0.1`` and ``max_nn = 30`` specifies search radius and maximum nearest neighbor.
It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.

The point cloud has normal direction now.
Once ``draw_geometries`` draws geometry, press :kbd:`n` key to see point normal.

.. image:: ../../_static/Basic/pointcloud/downsampled_normal.png
    :width: 400px

You can use :kbd:`-` or :kbd:`+` key to increase or decrease length of black needles representing normal direction.


.. _crop_point_cloud:

Crop point cloud
=====================================

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

.. image:: ../../_static/Basic/pointcloud/crop.png
    :width: 400px

.. _paint_point_cloud:

Paint point cloud
=====================================

The last script block paints the point cloud with yellow color.

.. code-block:: python

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    draw_geometries([chair])
    print("")

``paint_uniform_color`` paints all the points to be specified color.
The function accepts a list of red, green, and blue intensity in [0,1] range.

The chair becomes yellow:

.. image:: ../../_static/Basic/pointcloud/crop_color.png
    :width: 400px
