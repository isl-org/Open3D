.. _pointcloud:

Point Cloud
-------------------------------------

This tutorial demonstrates basic usage of a point cloud.

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

.. _visualize_point_cloud:

Visualize point cloud
=====================================

The first part of the tutorial reads a point cloud and visualizes it.

.. code-block:: python

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("../../TestData/fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    draw_geometries([pcd])

``read_point_cloud`` reads a point cloud from a file. It tries to decode the file based on the extension name. The supported extension names are: ``pcd``, ``ply``, ``xyz``, ``xyzrgb``, ``xyzn``, ``pts``.

``draw_geometries`` visualizes the point cloud.
Use mouse/trackpad to see the geometry from different view point.

.. image:: ../../_static/Basic/pointcloud/scene.png
    :width: 400px

It looks like a dense surface, but it is actually a point cloud rendered as surfels. The GUI supports various keyboard functions. One of them, the :kbd:`-` key reduces the size of the points (surfels). Press it multiple times, the visualization becomes:

.. image:: ../../_static/Basic/pointcloud/scene_small.png
    :width: 400px

.. note:: Press :kbd:`h` key to print out a complete list of keyboard instructions for the GUI. For more information of the visualization GUI, refer to :ref:`visualization` and :ref:`customized_visualization`.

.. note:: On OS X, the GUI window may not receive keyboard event. In this case, try to launch Python with ``pythonw`` instead of ``python``.

.. _voxel_downsampling:

Voxel downsampling
=====================================

Voxel downsampling uses a regular voxel grid to create a uniformly downsampled point cloud from an input point cloud. It is often used as a pre-processing step for many point cloud processing tasks. The algorithm operates in two steps:

1. Points are bucketed into voxels.
2. Each occupied voxel generates exact one point by averaging all points inside.

.. code-block:: python

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = voxel_down_sample(pcd, voxel_size = 0.05)
    draw_geometries([downpcd])

This is the downsampled point cloud:

.. image:: ../../_static/Basic/pointcloud/downsampled.png
    :width: 400px

.. _vertex_normal_estimation:

Vertex normal estimation
=====================================

Another basic operation for point cloud is point normal estimation.

.. code-block:: python

    print("Recompute the normal of the downsampled point cloud")
    estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(
            radius = 0.1, max_nn = 30))
    draw_geometries([downpcd])
    print("")

``estimate_normals`` computes normal for every point. The function finds adjacent points and calculate the principal axis of the adjacent points using covariance analysis.

The function takes an instance of ``KDTreeSearchParamHybrid`` class as an argument. The two key arguments ``radius = 0.1`` and ``max_nn = 30`` specifies search radius and maximum nearest neighbor. It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.

.. note:: The covariance analysis algorithm produces two opposite directions as normal candidates. Without knowing the global structure of the geometry, both can be correct. This is known as the normal orientation problem. Open3D tries to orient the normal to align with the original normal if it exists. Otherwise, Open3D does a random guess. Further orientation functions such as ``orient_normals_to_align_with_direction`` and ``orient_normals_towards_camera_location`` need to be called if the orientation is a concern.

Use ``draw_geometries`` to visualize the point cloud and press :kbd:`n` to see point normal. Key :kbd:`-` and key :kbd:`+` can be used to control the length of the normal.

.. image:: ../../_static/Basic/pointcloud/downsampled_normal.png
    :width: 400px

.. _crop_point_cloud:

Crop point cloud
=====================================

.. code-block:: python

    print("We load a polygon volume and use it to crop the original point cloud")
    vol = read_selection_polygon_volume("../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    draw_geometries([chair])
    print("")

``read_selection_polygon_volume`` reads a json file that specifies polygon selection area.
``vol.crop_point_cloud(pcd)`` filters out points. Only the chair remains.

.. image:: ../../_static/Basic/pointcloud/crop.png
    :width: 400px

.. _paint_point_cloud:

Paint point cloud
=====================================

.. code-block:: python

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    draw_geometries([chair])
    print("")

``paint_uniform_color`` paints all the points to a uniform color. The color is in RGB space, [0, 1] range.

.. image:: ../../_static/Basic/pointcloud/crop_color.png
    :width: 400px
