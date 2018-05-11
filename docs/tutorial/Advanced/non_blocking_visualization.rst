.. _non_blocking_visualization:

Non-blocking visualization
-------------------------------------

``draw_geometries()`` is useful for quick overview of static geometries. However, this function holds process until the visualization window is closed. This is not optimal when visualized geometry is needed to be updated without closing the window. This tutorial introduces useful methods in ``Visualizer`` class for this need.

.. code-block:: python

    # src/Python/Tutorial/Advanced/non_blocking_visualization.py

    # Open3D: www.open3d.org
    # The MIT License (MIT)
    # See license file or visit www.open3d.org for details

    from py3d import *
    import numpy as np
    import copy

    if __name__ == "__main__":
        set_verbosity_level(VerbosityLevel.Debug)
        source_raw = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        target_raw = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
        source = voxel_down_sample(source_raw, voxel_size = 0.02)
        target = voxel_down_sample(target_raw, voxel_size = 0.02)
        trans = [[0.862, 0.011, -0.507,  0.0],
                [-0.139, 0.967, -0.215,  0.7],
                [0.487, 0.255,  0.835, -1.4],
                [0.0, 0.0, 0.0, 1.0]]
        source.transform(trans)

        flip_transform = [[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]]
        source.transform(flip_transform)
        target.transform(flip_transform)

        vis = Visualizer()
        vis.create_window()
        vis.add_geometry(source)
        vis.add_geometry(target)
        threshold = 0.05
        icp_iteration = 100
        save_image = False

        for i in range(icp_iteration):
            reg_p2l = registration_icp(source, target, threshold,
                    np.identity(4), TransformationEstimationPointToPlane(),
                    ICPConvergenceCriteria(max_iteration = 1))
            source.transform(reg_p2l.transformation)
            vis.update_geometry()
            vis.reset_view_point(True)
            vis.poll_events()
            if save_image:
                vis.capture_screen_image("temp_%04d.jpg" % i)
        vis.destroy_window()

This script visualizes a live view of point cloud registration.

Prepare example data
````````````````````````````````````````````````````
.. code-block:: python

    set_verbosity_level(VerbosityLevel.Debug)
    source_raw = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target_raw = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    source = voxel_down_sample(source_raw, voxel_size = 0.02)
    target = voxel_down_sample(target_raw, voxel_size = 0.02)
    trans = [[0.862, 0.011, -0.507,  0.0],
            [-0.139, 0.967, -0.215,  0.7],
            [0.487, 0.255,  0.835, -1.4],
            [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)

    flip_transform = [[1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)

For ICP registration, this script reads two point clouds, downsample them. The source point cloud is transformed for misalignment. Both point clouds are flipped for better visualization.


Mimic draw_geometries() with Visualizer class
````````````````````````````````````````````````````

.. code-block:: python

    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)

These lines make an instance of visualizer class, opens a visualizer window, and add two geometries to the visualizer.

Transform geometry and visualize it
````````````````````````````````````````````````````

.. code-block:: python

    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = registration_icp(source, target, threshold,
                np.identity(4), TransformationEstimationPointToPlane(),
                ICPConvergenceCriteria(max_iteration = 1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry()
        vis.reset_view_point(True)
        vis.poll_events()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()

Each for-loop calls ``registration_icp``, but note that it forces only one ICP iteration using ``ICPConvergenceCriteria(max_iteration = 1)``. This is a trick to retrieve pose update from a single ICP iteration. After single iteration ICP, source geometry is transformed accordingly.

The next part of the script is the core of this tutorial. ``update_geometry`` informs any geometries in ``vis`` is updated. ``reset_view_point`` updates view point based on the updated geometries. By calling ``poll_events``, visualizer renders new frame. After for-loop finishes, ``destroy_window`` closes the window.

The result looks like below.

.. image:: ../../_static/Advanced/non_blocking_visualization/visualize_icp_iteration.gif
    :width: 400px
