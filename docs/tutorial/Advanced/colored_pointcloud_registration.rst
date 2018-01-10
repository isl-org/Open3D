.. _colored_point_registration:

Colored point cloud registration
-------------------------------------

Open3D provides ICP for point cloud alignment (refer :ref:`registration` in basic tutorial).
Both point-to-point ICP and point-to-plane ICP only considers geometric alignment.
There is a case the geometry does not helpful for the true alignment, but
only the color texture of the point cloud is evidence for the correct alignment.

This tutorial introduces a sophisticated registration method
that considers color texture of point cloud as well. This is a tutorial script.

.. code-block:: python

    # src/Python/Tutorial/Advanced/colored_pointcloud_registration.py

    import sys
    import numpy as np
    import copy
    sys.path.append("../..")
    from py3d import *


    def draw_registration_result_original_color(source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        draw_geometries([source_temp, target])


    if __name__ == "__main__":

        print("1. Load two point clouds and show initial pose")
        source = read_point_cloud("../../TestData/ColoredICP/frag_115.ply")
        target = read_point_cloud("../../TestData/ColoredICP/frag_116.ply")

        # draw initial alignment
        current_transformation = np.identity(4)
        draw_registration_result_original_color(
                source, target, current_transformation)

        # point to plane ICP
        current_transformation = np.identity(4);
        print("2. Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. Distance threshold 0.02.")
        result_icp = registration_icp(source, target, 0.02,
                current_transformation, TransformationEstimationPointToPlane())
        print(result_icp)
        draw_registration_result_original_color(
                source, target, result_icp.transformation)

        # colored pointcloud registration
         # This is implementation of following paper
         # J. Park, Q.-Y. Zhou, V. Koltun,
         # Colored Point Cloud Registration Revisited, ICCV 2017
        voxel_radius = [ 0.04, 0.02, 0.01 ];
        max_iter = [ 50, 30, 14 ];
        current_transformation = np.identity(4)
        print("3. Colored point cloud registration")
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter,radius,scale])

            print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = voxel_down_sample(source, radius)
            target_down = voxel_down_sample(target, radius)

            print("3-2. Estimate normal.")
            estimate_normals(source_down, KDTreeSearchParamHybrid(
                    radius = radius * 2, max_nn = 30))
            estimate_normals(target_down, KDTreeSearchParamHybrid(
                    radius = radius * 2, max_nn = 30))

            print("3-3. Applying colored point cloud registration")
            result_icp = registration_colored_icp(source_down, target_down,
                    radius, current_transformation,
                    ICPConvergenceCriteria(relative_fitness = 1e-6,
                    relative_rmse = 1e-6, max_iteration = iter))
            current_transformation = result_icp.transformation
            print(result_icp)
        draw_registration_result_original_color(
                source, target, result_icp.transformation)


.. _visualize_color_alignment:

Visualize color alignment
``````````````````````````````````````
Function ``draw_registration_result_original_color`` in this tutorial simply
visualizes multiple geometries using original color.

.. code-block:: python

    def draw_registration_result_original_color(source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        draw_geometries([source_temp, target])

The function makes hard copy of source point cloud to make source point cloud intact.
Note that ``draw_geometries`` can take a list of geometries and displays geometries in the list altogether.

.. code-block:: python

    print("1. Load two point clouds and show initial pose")
    source = read_point_cloud("../../TestData/ColoredICP/frag_115.ply")
    target = read_point_cloud("../../TestData/ColoredICP/frag_116.ply")

    # draw initial alignment
    current_transformation = np.identity(4)
    draw_registration_result_original_color(
            source, target, current_transformation)

This script displays below geometry

.. image:: ../../_static/Advanced/colored_pointcloud_registration/initial.png
    :width: 325px

.. image:: ../../_static/Advanced/colored_pointcloud_registration/initial_side.png
    :width: 325px

[first figure: front view] [second figure: side view]


.. _geometric_alignment:

Geometric alignment
``````````````````````````````````````

The next part of the script shows the alignment result using :ref:`point_to_plane_icp`.

.. code-block:: python

    # point to plane ICP
    current_transformation = np.identity(4);
    print("2. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    result_icp = registration_icp(source, target, 0.02,
            current_transformation, TransformationEstimationPointToPlane())
    print(result_icp)
    draw_registration_result_original_color(
            source, target, result_icp.transformation)

As the point-to-plane ICP does not consider color texture of point cloud, this produces following result. In a geometric view point, the two planar point clouds looks well aligned, but it is not optimal as the color texture is not correctly aligned.

.. image:: ../../_static/Advanced/colored_pointcloud_registration/point_to_plane.png
    :width: 325px

.. image:: ../../_static/Advanced/colored_pointcloud_registration/point_to_plane_side.png
    :width: 325px

[first figure: front view] [second figure: side view]


.. _multi_scale_geometric_color_alignment:

Multi-scale geometric + color alignment
``````````````````````````````````````````````

The next part of the tutorial script demonstrates colored point cloud registration.

.. code-block:: python

    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [ 0.04, 0.02, 0.01 ];
    max_iter = [ 50, 30, 14 ];
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter,radius,scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = voxel_down_sample(source, radius)
        target_down = voxel_down_sample(target, radius)

        print("3-2. Estimate normal.")
        estimate_normals(source_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        estimate_normals(target_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))

        print("3-3. Applying colored point cloud registration")
        result_icp = registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = iter))
        current_transformation = result_icp.transformation
        print(result_icp)
        draw_registration_result_original_color(
                source, target, result_icp.transformation)

This script is implementation of paper [Park2017]_.
The cost function of this method is linear combination of point-to-plane cost and
vertex intensity matching cost.
This simple extension allows to consider geometric as well as photometric assessment.

The script repetitively calls ``registration_colored_icp`` with various scale space.
The scale space idea is similar to multi-scale image alignment:
two images are downsampled, and aligned in lower resolution, and gradually refined in higher image resolution.
This multi-scale approach is helpful to handle large baseline.

For handling point clouds, the multi-scale idea is implemented as follows.

- Set output transformation matrix as identity
- Iterate from lower resolution to higher resolution

    - resampling original point cloud using ``voxel_down_sample``
    - estimate vertex normal of resampled point cloud using ``estimate_normals``
    - apply color ICP using ``registration_colored_icp``
    - update output transformation matrix

Refer :ref:`voxel_downsampling` and :ref:`vertex_normal_estimation` for more details about basic point cloud operation. The script produces following result. The planar points are aligned well and texture of point clouds matches.

.. image:: ../../_static/Advanced/colored_pointcloud_registration/colored.png
    :width: 325px

.. image:: ../../_static/Advanced/colored_pointcloud_registration/colored_side.png
    :width: 325px

[first figure: front view] [second figure: side view]
