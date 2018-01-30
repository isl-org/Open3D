.. _colored_point_registration:

Colored point cloud registration
-------------------------------------

This tutorial demonstrates an ICP variant that uses both geometry and color for registration. It implements the algorithm of [Park2017]_. The color information locks the alignment along the tangent plane. Thus this algorithm is more accurate and more robust than prior point cloud registration algorithms, while the running speed is comparable to that of ICP registration. This tutorial uses notations from :ref:`icp_registration`.

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

Helper visualization function
``````````````````````````````````````

.. code-block:: python

    def draw_registration_result_original_color(source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        draw_geometries([source_temp, target])

In order to demonstrate the alignment between colored point clouds, ``draw_registration_result_original_color`` renders point clouds with their original color.

Input
```````````````

.. code-block:: python

    print("1. Load two point clouds and show initial pose")
    source = read_point_cloud("../../TestData/ColoredICP/frag_115.ply")
    target = read_point_cloud("../../TestData/ColoredICP/frag_116.ply")

    # draw initial alignment
    current_transformation = np.identity(4)
    draw_registration_result_original_color(
            source, target, current_transformation)

This script reads a source point cloud and a target point cloud from two files. An identity matrix is used as initialization.

.. image:: ../../_static/Advanced/colored_pointcloud_registration/initial.png
    :width: 325px

.. image:: ../../_static/Advanced/colored_pointcloud_registration/initial_side.png
    :width: 325px


.. _geometric_alignment:

Point-to-plane ICP
``````````````````````````````````````

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

We first run :ref:`point_to_plane_icp` as a baseline approach. The visualization below shows misaligned green triangle textures. This is because geometric constraint does not prevent two planar surfaces from slipping.

.. image:: ../../_static/Advanced/colored_pointcloud_registration/point_to_plane.png
    :width: 325px

.. image:: ../../_static/Advanced/colored_pointcloud_registration/point_to_plane_side.png
    :width: 325px


.. _multi_scale_geometric_color_alignment:

Colored point cloud registration
``````````````````````````````````````````````

The core function for colored point cloud registration is ``registration_colored_icp``. Following [Park2017]_, it runs ICP iterations (see :ref:`point_to_point_icp` for details) with a joint optimization objective

.. math:: E(\mathbf{T}) = (1-\delta)E_{C}(\mathbf{T}) + \delta E_{G}(\mathbf{T}),

where :math:`\mathbf{T}` is the transformation matrix to be estimated. :math:`E_{C}` and :math:`E_{G}` are the photometric and geometric terms, respectively. :math:`\delta\in[0,1]` is a weight parameter that has been determined empirically.

The geometric term :math:`E_{G}` is the same as the :ref:`point_to_plane_icp` objective

.. math:: E_{G}(\mathbf{T}) = \sum_{(\mathbf{p},\mathbf{q})\in\mathcal{K}}\big((\mathbf{p} - \mathbf{T}\mathbf{q})\cdot\mathbf{n}_{\mathbf{p}}\big)^{2},

where :math:`\mathcal{K}` is the correspondence set in the current iteration. :math:`\mathbf{n}_{\mathbf{p}}` is the normal of point :math:`\mathbf{p}`.

The color term :math:`E_{C}` measures the difference between the color of point :math:`\mathbf{q}` (denoted as :math:`C(\mathbf{q})`) and the color of its projection on the tangent plane of :math:`\mathbf{p}`.

.. math:: E_{C}(\mathbf{T}) = \sum_{(\mathbf{p},\mathbf{q})\in\mathcal{K}}\big(C_{\mathbf{p}}(\mathbf{f}(\mathbf{T}\mathbf{q})) - C(\mathbf{q})\big)^{2},

where :math:`C_{\mathbf{p}}(\cdot)` is a precomputed function continuously defined on the tangent plane of :math:`\mathbf{p}`. Function :math:`\mathbf{f}(\cdot)` projects a 3D point to the tangent plane. More details refer to [Park2017]_.

To further improve efficiency, [Park2017]_ proposes a multi-scale registration scheme. This has been implemented in the following script.

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

In total, 3 layers of multi-resolution point clouds are created with :ref:`voxel_downsampling`. Normals are computed with :ref:`vertex_normal_estimation`. The core registration function ``registration_colored_icp`` is called for each layer, from coarse to fine. The output is a tight alignment of the two point clouds. Notice the green triangles on the wall.

.. image:: ../../_static/Advanced/colored_pointcloud_registration/colored.png
    :width: 325px

.. image:: ../../_static/Advanced/colored_pointcloud_registration/colored_side.png
    :width: 325px
