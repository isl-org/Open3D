.. _global_registration:

Global registration
-------------------------------------

The registration methods (:ref:`registration` and :ref:`colored_point_registration`)
introduced so far is fit for small amount of misalignment. It is often referred as **local** registration.
In practice, there is a need to consider more challenging cases if the pose of two point clouds are significantly different. In these case, the iteration of **local** registration tempts to be stuck at local minima.

This tutorial introduces **global** registration method that can register two point clouds
regardless how challenge their initial poses. The example script is shown below.

.. code-block:: python

    # src/Python/Tutorial/Advanced/global_registration.py

    import sys
    sys.path.append("../..")
    from py3d import *
    import numpy as np
    import copy

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        draw_geometries([source_temp, target_temp])

    if __name__ == "__main__":

        print("1. Load two point clouds and disturb initial pose.")
        source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
        trans_init = np.asarray([[0.0, 1.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        draw_registration_result(source, target, np.identity(4))

        print("2. Downsample with a voxel size 0.05.")
        source_down = voxel_down_sample(source, 0.05)
        target_down = voxel_down_sample(target, 0.05)

        print("3. Estimate normal with search radius 0.1.")
        estimate_normals(source_down, KDTreeSearchParamHybrid(
                radius = 0.1, max_nn = 30))
        estimate_normals(target_down, KDTreeSearchParamHybrid(
                radius = 0.1, max_nn = 30))

        print("4. Compute FPFH feature with search radius 0.25")
        source_fpfh = compute_fpfh_feature(source_down,
                KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
        target_fpfh = compute_fpfh_feature(target_down,
                KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))

        print("5. RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is 0.05, we use a liberal")
        print("   distance threshold 0.075.")
        result_ransac = registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, 0.075,
                TransformationEstimationPointToPoint(False), 4,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(0.075)],
                RANSACConvergenceCriteria(4000000, 500))
        print(result_ransac)
        draw_registration_result(source_down, target_down,
                result_ransac.transformation)

        print("6. Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold 0.02.")
        result_icp = registration_icp(source, target, 0.02,
                result_ransac.transformation,
                TransformationEstimationPointToPlane())
        print(result_icp)
        draw_registration_result(source, target, result_icp.transformation)

This script reads two point clouds, and intentionally challenge the initial pose. The later part of the script
registers point clouds using a **global** registration method and refine the registration using a **local** registration method.

Let's see the script from the first part.

.. code-block:: python

    print("1. Load two point clouds and disturb initial pose.")
    source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

This code snippet reads two point clouds as source and target.
The source point cloud is intentionally transformed with custom transformation matrix.
This script uses ``draw_registration_result`` that is introduced in :ref:`visualize_registration`.

The script displays two point clouds like below. This is challenging for ICP based methods.

.. image:: ../../_static/Advanced/global_registration/initial.png
    :width: 400px


.. _extract_geometric_feature:

Extract geometric feature
``````````````````````````````````````

To recover correct poses, it is necessary to extract some information from point cloud
that is invariant for the any poses of point cloud. Likewise local descriptors in images, there is
series of work that extracts pose invariant description of point clouds.
This is widely referred as **geometric feature**, or 3D feature.
Open3D provides FPFH [Rasu2009]_ as a default geometric feature.

Extracting feature descriptor from very dense point clouds is often prohibited as
it takes a long time. One good trick is to downsample point cloud and extract
geometric feature from downsampled points.

.. code-block:: python

    print("2. Downsample with a voxel size 0.05.")
    source_down = voxel_down_sample(source, 0.05)
    target_down = voxel_down_sample(target, 0.05)

    print("3. Estimate normal with search radius 0.1.")
    estimate_normals(source_down, KDTreeSearchParamHybrid(
            radius = 0.1, max_nn = 30))
    estimate_normals(target_down, KDTreeSearchParamHybrid(
            radius = 0.1, max_nn = 30))

Please refer :ref:`voxel_downsampling` and :ref:`vertex_normal_estimation` for more details.

As a next step, the script extracts geometric feature from downsampled point cloud.

.. code-block:: python

    print("4. Compute FPFH feature with search radius 0.25")
    source_fpfh = compute_fpfh_feature(source_down,
            KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
    target_fpfh = compute_fpfh_feature(target_down,
            KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))

``source_fpfh`` and ``target_fpfh`` are lists of 33 dimensional descriptors computed for every points. These two descriptors are matched using following script.


.. _feature_matching:

Feature matching
``````````````````````````````````````

Once geometric feature is extracted from point cloud, it can be matched to the feature of the other point cloud.
The feature matching is a problem of finding correct matches while avoiding many false positives.
By default, Open3D supports RANSAC based approach [Choi2015]_ for advanced geometric feature matching. Following script matches ``source_fpfh`` and ``target_fpfh``.

.. code-block:: python

    print("5. RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is 0.05, we use a liberal")
    print("   distance threshold 0.075.")
    result_ransac = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, 0.075,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(0.075)],
            RANSACConvergenceCriteria(4000000, 500))
    print(result_ransac)
    draw_registration_result(source_down, target_down,
            result_ransac.transformation)

The RANSAC is based on following idea

- Find a best matching descriptors and build correspondence set
- Iterate following loop

    - sample a few correspondences from the correspondence set
    - compute transformation matrix using the sampled correspondences
    - apply the computed transformation to source point cloud and count inlier points
    - update output if the computed transformation is better than that of in prior iterations

Function ``registration_ransac_based_on_feature_matching`` performs RANSAC based matching and takes several arguments. To list:

- source and target point clouds: ``source_down, target_down``
- n-dimentional feature descriptors: ``source_fpfh, target_fpfh``
- distance threshold that is used for determining inliers: ``0.075``
- a method for computing transformation matrix: ``TransformationEstimationPointToPoint``
- number of sampling correspondences: ``4``
- a list of correspondence checking criterion:

    - ``CorrespondenceCheckerBasedOnEdgeLength`` specify affordable length difference between edge from a source point set compared to edge of the target point set
    - ``CorrespondenceCheckerBasedOnDistance`` specify maximum Euclidean distance to consider transformed points are adjacent to another points

- RANSAC parameters: ``RANSACConvergenceCriteria(4000000, 500)``

    - maximum allowable iteration is ``4000000``
    - can terminate after ``500`` iteration if all the criterions are met

The estimated transformation from RANSAC loop is stored in ``result_ransac.transformation``. The script displays following registration.

.. image:: ../../_static/Advanced/global_registration/ransac.png
    :width: 400px

Note that the purpose of global registration is to get rough initial pose. The point clouds are downsampled, and the alignment is not perfect as the transformation is estimated just from a few feature correspondences.

.. _local_refinement:

Local refinement
``````````````````````````````````````

The registration result from RANSAC is good for challenging initial poses, but not guarantee tight alignment.
The final step for the global registration is local refinement. The tutorial uses :ref:`point_to_plane_icp`.

.. code-block:: python

    print("6. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold 0.02.")
    result_icp = registration_icp(source, target, 0.02,
            result_ransac.transformation,
            TransformationEstimationPointToPlane())
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

This script aligns two point cloud using ``result_icp.transformation`` as an initial pose. Note that it uses original point clouds not downsampled ones for more accurate result.

The final result is shown below.

.. image:: ../../_static/advanced/global_registration/icp.png
    :width: 400px

For the case where the colored texture alignment is important, point-to-plane ICP can be replaced with :ref:`colored_point_registration`.
