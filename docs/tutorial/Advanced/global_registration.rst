.. _global_registration:

Global registration
-------------------------------------

Both :ref:`icp_registration` and :ref:`colored_point_registration` are known as **local** registration methods because they rely on a rough alignment as initialization. This tutorial shows another class of registration methods, known as **global** registration. This family of algorithms do not require an alignment for initialization. They usually produce less tight alignment results and are used as initialization of the local methods.

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

Input
````````````````````````

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

This script reads a source point cloud and a target point cloud from two files. They are misaligned with an identity matrix as transformation.

.. image:: ../../_static/Advanced/global_registration/initial.png
    :width: 400px

.. _extract_geometric_feature:

Extract geometric feature
``````````````````````````````````````

.. code-block:: python

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

We down sample the point cloud, estimate normals, then compute a FPFH feature for each point. The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point. A nearest neighbor query in the 33-dimensinal space can return points with similar local geometric structures. See [Rasu2009]_ for details.

.. _feature_matching:

RANSAC
``````````````````````````````````````

.. code-block:: python

    print("5. RANSAC registration on down-sampled point clouds.")
    print("   Since the downsampling voxel size is 0.05, we use a liberal")
    print("   distance threshold 0.075.")
    result_ransac = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            fpfh, max_correspondence_distance = 0.075,
            TransformationEstimationPointToPoint(False),
            ransac_n = 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(0.075)],
            RANSACConvergenceCriteria(max_iteration = 4000000, max_validation = 500))
    print(result_ransac)
    draw_registration_result(source_down, target_down,
            result_ransac.transformation)

We use RANSAC for global registration. In each RANSAC iteration, ``ransac_n`` random points are picked from the source point cloud. Their corresponding points in the target point cloud are detected by querying the nearest neighbor in the 33-dimensional FPFH feature space. A pruning step takes fast pruning algorithms such as ``CorrespondenceCheckerBasedOnEdgeLength`` and ``CorrespondenceCheckerBasedOnDistance`` to quickly reject false matches early. Only matches that pass the pruning step are used to compute a transformation, which is validated on the entire point cloud.

The core function is ``registration_ransac_based_on_feature_matching``. The most important hyperparameter of this function is ``RANSACConvergenceCriteria``. It defines the maximum number of RANSAC iterations and the maximum number of validation steps. The larger these two numbers are, the more accurate the result is, but also the more time the algorithm takes.

We set the RANSAC parameters based on the empirical value provided by [Choi2015]_. The result is

.. image:: ../../_static/Advanced/global_registration/ransac.png
    :width: 400px

.. _local_refinement:

Local refinement
``````````````````````````````````````

For performance reason, the global registration is only performed on a heavily down-sampled point cloud. The result is also not tight. We use :ref:`point_to_plane_icp` to further refine the alignment.

.. code-block:: python

    print("6. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold 0.02.")
    result_icp = registration_icp(source, target, 0.02,
            result_ransac.transformation,
            TransformationEstimationPointToPlane())
    print(result_icp)
    draw_registration_result(source, target, result_icp.transformation)

Outputs a tight alignment. This summarizes a complete pairwise registration workflow.

.. image:: ../../_static/Advanced/global_registration/icp.png
    :width: 400px
