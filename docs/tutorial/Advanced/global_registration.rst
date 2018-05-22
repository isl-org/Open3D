.. _global_registration:

Global registration
-------------------------------------

Both :ref:`icp_registration` and :ref:`colored_point_registration` are known as **local** registration methods because they rely on a rough alignment as initialization. This tutorial shows another class of registration methods, known as **global** registration. This family of algorithms do not require an alignment for initialization. They usually produce less tight alignment results and are used as initialization of the local methods.

.. code-block:: python

    # src/Python/Tutorial/Advanced/global_registration.py

    from open3d import *
    import numpy as np
    import copy

    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        draw_geometries([source_temp, target_temp])

    def prepare_dataset(voxel_size):
        print(":: Load two point clouds and disturb initial pose.")
        source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        draw_registration_result(source, target, np.identity(4))

        print(":: Downsample with a voxel size %.3f." % voxel_size)
        source_down = voxel_down_sample(source, voxel_size)
        target_down = voxel_down_sample(target, voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        estimate_normals(source_down, KDTreeSearchParamHybrid(
                radius = radius_normal, max_nn = 30))
        estimate_normals(target_down, KDTreeSearchParamHybrid(
                radius = radius_normal, max_nn = 30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        source_fpfh = compute_fpfh_feature(source_down,
                KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
        target_fpfh = compute_fpfh_feature(target_down,
                KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, 0.075,
                TransformationEstimationPointToPoint(False), 4,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(0.075)],
                RANSACConvergenceCriteria(4000000, 500))
        return result

    def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = registration_icp(source, target, distance_threshold,
                result_ransac.transformation,
                TransformationEstimationPointToPlane())
        return result

    if __name__ == "__main__":
        voxel_size = 0.05 # means 5cm for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                source_fpfh, target_fpfh, voxel_size)
        print(result_ransac)
        draw_registration_result(source_down, target_down,
                result_ransac.transformation)

        result_icp = refine_registration(source, target,
                source_fpfh, target_fpfh, voxel_size)
        print(result_icp)
        draw_registration_result(source, target, result_icp.transformation)

Input
````````````````````````

.. code-block:: python

    # in prepare_dataset function

    print(":: Load two point clouds and disturb initial pose.")
    source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
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

    # in prepare_dataset function

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    source_down = voxel_down_sample(source, voxel_size)
    target_down = voxel_down_sample(target, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    estimate_normals(source_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))
    estimate_normals(target_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    source_fpfh = compute_fpfh_feature(source_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
    target_fpfh = compute_fpfh_feature(target_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))

We down sample the point cloud, estimate normals, then compute a FPFH feature for each point. The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point. A nearest neighbor query in the 33-dimensinal space can return points with similar local geometric structures. See [Rasu2009]_ for details.

.. _feature_matching:

RANSAC
``````````````````````````````````````

.. code-block:: python

    # in execute_global_registration function

    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, 0.075,
            TransformationEstimationPointToPoint(False), 4,
            [CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(0.075)],
            RANSACConvergenceCriteria(4000000, 500))

We use RANSAC for global registration. In each RANSAC iteration, ``ransac_n`` random points are picked from the source point cloud. Their corresponding points in the target point cloud are detected by querying the nearest neighbor in the 33-dimensional FPFH feature space. A pruning step takes fast pruning algorithms  to quickly reject false matches early.

Open3D provides the following pruning algorithms:

- ``CorrespondenceCheckerBasedOnDistance`` checks if aligned point clouds are close (less than specified threshold).
- ``CorrespondenceCheckerBasedOnEdgeLength`` checks if the lengths of any two arbitrary edges (line formed by two vertices) individually drawn from source and target correspondences are similar. This tutorial checks that :math:`||edge_{source}|| > 0.9 \times ||edge_{target}||` and :math:`||edge_{target}|| > 0.9 \times ||edge_{source}||` are true.
- ``CorrespondenceCheckerBasedOnNormal`` considers vertex normal affinity of any correspondences. It computes dot product of two normal vectors. It takes radian value for the threshold.

Only matches that pass the pruning step are used to compute a transformation, which is validated on the entire point cloud. The core function is ``registration_ransac_based_on_feature_matching``. The most important hyperparameter of this function is ``RANSACConvergenceCriteria``. It defines the maximum number of RANSAC iterations and the maximum number of validation steps. The larger these two numbers are, the more accurate the result is, but also the more time the algorithm takes.

We set the RANSAC parameters based on the empirical value provided by [Choi2015]_. The result is

.. image:: ../../_static/Advanced/global_registration/ransac.png
    :width: 400px

.. Note:: Open3D provides faster implementation for global registration. Please refer :ref:`fast_global_registration`.

.. _local_refinement:

Local refinement
``````````````````````````````````````

For performance reason, the global registration is only performed on a heavily down-sampled point cloud. The result is also not tight. We use :ref:`point_to_plane_icp` to further refine the alignment.

.. code-block:: python

    # in refine_registration function

    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = registration_icp(source, target, distance_threshold,
            result_ransac.transformation,
            TransformationEstimationPointToPlane())

Outputs a tight alignment. This summarizes a complete pairwise registration workflow.

.. image:: ../../_static/Advanced/global_registration/icp.png
    :width: 400px
