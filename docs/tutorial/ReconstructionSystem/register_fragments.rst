.. _reconstruction_system_register_fragments:

Register fragments
-------------------------------------

Once the fragments of the scene obtained from :ref:`reconstruction_system_make_fragments`, the next step is register fragments and optimize pairwise registration altogether.

This tutorial reviews `src/Python/Tutorial/ReconstructionSystem/register_fragments.py <../../../../../src/Python/Tutorial/ReconstructionSystem/register_fragments.py>`_ function by function.


Input arguments
``````````````````````````````````````

.. code-block:: python

    if __name__ == "__main__":
        set_verbosity_level(VerbosityLevel.Debug)
        parser = argparse.ArgumentParser(description="register fragments.")
        parser.add_argument("path_dataset", help="path to the dataset")
        args = parser.parse_args()

        ply_file_names = get_file_list(args.path_dataset + folder_fragment, ".ply")
        make_folder(args.path_dataset + folder_scene)
        register_point_cloud(args.path_dataset, ply_file_names)
        optimize_posegraph_for_scene(args.path_dataset)

This script runs with ``python make_fragments.py [path]``. [path] should have subfolders *fragments* where ply files and posegraph json files of the fragments are in. This script must run after :ref:`reconstruction_system_make_fragments`.

The main function runs ``register_point_cloud`` and ``optimize_posegraph_for_scene``. These functions register any pairs of fragments and optimize a posegraph.


Preprocess point cloud
``````````````````````````````````````

.. code-block:: python

    def preprocess_point_cloud(pcd):
        pcd_down = voxel_down_sample(pcd, 0.05)
        estimate_normals(pcd_down,
                KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
        pcd_fpfh = compute_fpfh_feature(pcd_down,
                KDTreeSearchParamHybrid(radius = 0.25, max_nn = 100))
        return (pcd_down, pcd_fpfh)

This function downsample point cloud to make a point cloud sparser and regularly distributed. This is virtue as it avoids biased point cloud density, and helps better performance for registration methods (see :ref:`voxel_downsampling` for more details).

Point cloud normal is estimated from the downsampled points with larger search radius for covariance analysis (see :ref:`vertex_normal_estimation` for more details).

The FPFH feature extracted from downsampled point clouds instead of using original point cloud to save computation time (see :ref:`extract_geometric_feature` for more details).  All the unit in this script in meter unit.


.. _reconstruction_system_feature_matching:

Feature matching
``````````````````````````````````````

This function matches two point clouds using feature descriptor. Using modified RANSAC based matching scheme [Choi2015]_, ``registration_ransac_based_on_feature_matching`` outputs 4x4 registration matrix. Please refer :ref:`feature_matching` for detaild explanation of input arguments.

.. code-block:: python

    def register_point_cloud_fpfh(source, target,
            source_fpfh, target_fpfh):
        result_ransac = registration_ransac_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh, 0.075,
                TransformationEstimationPointToPoint(False), 4,
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                CorrespondenceCheckerBasedOnDistance(0.075),
                CorrespondenceCheckerBasedOnNormal(0.52359878)],
                RANSACConvergenceCriteria(4000000, 2000))
        if (result_ransac.transformation.trace() == 4.0):
            return (False, np.identity(4))
        else:
            return (True, result_ransac)

There are two return cases. If the registration result is identity, it means matching failure, so the function returns failure signal ``False``.


.. _reconstruction_system_compute_initial_registration:

Compute initial registration
``````````````````````````````````````

This function provides initial alignment. The initial alignment feeds into fine-grained registration in the next step. Let's see the function below.

.. code-block:: python

    def compute_initial_registration(s, t, source_down, target_down,
            source_fpfh, target_fpfh, path_dataset, draw_result = False):

        if t == s + 1: # odometry case
            print("Using RGBD odometry")
            pose_graph_frag = read_pose_graph(path_dataset +
                    template_fragment_posegraph_optimized % s)
            n_nodes = len(pose_graph_frag.nodes)
            transformation = np.linalg.inv(
                    pose_graph_frag.nodes[n_nodes-1].pose)
            print(pose_graph_frag.nodes[0].pose)
            print(transformation)
        else: # loop closure case
            print("register_point_cloud_fpfh")
            (success_ransac, result_ransac) = register_point_cloud_fpfh(
                    source_down, target_down,
                    source_fpfh, target_fpfh)
            if not success_ransac:
                print("No resonable solution. Skip this pair")
                return (False, np.identity(4))
            else:
                transformation = result_ransac.transformation
            print(transformation)

        if draw_result:
            draw_registration_result(source_down, target_down,
                    transformation)
        return (True, transformation)


There are two cases how the initial alignment is computed.

- Case 1: if the source and target fragment are from sequential frames (for example fragment_000.ply and fragment_001.ply pair), the function uses the camera pose of the last frame in source fragment

    - The canonical domain of the source fragment is identity
    - Therefore the inverse matrix of the last camera pose can be good approximate for source to target transformation matrix
    - This corresponds to the code ``transformation = np.linalg.inv(pose_graph_frag.nodes[n_nodes-1].pose)``

- Case 2: if not case 1, do global registration using geometric feature

    - Function ``register_point_cloud_fpfh`` is called in this case



Fine-grained registration
``````````````````````````````````````

The following two functions are for fine-grained registration of point clouds. These two functions uses rough transformation matrix obtained from  :ref:`reconstruction_system_feature_matching` as an initial matrix. One of this two functions conditionally runs depending on user selection.

The first function ``register_point_cloud_icp`` is point-to-plane ICP [ChenAndMedioni1992]_. It minimizes geometric alignment. The detailed explanations can be found from :ref:`reconstruction_system_compute_initial_registration`.

.. code-block:: python

    def register_point_cloud_icp(source, target,
            init_transformation = np.identity(4)):
        result_icp = registration_icp(source, target, 0.02,
                init_transformation,
                TransformationEstimationPointToPlane())
        print(result_icp)
        information_matrix = get_information_matrix_from_point_clouds(
                source, target, 0.075, result_icp.transformation)
        return (result_icp.transformation, information_matrix)


The another function ``register_colored_point_cloud_icp`` implements colored ICP [Park2017]_. It also uses multi-scale approach to cover larger baselines and to avoid local minima. It has hybrid term that jointly optimizes alignment of colored texture and geometries.

.. code-block:: python

    def register_colored_point_cloud_icp(source, target,
            init_transformation = np.identity(4), draw_result = False):
        voxel_radius = [ 0.05, 0.025, 0.0125 ]
        max_iter = [ 50, 30, 14 ]
        current_transformation = init_transformation
        for scale in range(3): # multi-scale approach
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print("radius %f" % radius)
            source_down = voxel_down_sample(source, radius)
            target_down = voxel_down_sample(target, radius)
            estimate_normals(source_down, KDTreeSearchParamHybrid(
                    radius = radius * 2, max_nn = 30))
            print(np.asarray(source_down.normals))
            estimate_normals(target_down, KDTreeSearchParamHybrid(
                    radius = radius * 2, max_nn = 30))
            result_icp = registration_colored_icp(source_down, target_down,
                    radius, current_transformation,
                    ICPConvergenceCriteria(relative_fitness = 1e-6,
                    relative_rmse = 1e-6, max_iteration = iter))
            current_transformation = result_icp.transformation

        information_matrix = get_information_matrix_from_point_clouds(
                source, target, 0.075, result_icp.transformation)
        if draw_result:
            draw_registration_result_original_color(source, target,
                    result_icp.transformation)
        return (result_icp.transformation, information_matrix)

This function is introduced in tutorial :ref:`colored_point_registration`. Please refer it for more details.

Below function ``local_refinement`` calls one of function (``register_point_cloud_icp`` or ``register_colored_point_cloud_icp``) for fine-grained refinement of initial registration.

.. code-block:: python

    def local_refinement(source, target, source_down, target_down,
            transformation_init, registration_type = "color",
            draw_result = False):

        if (registration_type == "color"):
            print("register_colored_point_cloud")
            (transformation, information) = \
                    register_colored_point_cloud_icp(
                    source, target, transformation_init)
        else:
            print("register_point_cloud_icp")
            (transformation, information) = \
                    register_point_cloud_icp(
                    source_down, target_down, transformation_init)

        if draw_result:
            draw_registration_result_original_color(
                    source_down, target_down, transformation)
        return (transformation, information)


Make a posegraph
``````````````````````````````````````

After local-refinement, the next step is to make a posegraph. The posegraph is necessary to optimize all the pairwise alignments to make globally tight alignment of every point clouds.

.. code-block:: python

    def update_posegrph_for_scene(s, t, transformation, information,
            odometry, pose_graph):

        print("Update PoseGraph")
        if t == s + 1: # odometry case
            odometry = np.dot(transformation, odometry)
            odometry_inv = np.linalg.inv(odometry)
            pose_graph.nodes.append(PoseGraphNode(odometry_inv))
            pose_graph.edges.append(
                    PoseGraphEdge(s, t, transformation,
                    information, True))
        else: # loop closure case
            pose_graph.edges.append(
                    PoseGraphEdge(s, t, transformation,
                    information, True))

Note that the script builds posegraph for fragment. Likewise :ref:`make_fragments_make_a_posegraph` in make_fragments.py, this script adds nodes and edges depending on whether it is odometry case or not. However, optimizing posegraph here corresponds to optimize the geometry of the whole scene, not a fragment. Another difference is that the posegraph is build for the point clouds, not RGBD frames.


Main registration loop
``````````````````````````````````````

The function ``register_point_cloud`` below calls all the functions introduced above.

.. code-block:: python

    def register_point_cloud(path_dataset, ply_file_names,
            registration_type = "color", draw_result = False):
        pose_graph = PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(PoseGraphNode(odometry))
        info = np.identity(6)

        n_files = len(ply_file_names)
        for s in range(n_files):
            for t in range(s + 1, n_files):
                print("reading %s ..." % ply_file_names[s])
                source = read_point_cloud(ply_file_names[s])
                print("reading %s ..." % ply_file_names[t])
                target = read_point_cloud(ply_file_names[t])
                (source_down, source_fpfh) = preprocess_point_cloud(source)
                (target_down, target_fpfh) = preprocess_point_cloud(target)

                (success_global, transformation_init) = \
                        compute_initial_registration(
                        s, t, source_down, target_down,
                        source_fpfh, target_fpfh, path_dataset)
                if not success_global:
                    continue

                (transformation_icp, information_icp) = \
                        local_refinement(source, target,
                        source_down, target_down, transformation_init,
                        registration_type, draw_result)

                update_posegrph_for_scene(s, t,
                        transformation_icp, information_icp,
                        odometry, pose_graph)
                print(pose_graph)

        write_pose_graph(path_dataset + template_global_posegraph, pose_graph)


The workflow of the main function follows:

- Step 1: read two point clouds
- Step 2: ``compute_initial_registration`` using either

    - odometry from fragment
    - feature based registration

- Step 3: ``local_refinement`` using either

    - point-to-plane ICP
    - colored ICP

- Step 4: ``update_posegrph_for_scene``
- Step 5: ``optimize_posegraph_for_scene``

Results
``````````````````````````````````````

The following is messages from posegraph optmization.

.. code-block:: python

    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 52 edges.
    Line process weight : 416.822452
    [Initial     ] residual : 3.560956e+07, lambda : 1.227002e+01
    [Iteration 00] residual : 2.115086e+04, valid edges : 2, time : 0.000 sec.
    [Iteration 01] residual : 2.011877e+04, valid edges : 5, time : 0.000 sec.
    [Iteration 02] residual : 1.838354e+04, valid edges : 8, time : 0.000 sec.
    [Iteration 03] residual : 1.557901e+04, valid edges : 25, time : 0.000 sec.
    :
    [Iteration 21] residual : 5.580001e+03, valid edges : 42, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.019 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 42 edges.
    Line process weight : 404.368527
    [Initial     ] residual : 2.080906e+03, lambda : 8.109836e+01
    [Iteration 00] residual : 2.015805e+03, valid edges : 41, time : 0.000 sec.
    [Iteration 01] residual : 2.002335e+03, valid edges : 41, time : 0.000 sec.
    [Iteration 02] residual : 1.999133e+03, valid edges : 41, time : 0.000 sec.
    [Iteration 03] residual : 1.997591e+03, valid edges : 41, time : 0.000 sec.
    :
    [Iteration 26] residual : 1.988630e+03, valid edges : 39, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.007 sec.
    CompensateReferencePoseGraphNode : reference : 0


The message indicates there is 14 fragments and 52 valid matching pairs between fragments. After 21 iteration, the pose of the fragments are optimized and 42 edges are remained. After pruning invalid edges, it does posegraph optimization again using only valid edges, resulting ignore two edges more.

The overall registration error after optimization is 1.988630e+03 which is reduced from 3.560956e+07.
