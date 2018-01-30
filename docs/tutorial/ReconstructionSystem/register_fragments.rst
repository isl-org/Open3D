.. _reconstruction_system_register_fragments:

Register fragments
-------------------------------------

Once the fragments of the scene are created, the next step is to align them in a global space.

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

This script runs with ``python register_fragments.py [path]``. ``[path]`` should have subfolders *fragments* which stores fragments in .ply files and a pose graph in a .json file.

The main function runs ``register_point_cloud`` and ``optimize_a_posegraph_for_scene``. The first function performs pairwise registration. The second function performs multiway registration.


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

This function downsample point cloud to make a point cloud sparser and regularly distributed. Normals and FPFH feature are precomputed. See :ref:`voxel_downsampling`, :ref:`vertex_normal_estimation`, and :ref:`extract_geometric_feature` for more details.


.. _reconstruction_system_feature_matching:

Pairwise global registration
``````````````````````````````````````

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

This function uses :ref:`feature_matching` for pairwise global registration.


.. _reconstruction_system_compute_initial_registration:

Compute initial registration
``````````````````````````````````````

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


This function computes a rough alignment between two fragments. The rough alignments are used to initialize ICP refinement. If the fragments are neighboring fragments, the rough alignment is determined by an aggregating RGBD odometry obtained from :ref:`reconstruction_system_make_fragments`. Otherwise, ``register_point_cloud_fpfh`` is called to perform global registration. Note that global registration is less reliable according to [Choi2015]_.


Fine-grained registration
``````````````````````````````````````

.. code-block:: python

    def register_point_cloud_icp(source, target,
            init_transformation = np.identity(4)):
        result_icp = registration_icp(source, target, 0.02,
                init_transformation,
                TransformationEstimationPointToPlane())
        print(result_icp)
        information_matrix = get_information_matrix_from_point_clouds(
                source, target, 0.03, result_icp.transformation)
        return (result_icp.transformation, information_matrix)

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
                source, target, 0.03, result_icp.transformation)
        if draw_result:
            draw_registration_result_original_color(source, target,
                    result_icp.transformation)
        return (result_icp.transformation, information_matrix)

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

Two options are given for the fine-grained registration: :ref:`point_to_plane_icp` and :ref:`colored_point_registration`. The latter is recommended since it uses color information to prevent drift. Details see [Park2017]_.


Multiway registration
``````````````````````````````````````

.. code-block:: python

    def update_odometry_posegrph(s, t, transformation, information,
            odometry, pose_graph):

        print("Update PoseGraph")
        if t == s + 1: # odometry case
            odometry = np.dot(transformation, odometry)
            odometry_inv = np.linalg.inv(odometry)
            pose_graph.nodes.append(PoseGraphNode(odometry_inv))
            pose_graph.edges.append(
                    PoseGraphEdge(s, t, transformation,
                    information, uncertain = False))
        else: # loop closure case
            pose_graph.edges.append(
                    PoseGraphEdge(s, t, transformation,
                    information, uncertain = True))
        return (odometry, pose_graph)

This script uses the technique demonstrated in :ref:`multiway_registration`. Function ``update_posegrph_for_scene`` builds a pose graph for multiway registration of all fragments. Each graph node represents a fragments and its pose which transforms the geometry to the global space.

Once a pose graph is built, function ``optimize_posegraph_for_scene`` is called for multiway registration.

.. code-block:: python

    def run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
            max_correspondence_distance):
        # to display messages from global_optimization
        set_verbosity_level(VerbosityLevel.Debug)
        method = GlobalOptimizationLevenbergMarquardt()
        criteria = GlobalOptimizationConvergenceCriteria()
        option = GlobalOptimizationOption(
                max_correspondence_distance = max_correspondence_distance,
                edge_prune_threshold = 0.25,
                reference_node = 0)
        pose_graph = read_pose_graph(pose_graph_name)
        global_optimization(pose_graph, method, criteria, option)
        write_pose_graph(pose_graph_optmized_name, pose_graph)
        set_verbosity_level(VerbosityLevel.Error)

    def optimize_posegraph_for_scene(path_dataset):
        pose_graph_name = path_dataset + template_global_posegraph
        pose_graph_optmized_name = path_dataset + \
                template_global_posegraph_optimized
        run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
                max_correspondence_distance = 0.03)


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

                (odometry, pose_graph) = update_odometry_posegrph(s, t,
                        transformation_icp, information_icp,
                        odometry, pose_graph)
                print(pose_graph)

        write_pose_graph(path_dataset + template_global_posegraph, pose_graph)

The main workflow is: pairwise global registration -> local refinement -> multiway registration.

Results
``````````````````````````````````````

The following is messages from pose graph optimization.

.. code-block:: sh

    PoseGraph with 14 nodes and 52 edges.
    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 52 edges.
    Line process weight : 49.899808
    [Initial     ] residual : 1.307073e+06, lambda : 8.415505e+00
    [Iteration 00] residual : 1.164909e+03, valid edges : 31, time : 0.000 sec.
    [Iteration 01] residual : 1.026223e+03, valid edges : 34, time : 0.000 sec.
    [Iteration 02] residual : 9.263710e+02, valid edges : 41, time : 0.000 sec.
    [Iteration 03] residual : 8.434943e+02, valid edges : 40, time : 0.000 sec.
    :
    [Iteration 22] residual : 8.002788e+02, valid edges : 41, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.006 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 14 nodes and 41 edges.
    Line process weight : 52.121020
    [Initial     ] residual : 3.490871e+02, lambda : 1.198591e+01
    [Iteration 00] residual : 3.409909e+02, valid edges : 40, time : 0.000 sec.
    [Iteration 01] residual : 3.393578e+02, valid edges : 40, time : 0.000 sec.
    [Iteration 02] residual : 3.390909e+02, valid edges : 40, time : 0.000 sec.
    [Iteration 03] residual : 3.390108e+02, valid edges : 40, time : 0.000 sec.
    :
    [Iteration 08] residual : 3.389679e+02, valid edges : 40, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.002 sec.
    CompensateReferencePoseGraphNode : reference : 0


There are 14 fragments and 52 valid matching pairs between fragments. After 23 iteration, 11 edges are detected to be false positive. After they are pruned, pose graph optimization runs again to achieve tight alignment.
