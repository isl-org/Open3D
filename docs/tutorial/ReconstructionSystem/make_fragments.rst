.. _reconstruction_system_make_fragments:

Make fragments
-------------------------------------

The first step of the scene reconstruction system is to create fragments from short RGBD sequences.

Input arguments
``````````````````````````````````````

.. code-block:: python

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
                description="making fragments from RGBD sequence.")
        parser.add_argument("path_dataset", help="path to the dataset")
        parser.add_argument("-path_intrinsic",
                help="path to the RGBD camera intrinsic")
        args = parser.parse_args()

        # check opencv python package
        with_opencv = initialize_opencv()
        if with_opencv:
            from opencv_pose_estimation import pose_estimation
        process_fragments(args.path_dataset, args.path_intrinsic)

The script runs with ``python make_fragments.py [path]``. ``[path]`` should have subfolders *image* and *depth* to store the color images and depth images respectively. We assume the color images and the depth images are synchronized and registered. The optional argument ``-path_intrinsic`` specifies path to a json file that stores the camera intrinsic matrix (See :ref:`reading_camera_intrinsic` for details). If it is not given, the PrimeSense factory setting is used.


.. _make_fragments_register_rgbd_image_pairs:

Register RGBD image pairs
``````````````````````````````````````

.. code-block:: python

    def register_one_rgbd_pair(s, t, color_files, depth_files,
            intrinsic, with_opencv):
        # read images
        color_s = read_image(color_files[s])
        depth_s = read_image(depth_files[s])
        color_t = read_image(color_files[t])
        depth_t = read_image(depth_files[t])
        source_rgbd_image = create_rgbd_image_from_color_and_depth(color_s, depth_s,
                depth_trunc = 3.0, convert_rgb_to_intensity = True)
        target_rgbd_image = create_rgbd_image_from_color_and_depth(color_t, depth_t,
                depth_trunc = 3.0, convert_rgb_to_intensity = True)

        if abs(s-t) is not 1:
            if with_opencv:
                success_5pt, odo_init = pose_estimation(
                        source_rgbd_image, target_rgbd_image, intrinsic, False)
                if success_5pt:
                    [success, trans, info] = compute_rgbd_odometry(
                            source_rgbd_image, target_rgbd_image, intrinsic,
                            odo_init, RGBDOdometryJacobianFromHybridTerm(),
                            OdometryOption())
                    return [success, trans, info]
            return [False, np.identity(4), np.identity(6)]
        else:
            odo_init = np.identity(4)
            [success, trans, info] = compute_rgbd_odometry(
                    source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
                    RGBDOdometryJacobianFromHybridTerm(), OdometryOption())
            return [success, trans, info]

The function reads a pair of RGBD images and registers the ``source_rgbd_image`` to the ``target_rgbd_image``. Open3D function ``compute_rgbd_odometry`` is called to align the RGBD images. For adjacent RGBD images, an identity matrix is used as initialization. For non-adjacent RGBD images, wide baseline matching is used as an initialization. In particular, function ``pose_estimation`` computes OpenCV ORB feature to match sparse features over wide baseline images, then performs 5-point RANSAC to estimate a rough alignment. It is used as the initialization of ``compute_rgbd_odometry``.


.. _make_fragments_make_a_posegraph:

Multiway registration
``````````````````````````````````````

.. code-block:: python

    def make_posegraph_for_fragment(path_dataset, sid, eid, color_files, depth_files,
            fragment_id, n_fragments, intrinsic, with_opencv):
        set_verbosity_level(VerbosityLevel.Error)
        pose_graph = PoseGraph()
        trans_odometry = np.identity(4)
        pose_graph.nodes.append(PoseGraphNode(trans_odometry))
        for s in range(sid, eid):
            for t in range(s + 1, eid):
                # odometry
                if t == s + 1:
                    print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                            % (fragment_id, n_fragments-1, s, t))
                    [success, trans, info] = register_one_rgbd_pair(
                            s, t, color_files, depth_files, intrinsic, with_opencv)
                    trans_odometry = np.dot(trans, trans_odometry)
                    trans_odometry_inv = np.linalg.inv(trans_odometry)
                    pose_graph.nodes.append(PoseGraphNode(trans_odometry_inv))
                    pose_graph.edges.append(
                            PoseGraphEdge(s-sid, t-sid, trans, info, uncertain = False))

                # keyframe loop closure
                if s % n_keyframes_per_n_frame == 0 \
                        and t % n_keyframes_per_n_frame == 0:
                    print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                            % (fragment_id, n_fragments-1, s, t))
                    [success, trans, info] = register_one_rgbd_pair(
                            s, t, color_files, depth_files, intrinsic, with_opencv)
                    if success:
                        pose_graph.edges.append(
                                PoseGraphEdge(s-sid, t-sid, trans, info, uncertain = True))
        write_pose_graph(path_dataset + template_fragment_posegraph % fragment_id,
                pose_graph)

This script uses the technique demonstrated in :ref:`multiway_registration`. Function ``make_posegraph_for_fragment`` builds a pose graph for multiway registration of all RGBD images in this sequence. Each graph node represents an RGBD image and its pose which transforms the geometry to the global fragment space. For efficiency, only key frames are used.

Once a pose graph is created, multiway registration is performed by calling function ``optimize_posegraph_for_fragment``.

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


    def optimize_posegraph_for_fragment(path_dataset, fragment_id):
        pose_graph_name = path_dataset + template_fragment_posegraph % fragment_id
        pose_graph_optmized_name = path_dataset + \
                template_fragment_posegraph_optimized % fragment_id
        run_posegraph_optimization(pose_graph_name, pose_graph_optmized_name,
                max_correspondence_distance = 0.03)

This function calls ``global_optimization`` to estimate poses of the RGBD images.

.. _make_fragments_make_a_fragment_mesh:

Make a fragment mesh
``````````````````````````````````````

.. code-block:: python

    def integrate_rgb_frames_for_fragment(color_files, depth_files,
            fragment_id, n_fragments, pose_graph_name, intrinsic):
        pose_graph = read_pose_graph(pose_graph_name)
        volume = ScalableTSDFVolume(voxel_length = 3.0 / 512.0,
                sdf_trunc = 0.04, with_color = True)

        for i in range(len(pose_graph.nodes)):
            i_abs = fragment_id * n_frames_per_fragment + i
            print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)."
                    % (fragment_id, n_fragments-1,
                    i_abs, i+1, len(pose_graph.nodes)))
            color = read_image(color_files[i_abs])
            depth = read_image(depth_files[i_abs])
            rgbd = create_rgbd_image_from_color_and_depth(color, depth,
                    depth_trunc = 3.0, convert_rgb_to_intensity = False)
            pose = pose_graph.nodes[i].pose
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def make_mesh_for_fragment(path_dataset, color_files, depth_files,
            fragment_id, n_fragments, intrinsic):
        mesh = integrate_rgb_frames_for_fragment(
                color_files, depth_files, fragment_id, n_fragments,
                path_dataset + template_fragment_posegraph_optimized % fragment_id,
                intrinsic)
        mesh_name = path_dataset + template_fragment_mesh % fragment_id
        write_triangle_mesh(mesh_name, mesh, False, True)

Once the poses are estimates, :ref:`rgbd_integration` is used to reconstruct a colored fragment from each RGBD sequence.

Batch processing
``````````````````````````````````````

.. code-block:: python

    def process_fragments(path_dataset, path_intrinsic):
        if path_intrinsic:
            intrinsic = read_pinhole_camera_intrinsic(path_intrinsic)
        else:
            intrinsic = PinholeCameraIntrinsic.prime_sense_default

        make_folder(path_dataset + folder_fragment)
        [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
        n_files = len(color_files)
        n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))

        for fragment_id in range(n_fragments):
            sid = fragment_id * n_frames_per_fragment
            eid = min(sid + n_frames_per_fragment, n_files)
            make_posegraph_for_fragment(path_dataset, sid, eid, color_files, depth_files,
                    fragment_id, n_fragments, intrinsic, with_opencv)
            optimize_posegraph_for_fragment(path_dataset, fragment_id)
            make_mesh_for_fragment(path_dataset, color_files, depth_files,
                    fragment_id, n_fragments, intrinsic)

The main function calls each individual function explained above.

Results
``````````````````````````````````````

.. code-block:: sh

    Fragment 000 / 013 :: RGBD matching between frame : 0 and 1
    Fragment 000 / 013 :: RGBD matching between frame : 0 and 5
    Fragment 000 / 013 :: RGBD matching between frame : 0 and 10
    Fragment 000 / 013 :: RGBD matching between frame : 0 and 15
    Fragment 000 / 013 :: RGBD matching between frame : 0 and 20
    :
    Fragment 000 / 013 :: RGBD matching between frame : 95 and 96
    Fragment 000 / 013 :: RGBD matching between frame : 96 and 97
    Fragment 000 / 013 :: RGBD matching between frame : 97 and 98
    Fragment 000 / 013 :: RGBD matching between frame : 98 and 99

The following is a log from ``optimize_a_posegraph_for_fragment``.

.. code-block:: sh

    [GlobalOptimizationLM] Optimizing PoseGraph having 100 nodes and 195 edges.
    Line process weight : 389.309502
    [Initial     ] residual : 3.223357e+05, lambda : 1.771814e+02
    [Iteration 00] residual : 1.721845e+04, valid edges : 157, time : 0.022 sec.
    [Iteration 01] residual : 1.350251e+04, valid edges : 168, time : 0.017 sec.
    :
    [Iteration 32] residual : 9.779118e+03, valid edges : 179, time : 0.013 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.519 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 100 nodes and 179 edges.
    Line process weight : 398.292104
    [Initial     ] residual : 5.120047e+03, lambda : 2.565362e+02
    [Iteration 00] residual : 5.064539e+03, valid edges : 179, time : 0.014 sec.
    [Iteration 01] residual : 5.037665e+03, valid edges : 178, time : 0.015 sec.
    :
    [Iteration 11] residual : 5.017307e+03, valid edges : 177, time : 0.013 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.197 sec.
    CompensateReferencePoseGraphNode : reference : 0

The following is a log from ``integrate_rgb_frames_for_fragment``.

.. code-block:: sh

    Fragment 000 / 013 :: integrate rgbd frame 0 (1 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 1 (2 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 2 (3 of 100).
    :
    Fragment 000 / 013 :: integrate rgbd frame 97 (98 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 98 (99 of 100).
    Fragment 000 / 013 :: integrate rgbd frame 99 (100 of 100).

The following images show some of the fragments made by this script.

.. image:: ../../_static/ReconstructionSystem/make_fragments/fragment_0.png
    :width: 325px

.. image:: ../../_static/ReconstructionSystem/make_fragments/fragment_1.png
    :width: 325px

.. image:: ../../_static/ReconstructionSystem/make_fragments/fragment_2.png
    :width: 325px

.. image:: ../../_static/ReconstructionSystem/make_fragments/fragment_3.png
    :width: 325px
