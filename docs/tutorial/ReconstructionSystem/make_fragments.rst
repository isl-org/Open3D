.. _reconstruction_system_make_fragments:

Make fragments
-------------------------------------

The first step for scene recontruction from RGBD frames is making fragments. Fragments are small piece of the scene, it is necessary as the RGBD odometry is error prone for the large scene.

This tutorial reviews `src/Python/Tutorial/ReconstructionSystem/make_fragments.py <../../../../../src/Python/Tutorial/ReconstructionSystem/make_fragments.py>`_ function by function. The tutorial script makes fragments from RGBD image sequence.


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

The script runs with ``python make_fragments.py [path]``. [path] should have subfolders *image* and *depth* in which frames are synchronized and aligned. The optional argument ``-path_intrinsic`` specifies path to json file that has a camera intrinsic matrix. An example about reading RGBD camera intrinsic is found from :ref:`reading_camera_intrinsic`.

The next part of the main function loads OpenCV module if available. The script utilizes OpenCV for feature extraction, matching, and 5-point pose estimation algorithm. OpenCV module is optional requirement, but it is helpful to match two RGBD images if the baseline is wide.

.. _make_fragments_register_rgbd_image_pairs:

Register RGBD image pairs
``````````````````````````````````````

.. code-block:: python

    def process_one_rgbd_pair(s, t, color_files, depth_files,
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

The function reads source and target RGBD frames and makes two RGBD image instance ``source_rgbd_image`` and ``target_rgbd_image``.

``s`` and ``t`` in function argument indicate RGBD frame ids of source and target. Based on two cases, this function matches RGBD frames in two different ways.

- if s and t are not adjacent (abs(s-t) is not 1), it does *wide baseline matching*
- if s and t are adjacent, it performs *RGBD odometry*

*Wide baseline matching* calls ``pose_estimation``. This function uses OpenCV ORB feature to match sparse features over wide baseline images. With matched features, ``pose_estimation`` performs 5-point RANSAC to reliably estimate rough motion between frames. The estimated pose ``odo_init`` is fed into ``compute_rgbd_odometry`` for fine-grained alignment. In the case of *RGBD odometry*, identity matrix is used as an initial pose.

Function ``process_one_rgbd_pair`` returns whether the matching of two RGBD frame ``s`` and ``t`` is successful, estimated 4x4 transformation matrix, and 6x6 information matrix. Information matrix is computed for posegraph optimization for joint alignment of multiple frames. Refer :ref:`global_optimization` to see how information matrix is used.


.. _make_fragments_make_a_posegraph:

Make a posegraph
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
                    [success, trans, info] = process_one_rgbd_pair(
                            s, t, color_files, depth_files, intrinsic, with_opencv)
                    trans_odometry = np.dot(trans, trans_odometry)
                    trans_odometry_inv = np.linalg.inv(trans_odometry)
                    pose_graph.nodes.append(PoseGraphNode(trans_odometry_inv))
                    pose_graph.edges.append(
                            PoseGraphEdge(s-sid, t-sid, trans, info, False))

                # keyframe loop closure
                if s % n_keyframes_per_n_frame == 0 \
                        and t % n_keyframes_per_n_frame == 0:
                    print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
                            % (fragment_id, n_fragments-1, s, t))
                    [success, trans, info] = process_one_rgbd_pair(
                            s, t, color_files, depth_files, intrinsic, with_opencv)
                    if success:
                        pose_graph.edges.append(
                                PoseGraphEdge(s-sid, t-sid, trans, info, True))
        write_pose_graph(path_dataset + template_fragment_posegraph % fragment_id,
                pose_graph)

Function ``make_posegraph_for_fragment`` builds a ``pose_graph`` of RGBD image matchings. It adds nodes and edges. Posegraph node holds absolute camera odometry, and posegraph edge holds pairwise alignment information including source and target node ID, transformation matrix and information matrix. This information is obtained from ``process_one_rgbd_pair``.

To simplify, this function is based on following idea

.. code-block:: shell

    for s in range(sid, eid):
        for t in range(s + 1, eid):

            if t is adjacent frame of s:
                process_one_rgbd_pair (with odometry matching mode)
                add posegraph node using camera odometry
                add posegraph edge of matching s and t

            if s and t is keyframe:
                process_one_rgbd_pair (with wide baseline matching mode)
                add posegraph edge of matching s and t

    write posegraph

The principle idea for building posegraph is the same as :ref:`global_optimization`, but it does not match every combinations of RGBD frames. Instead this script matches keyframes to save computation time.


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

Once the posegraph is made by ``make_posegraph_for_fragment``, this function is used for integrating RGBD frames into TSDF volume. The basic idea is the same as :ref:`rgbd_integration`. It defines 3x3x3 cubic with 512x512x512 resolution with color. The function makes a RGBD frame ``rgbd``, retrieve estimated camera pose ``pose``. It is integrated into the TSDF volume ``volume.integrate()``. The colored mesh is extracted by ``volume.extract_triangle_mesh``.

This function is called by following script.

.. code-block:: python

    def make_mesh_for_fragment(path_dataset, color_files, depth_files,
            fragment_id, n_fragments, intrinsic):
        mesh = integrate_rgb_frames_for_fragment(
                color_files, depth_files, fragment_id, n_fragments,
                path_dataset + template_fragment_posegraph_optimized % fragment_id,
                intrinsic)
        mesh_name = path_dataset + template_fragment_mesh % fragment_id
        write_triangle_mesh(mesh_name, mesh, False, True)

Previous function ``integrate_rgb_frames_for_fragment`` is called by this function. This function saves mesh file into dataset folder.


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

The functions explained above is called by this main function. The first part of this function reads RGBD camera intrinsic using ``read_pinhole_camera_intrinsic`` if specified by user, and get the RGBD image sequence file list using ``get_rgbd_file_lists``. The number of fragments is computed using the number of RGBD frames. For example, if there is a 1650 frames, this function will make 17 fragments as one fragment is made from 100 frames.

The next for-loop calls function ``make_posegraph_for_fragment``, ``optimize_posegraph_for_fragment``, and ``make_mesh_for_fragment``.


Results
``````````````````````````````````````

For each fragment, this is a printed message from :ref:`make_fragments_register_rgbd_image_pairs`.

.. code-block:: shell

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

The following is a log from ``optimize_posegraph_for_fragment``.

.. code-block:: shell

    [GlobalOptimizationLM] Optimizing PoseGraph having 100 nodes and 196 edges.
    Line process weight : 1209.438798
    [Initial     ] residual : 2.609760e+05, lambda : 2.044341e+02
    [Iteration 00] residual : 3.786013e+04, valid edges : 78, time : 0.016 sec.
    [Iteration 01] residual : 2.206913e+04, valid edges : 85, time : 0.015 sec.
    :
    [Iteration 14] residual : 1.779927e+04, valid edges : 88, time : 0.013 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.225 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 100 nodes and 187 edges.
    Line process weight : 1230.270792
    [Initial     ] residual : 1.052490e+04, lambda : 2.805398e+02
    [Iteration 00] residual : 1.043319e+04, valid edges : 88, time : 0.013 sec.
    [Iteration 01] residual : 1.041026e+04, valid edges : 88, time : 0.014 sec.
    :
    [Iteration 05] residual : 1.040701e+04, valid edges : 88, time : 0.013 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.089 sec.
    CompensateReferencePoseGraphNode : reference : 0

The following is a log from :ref:`make_fragments_make_a_fragment_mesh`.

.. code-block:: shell

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
