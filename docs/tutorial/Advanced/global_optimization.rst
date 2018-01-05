.. _global_optimization:

Global optimization
-------------------------------------

The registration methods introduced so far (:ref:`point_to_point_icp`, :ref:`point_to_plane_icp`, and :ref:`point_to_point_icp`) handles pairwise registration.

Let's consider a general case: there are more than two point clouds, and the goal is to **jointly optimize any combination of pairwise alignment altogether**. For this purpose, Open3D provides advanced graph optimization method called posegraph optimization [Choi2015]_ [Park2017]_.


.. _introduction_to_posegraph:

Introduction to posegraph
``````````````````````````````````````

**Posegraph** is special type graph data structure. It has following elements:

- **node**
    - it represents a single geometry like RGBD image or a point cloud
    - a node has a global pose
    - the global pose is the variable to be optimized

- **edge**
    - it represents relationship between two nodes
    - a edge has relative 4x4 transformation between two nodes
    - a edge also has 6x6 information matrix

The **information matrix** is very important to assess the quality of alignment. Note that any rigid motion is expressed by 6-dimensional vectors: 3-dimension for rotation and the other 3-dimension for translation. By definition of information matrix, a 6-dimensional vector ``x`` is multiplied with information matrix ``I`` as a form of ``x^T*I*x``, where ``^T`` is vector transpose and ``*`` is matrix multiplication.

``x^T*I*x`` is good approximation of sum of point-to-point Euclidean distance. The beauty is that there is no need to explicitly finding adjacent point depending on relative pose ``x``. This makes optimizing posegraph much easier.

The following script builds a posegraph and optimizes it to align three point cloud altogether.

.. code-block:: python

    # src/Python/Tutorial/Advanced/global_optimization.py

    import sys
    sys.path.append("../..")
    from py3d import *
    from trajectory_io import *

    if __name__ == "__main__":

        set_verbosity_level(VerbosityLevel.Debug)
        traj = read_trajectory("../../TestData/ICP/init.log")
        pcds = []
        for i in range(3):
            pcd = read_point_cloud(
                    "../../TestData/ICP/cloud_bin_%d.pcd" % i)
            downpcd = voxel_down_sample(pcd, voxel_size = 0.02)
            pcds.append(downpcd)
        draw_geometries(pcds)

        pose_graph = PoseGraph()
        odometry = np.identity(4)
        pose_graph.nodes.append(PoseGraphNode(odometry))

        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                source = pcds[source_id]
                target = pcds[target_id]

                print("Apply point-to-plane ICP")
                result_icp = registration_icp(source, target, 0.30,
                        np.identity(4),
                        TransformationEstimationPointToPlane())
                transformation_icp = result_icp.transformation
                information_icp = get_information_matrix_from_point_clouds(
                        source, target, 0.30, result_icp.transformation)
                print(transformation_icp)

                print("Build PoseGraph")
                if target_id == source_id + 1: # odometry case
                    odometry = np.dot(transformation_icp, odometry)
                    pose_graph.nodes.append(
                            PoseGraphNode(np.linalg.inv(odometry)))
                    pose_graph.edges.append(
                            PoseGraphEdge(source_id, target_id,
                            transformation_icp, information_icp, False))
                else: # loop closure case
                    pose_graph.edges.append(
                            PoseGraphEdge(source_id, target_id,
                            transformation_icp, information_icp, True))

        print("Optimizing PoseGraph ...")
        global_optimization(pose_graph,
                GlobalOptimizationLevenbergMarquardt(),
                GlobalOptimizationConvergenceCriteria(),
                GlobalOptimizationOption())

        print("Transform points and display")
        for point_id in range(n_pcds):
            print(pose_graph.nodes[point_id].pose)
            pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        draw_geometries(pcds)

The first part of tutorial script reads point clouds, downsample them, and visualize them together.

.. code-block:: python

    set_verbosity_level(VerbosityLevel.Debug)
    traj = read_trajectory("../../TestData/ICP/init.log")
    pcds = []
    for i in range(3):
        pcd = read_point_cloud(
                "../../TestData/ICP/cloud_bin_%d.pcd" % i)
        downpcd = voxel_down_sample(pcd, voxel_size = 0.02)
        pcds.append(downpcd)
    draw_geometries(pcds)

The initial poses for the point clouds are shown below. More details about ``voxel_down_sample`` and ``draw_geometries`` can be found from :ref:`voxel_downsampling` and :ref:`draw_multiple_geometries`.

.. image:: ../../_static/Advanced/global_optimization/initial.png
    :width: 400px


.. _build_a_posegraph:

Build a posegraph
``````````````````````````````````````

The next part of the tutorial script builds a posegraph.

.. code-block:: python

    pose_graph = PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(PoseGraphNode(odometry))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            source = pcds[source_id]
            target = pcds[target_id]

            print("Apply point-to-plane ICP")
            result_icp = registration_icp(source, target, 0.30,
                    np.identity(4),
                    TransformationEstimationPointToPlane())
            transformation_icp = result_icp.transformation
            information_icp = get_information_matrix_from_point_clouds(
                    source, target, 0.30, result_icp.transformation)
            print(transformation_icp)

            print("Build PoseGraph")
            if target_id == source_id + 1: # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                        PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                        PoseGraphEdge(source_id, target_id,
                        transformation_icp, information_icp, False))
            else: # loop closure case
                pose_graph.edges.append(
                        PoseGraphEdge(source_id, target_id,
                        transformation_icp, information_icp, True))

An instance of posegraph is made by constructor ``PoseGraph()``. Nodes and edges of posegraph is expressed as Python list type. The new element can be easily added using ``nodes.append()`` or ``edges.append()``.

Two nested for-loop in the script is for matching every pair of point clouds. As there are three point clouds, it will match [0-1] [0-2] [1-2] point cloud pairs. The matching is done with :ref:`point_to_plane_icp`. This choice is enough for this example because the initial misalignment can be handled by vanilla ICP. If the initial pose of point clouds are challenging, it is recommended to use :ref:`global_registration`.

Posegraph should have nodes as many as number of the point cloud. For an initial estimate of poses in nodes, the script uses accumulated transformation between sequencial geometries. For example, an initial pose for the geometry 2 is ``inv(T_01*T_12)`` where ``T_ij`` is transformation from ``i`` to ``j``. The first node gets the pose as identity. This idea is applied for ``odometry`` in the script.

The later part of the nested for-loop adds nodes or edges. The scripts two cases to be considered.

- Case 1: odometry case

    - this case is valid if two geometry is sequentially captured
    - the script adds nodes
    - the script adds posegraph edge to be less flexible to adjust.
    - for marking flexiblity, it uses ``False`` when making ``PoseGraphEdge``

- Case 2: loop closure

    - if the two point clouds are matched randomly, the two point clouds are not guarantees there is overlapping region. Therefore, this case is less confident.
    - in this case, it marks uncertain as ``True`` when making ``PoseGraphEdge``

As a result, the posegraph will have three nodes (for point cloud 0, 1, 2) and three edges (for [0,1], [0,2], [1,2]). [0,1] and [1,2] is considered as odometry case and marked as confident. [0,2] is not odometry case and marked as not confident.

.. _optimize_a_posegraph:

Optimize a posegraph
``````````````````````````````````````
Posegraph optimization is done with minimizing convex cost function. Open3D provides function ``global_optimization``. Let's check the next part of the tutorial script.

.. code-block:: python

    print("Optimizing PoseGraph ...")
    global_optimization(pose_graph,
            GlobalOptimizationLevenbergMarquardt(),
            GlobalOptimizationConvergenceCriteria(),
            GlobalOptimizationOption())

``global_optimization`` takes ``pose_graph`` and optimizes the graph in-place. Users can specify ``GlobalOptimizationGaussNewton`` or ``GlobalOptimizationLevenbergMarquardt`` as a convex optimization method. Levenberg Marquardt is recommended as it is more effective for the optimization. Specific parameters for optimization can be tuned up using ``GlobalOptimizationConvergenceCriteria``. These parameters defines maximum number of iterations and various optimization parameters such as scaling factors. More practical parameters are in ``GlobalOptimizationOption``. It cab specify how the information matrix is computed with distance threshold.

The script displays following output.

.. code-block:: shell

    Optimizing PoseGraph ...
    [GlobalOptimizationLM] Optimizing PoseGraph having 3 nodes and 3 edges.
    Line process weight : 7.796553
    [Initial     ] residual : 8.789272e+02, lambda : 1.263999e+01
    [Iteration 00] residual : 7.726156e+00, valid edges : 0, time : 0.000 sec.
    [Iteration 01] residual : 7.725927e+00, valid edges : 0, time : 0.000 sec.
    Current_residual - new_residual < 1.000000e-06 * current_residual
    [GlobalOptimizationLM] total time : 0.000 sec.
    [GlobalOptimizationLM] Optimizing PoseGraph having 3 nodes and 2 edges.
    Line process weight : 7.914725
    [Initial     ] residual : 2.184441e-03, lambda : 1.264504e+01
    [Iteration 00] residual : 5.134888e-06, valid edges : 0, time : 0.000 sec.
    [Iteration 01] residual : 6.945283e-09, valid edges : 0, time : 0.000 sec.
    Current_residual < 1.000000e-06
    [GlobalOptimizationLM] total time : 0.000 sec.
    CompensateReferencePoseGraphNode : reference : -1

The global optimization performs twice on the posegraph. The first iteration optimizes poses for the original posegraph, and the second iteration filters runs without unreliable edges.


.. _visualize_optimization:

Visualize optimization
``````````````````````````````````````
To see how well the joint optimization is done, the following script transforms all the point clouds using optimized posegraph.

.. code-block:: python

    print("Transform points and display")
    for point_id in range(n_pcds):
        print(pose_graph.nodes[point_id].pose)
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    draw_geometries(pcds)

The script apply ``pose_graph.nodes[point_id].pose`` to transform ``pcds[point_id]``. The visualized point clouds are below.

.. image:: ../../_static/Advanced/global_optimization/optimized.png
    :width: 400px


This example shows joint optimization for the point clouds. This idea can be adopted to RGBD image sequence optimization too. More examples with RGBD sequence can be found from :ref:`reconstruction_system_make_fragments`.
