.. _global_optimization:

Global optimization
-------------------------------------

The registration methods so far handles pairwise registration. The goal for that problem is to compute transformation matrix that can align source and target point cloud. Let's consider general cases: there are more than two point clouds, and the goal is to jointly optimize any combination of pairwise alignment altogether. For this purpose, Open3D provides advanced graph optimization method called posegraph optimization [Choi2015]_ [Park2017]_.

Posegraph is special type graph data structure.

- node represents RGBD image or point cloud

	- it has global pose

- edge represents relationship between nodes

	- relative 4x4 transformation between two nodes
	- 6x6 information matrix

Here, the information matrix I is very handy measure to assess the quality of alignment. Note that any rigid motion is expressed by 6-dimensional vectors x: 3 for rotation and the other 3 for translation. This 6-dimensional vector is multiplied with information matrix as a form of x^TIx, where ^T is vector transpose. It is good approximation of point to point distance without explicitly calculating real adjacent point.

The following script builds posegraph and optimizes three point cloud altogether.

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

More details about ``voxel_down_sample`` and ``draw_geometries`` can be found from :ref:`voxel_downsampling` and :ref:`draw_multiple_geometries`.

.. image:: ../../_static/Advanced/global_optimization/initial.png
	:width: 400px


.. _build_posegraph:

Build posegraph
``````````````````````````````````````

The next part of the tutorial script builds posegraph.

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

An instance of posegraph is made by constructor ``PoseGraph()``. Posegraph should have nodes as many as number of the point cloud. For the global poses of the point cloud is represented as odometry, which is accumulated movement. The first node has odometry as identity.

Two nested for-loop in the next is for matching every pair of point clouds. As there are three point clouds, it will match [0-1] [0-2] [1-2] point cloud pairs. The matching is done with :ref:`point_to_plane_icp`. This choice is enough for this example as the initial misalignment can be handled by vanilla ICP. If the initial pose of point clouds are challenging, it is recommended to use :ref:`global_registration`.

The later part of the nested for-loop adds node or edge elements to the posegraph using ``pose_graph.nodes.append`` or ``pose_graph.edges.append``. There are two cases when adding a node or edge.

Let's assume the point clouds are sequentially captured.

- Case 1: edge odometry

	- the script adds nodes
	- if the two point clouds are captured sequentially, it should be more reliable and more overlapping portions.
	- the script adds posegraph edge to be less flexible to adjust.
	- in this case, it marks uncertain as ``False`` when making ``PoseGraphEdge``

- Case 2: loop closure

	- if the two point clouds are matched randomly, this matching result may not be very confident.
	- in this case, it marks uncertain as ``True`` when making ``PoseGraphEdge``

As a result the posegraph will have three nodes (for point cloud 0, 1, 2) and three edges (for [0,1], [0,2], [1,2]).

.. _optimize_posegraph:

Optimize posegraph
``````````````````````````````````````
Posegraph optimization is mathematically minimizing convex cost function.
Check out the next part of the tutorial script.

.. code-block:: python

	print("Optimizing PoseGraph ...")
	global_optimization(pose_graph,
			GlobalOptimizationLevenbergMarquardt(),
			GlobalOptimizationConvergenceCriteria(),
			GlobalOptimizationOption())

``global_optimization`` takes pose_graph and optimize it in-place. Users can specify ``GlobalOptimizationGaussNewton`` or ``GlobalOptimizationLevenbergMarquardt`` as an convex optimization method. Levenberg Marquardt is recommended method. Specific parameters for optimization can be tuned up using ``GlobalOptimizationConvergenceCriteria``. These parameters defines maximum number of iterations and various optimization parameters such as scaling factors. More practical parameters are in ``GlobalOptimizationOption``. It specify how the information matrix is computed with distance threshold to consider adjacent points.

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

The global optimization performs twice on the posegraph. The first iteration optimizes poses for the nodes, and the second iteration filters out unreliable edges and optimizes again.


.. _visualize_optimized_posegraph:

Visualize optimized posegraph
``````````````````````````````````````
To see how well the joint optimization is done, the following script transforms all the point clouds using optimized posegraph.

.. code-block:: python

	print("Transform points and display")
	for point_id in range(n_pcds):
		print(pose_graph.nodes[point_id].pose)
		pcds[point_id].transform(pose_graph.nodes[point_id].pose)
	draw_geometries(pcds)

Note that the output from ``global_optimization`` is its refined poses of nodes. The script apply ``pose_graph.nodes[point_id].pose`` to ``pcds[point_id]``. The visualized point clouds are below

.. image:: ../../_static/Advanced/global_optimization/optimized.png
	:width: 400px


This example shows joint optimization for the point clouds. This idea can be adopted to RGBD image sequence optimization. More examples can be found from ReconstructionSystem.
