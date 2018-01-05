.. _registration:

Registration
-------------------------------------

ICP (iterative closest point) is the popular method used for aligning two point clouds.
This tutorial demonstrates point-to-point and point-to-plane ICP.

.. code-block:: python

	# src/Python/Tutorial/Basic/icp.py

	def draw_registration_result(source, target, transformation):
		source_temp = copy.deepcopy(source)
		target_temp = copy.deepcopy(target)
		source_temp.paint_uniform_color([1, 0.706, 0])
		target_temp.paint_uniform_color([0, 0.651, 0.929])
		source_temp.transform(transformation)
		draw_geometries([source_temp, target_temp])

	if __name__ == "__main__":
		source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
		target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
		threshold = 0.02
		trans_init = np.asarray(
					[[0.862, 0.011, -0.507,  0.5],
					[-0.139, 0.967, -0.215,  0.7],
					[0.487, 0.255,  0.835, -1.4],
					[0.0, 0.0, 0.0, 1.0]])
		draw_registration_result(source, target, trans_init)
		print("Initial alignment")
		evaluation = evaluate_registration(source, target,
				threshold, trans_init)
		print(evaluation)

		print("Apply point-to-point ICP")
		reg_p2p = registration_icp(source, target, threshold, trans_init,
				TransformationEstimationPointToPoint())
		print(reg_p2p)
		print("Transformation is:")
		print(reg_p2p.transformation)
		print("")
		draw_registration_result(source, target, reg_p2p.transformation)

		print("Apply point-to-plane ICP")
		reg_p2l = registration_icp(source, target, threshold, trans_init,
				TransformationEstimationPointToPlane())
		print(reg_p2l)
		print("Transformation is:")
		print(reg_p2l.transformation)
		print("")
		draw_registration_result(source, target, reg_p2l.transformation)

The tutorial script reads two point clouds (namely source and target),
and align them.
This script has a function ``draw_registration_result``
for visualizing aligned point clouds.


.. _visualize_registration:

Visualize registration
=====================================

.. code-block:: python

	def draw_registration_result(source, target, transformation):
		source_temp = copy.deepcopy(source)
		target_temp = copy.deepcopy(target)
		source_temp.paint_uniform_color([1, 0.706, 0])
		target_temp.paint_uniform_color([0, 0.651, 0.929])
		source_temp.transform(transformation)
		draw_geometries([source_temp, target_temp])

For better visualization, the function paints yellow color for the source
and paints cyan color for the target point cloud.
Note that the function utilizes ``copy.deepcopy`` to make hardcopies of two point clouds.
Without making hardcopies, ``transform`` or ``paint_uniform_color`` method
will transform original point clod, which is not an appropriate for
the visualization function.

.. code-block:: python

	source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
	target = read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
	threshold = 0.02
	trans_init = np.asarray(
				[[0.862, 0.011, -0.507,  0.5],
				[-0.139, 0.967, -0.215,  0.7],
				[0.487, 0.255,  0.835, -1.4],
				[0.0, 0.0, 0.0, 1.0]])
	draw_registration_result(source, target, trans_init)

This script will show the following initial alignment using ``trans_init``.

.. image:: ../../_static/Basic/icp/initial.png
	:width: 400px

``evaluate_registration`` is used for displaying how the alignment is good or bad.

.. code-block:: python

	print("Initial alignment")
	evaluation = evaluate_registration(source, target,
			threshold, trans_init)
	print(evaluation)

This script prints the following:

.. code-block:: shell

	Initial alignment
	RegistrationResult with fitness = 0.174723, inlier_rmse = 0.011771,
	and correspondence_set size of 34741
	Access transformation to get result.

This message indicates 34741 points are overlapped.
Let's align these point clouds and get more number of overlapped points.


.. _point_to_point_icp:

Point-to-point ICP
=====================================
Point to point ICP [BeslAndMcKay1992]_ aligns the point cloud using following idea:

- Step 1: Finding neighboring points between source point cloud and target point cloud.
- Step 2: Compute the rigid transformation that minimizes ||Xs - Xt||_2

	* Xs is a source point
	* Xt is a target point
	* || ||_2 is L2 norm

- Step 3: Transform source point cloud
- Iterate step 1, 2 and 3 until converged, or terminate after few iterations.

This is a script for point-to-point ICP.

.. code-block:: python

	print("Apply point-to-point ICP")
	reg_p2p = registration_icp(source, target, threshold, trans_init,
			TransformationEstimationPointToPoint())
	print(reg_p2p)
	print("Transformation is:")
	print(reg_p2p.transformation)
	print("")
	draw_registration_result(source, target, reg_p2p.transformation)

In this script, ``registration_icp`` takes following arguments: two point clouds,
Euclidean distance threshold for determining neighboring points,
4x4 numpy matrix for initial transformation, and alignment method
``TransformationEstimationPointToPoint``.
Note that the transformation matrix moves the source to align with the target.

The script will show:

.. image:: ../../_static/Basic/icp/point_to_point.png
	:width: 400px

with following message

.. code-block:: shell

	Apply point-to-point ICP
	RegistrationResult with fitness = 0.372450, inlier_rmse = 0.007760,
	and correspondence_set size of 74056
	Access transformation to get result.
	Transformation is:
	[[ 0.83924644  0.01006041 -0.54390867  0.64639961]
	 [-0.15102344  0.96521988 -0.21491604  0.75166079]
	 [ 0.52191123  0.2616952   0.81146378 -1.50303533]
	 [ 0.          0.          0.          1.        ]]

It produces 74056 overlapping points, but it is not converged well.
The cure is to increase the number of ICP iterations.

Changing ICP parameters
``````````````````````````````````````
To change the number of ICP iteration, it is required to define ``ICPConvergenceCriteria``.
The following script specifies 2000 iterations for point-to-point ICP.
Without specified, the default parameter for ICP iteration is 1000.

.. code-block:: python

	reg_p2p = registration_icp(source, target, threshold, trans_init,
			TransformationEstimationPointToPoint(),
			ICPConvergenceCriteria(max_iteration = 2000))

The alignment results are below.

.. image:: ../../_static/Basic/icp/point_to_point_2000.png
	:width: 400px

.. code-block:: shell

	Apply point-to-point ICP
	RegistrationResult with fitness = 0.621123, inlier_rmse = 0.006583,
	and correspondence_set size of 123501
	Access transformation to get result.
	Transformation is:
	[[ 0.84024592  0.00687676 -0.54241281  0.6463702 ]
	 [-0.14819104  0.96517833 -0.21706206  0.81180074]
	 [ 0.52111439  0.26195134  0.81189372 -1.48346821]
	 [ 0.          0.          0.          1.        ]]

This script produces better alignment than the result from 1000 iterations.
The number of overlapping points are 123501. It was 74056 with 1000 ICP iterations.


.. _point_to_plane_icp:

Point-to-plane ICP
=====================================

Point-to-plane ICP [ChenAndMedioni1992]_ is strong complement of point-to-plane ICP.
It minimizes different cost function.

- Step 1: Finding neighboring points between source point cloud and target point cloud.
- Step 2: Compute the rigid transformation that minimizes || dot(Xs - Xt, Nt) ||_2

	* Xs is a source point
	* Xt is a target point
	* Nt is normal direction of a target point
	* function dot() is a vector dot product operator and || ||_2 is L2 norm

- Step 3: Transform source point cloud
- Iterate step 1, 2 and 3 until converged, or terminate after few iterations.

The following script uses ``registration_icp`` that is the same as point-to-point example.
The difference is to specifying ``TransformationEstimationPointToPlane()``.

.. code-block:: python

	print("Apply point-to-plane ICP")
	reg_p2l = registration_icp(source, target, threshold, trans_init,
			TransformationEstimationPointToPlane())
	print(reg_p2l)
	print("Transformation is:")
	print(reg_p2l.transformation)
	print("")
	draw_registration_result(source, target, reg_p2l.transformation)

In general point-to-plane shows better convergence behavior than the point-to-point ICP.

.. note:: Note that point-to-plane utilizes point normal, hence an input point cloud should have point normal. To compute normal from point cloud, see :ref:`vertex_normal_estimation`.

Finally, the script shows the following alignment results.

.. image:: ../../_static/Basic/icp/point_to_plane.png
	:width: 400px

.. code-block:: shell

	Apply point-to-plane ICP
	RegistrationResult with fitness = 0.620972, inlier_rmse = 0.006581,
	and correspondence_set size of 123471
	Access transformation to get result.
	Transformation is:
	[[ 0.84023324  0.00618369 -0.54244126  0.64720943]
	 [-0.14752342  0.96523919 -0.21724508  0.81018928]
	 [ 0.52132423  0.26174429  0.81182576 -1.48366001]
	 [ 0.          0.          0.          1.        ]]
