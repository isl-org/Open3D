.. _kdtree:

ICP
-------------------------------------

ICP (iterative closest point) is popular method used for aligning two point clouds.
Open3D supports various ICP methods.

.. code-block:: python

	# src/Python/Tutorial/Basic/icp.py

	import sys
	sys.path.append("../..")
	from py3d import *
	from trajectory_io import *

	if __name__ == "__main__":

		traj = read_trajectory("../../TestData/ICP/init.log")
		pcds = []
		threshold = 0.02
		for i in range(3):
			pcds.append(read_point_cloud(
					"../../TestData/ICP/cloud_bin_{:d}.pcd".format(i)))

		for reg in traj:
			target = pcds[reg.metadata[0]]
			source = pcds[reg.metadata[1]]
			trans = reg.pose
			evaluation_init = evaluate_registration(source, target, threshold, trans)
			print(evaluation_init)

			print("Apply point-to-point ICP")
			reg_p2p = registration_icp(source, target, threshold, trans,
					TransformationEstimationPointToPoint())
			print(reg_p2p)
			print("Transformation is:")
			print(reg_p2p.transformation)

			print("Apply point-to-plane ICP")
			reg_p2l = registration_icp(source, target, threshold, trans,
					TransformationEstimationPointToPlane())
			print(reg_p2l)
			print("Transformation is:")
			print(reg_p2l.transformation)
			print("")

		print("")


.. _point_to_point_icp:

Point to point ICP
=====================================
[BESLandMCKAY1992]_



.. _point_to_plane_icp:

Point to plane ICP
=====================================
[CHENandMEDIONI1992]_


.. [BESLandMCKAY1992] Paul J. Besl and Neil D. McKay,
	A Method for Registration of 3-D Shapes, PAMI, 1992.

.. [CHENandMEDIONI1992] Y. Chen and G. G. Medioni,
 	Object modelling by registration of multiple range images.
	Image and Vision Computing, 10(3), 1992
