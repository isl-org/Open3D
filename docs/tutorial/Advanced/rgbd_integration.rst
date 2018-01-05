.. _rgbd_integration:

RGBD integration
-------------------------------------

Once the accurate RGBD odometry is computed, the next step is to make water tight mesh from RGBD sequence.
This tutorial shows how to integrate rgbd frames into thresholded signed distance (TSDF) volume and extract a mesh from the volume.

.. code-block:: python

	# src/Python/Tutorial/Advanced/rgbd_integration.py

	import sys
	sys.path.append("../..")
	from py3d import *
	from trajectory_io import *
	import numpy as np

	if __name__ == "__main__":
		camera_poses = read_trajectory("../../TestData/RGBD/odometry.log")
		volume = ScalableTSDFVolume(voxel_length = 4.0 / 512.0,
				sdf_trunc = 0.04, with_color = True)

		for i in range(len(camera_poses)):
			print("Integrate {:d}-th image into the volume.".format(i))
			color = read_image("../../TestData/RGBD/color/{:05d}.jpg".format(i))
			depth = read_image("../../TestData/RGBD/depth/{:05d}.png".format(i))
			rgbd = create_rgbd_image_from_color_and_depth(color, depth,
					depth_trunc = 4.0, convert_rgb_to_intensity = False)
			volume.integrate(rgbd, PinholeCameraIntrinsic.prime_sense_default,
					np.linalg.inv(camera_poses[i].pose))

		print("Extract a triangle mesh from the volume and visualize it.")
		mesh = volume.extract_triangle_mesh()
		mesh.compute_vertex_normals()
		draw_geometries([mesh])


.. _using_log_file_format:

Using log file format
``````````````````````````````````````
This script uses `Log file format <http://redwood-data.org/indoor/fileformat.html>`_ to retrieve camera odometry. If log file represents global registration of many point clouds It consists of source ID, target ID, total number of point clouds, and 4x4 transformation matrix. If the log file represents camera odometry, it the meta data field represents odometry ID twice and number of frames to distinguish it is for odometry.

The example data below is odometry.

.. code-block:: shell

	# src/test/TestData/RGBD/odometry.log
	0   0   1
	1   0   0   2
	0   1   0   2
	0   0   1 -0.3
	0   0   0   1
	1   1   2
	0.999988  3.08668e-005  0.0049181  1.99962
	-8.84184e-005  0.999932  0.0117022  1.97704
	-0.0049174  -0.0117024  0.999919  -0.300486
	0  0  0  1
	:

The function ``read_trajectory`` reads a list of camera poses. There is a function ``write_trajectory`` in ``src/Python/Tutorial/Advanced/trajectory_io.py``.


.. _tsdf_volume_integration:

TSDF volume integration
``````````````````````````````````````
The following script integrates five RGBD frames into TSDF volume.

.. code-block:: python

	volume = ScalableTSDFVolume(voxel_length = 4.0 / 512.0,
			sdf_trunc = 0.04, with_color = True)

	for i in range(len(camera_poses)):
		print("Integrate {:d}-th image into the volume.".format(i))
		color = read_image("../../TestData/RGBD/color/{:05d}.jpg".format(i))
		depth = read_image("../../TestData/RGBD/depth/{:05d}.png".format(i))
		rgbd = create_rgbd_image_from_color_and_depth(color, depth,
				depth_trunc = 4.0, convert_rgb_to_intensity = False)
		volume.integrate(rgbd, PinholeCameraIntrinsic.prime_sense_default,
				np.linalg.inv(camera_poses[i].pose))

The script first defines a volume using ``ScalableTSDFVolume``. It sets voxels of TSDF to be 4.0/512.0m size. ``sdf_trunc = 0.04`` sets truncation value. This indicates volume having smaller than 0.04 is ignored for memory efficiency. Integrating depth maps into TSDF volume is implementation of [Curless1996]_ and [Newcombe2011]_.

``with_color = True`` flag indicate that there are separate TSDF volumes will be used for the color channels. This flag will make colored mesh once it is extracted from TSDF volume. This color integration idea is from [Park2017]_ and actively used for :ref:`colored_point_registration`.


.. _extract_a_mesh:

Extract a mesh
``````````````````````````````````````
After integrating few frames, the mesh can be extracted from TSDF volume using marching cubes [LorensenAndCline1987]_. Below script does mesh extraction.

.. code-block:: python

	print("Extract a triangle mesh from the volume and visualize it.")
	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	draw_geometries([mesh])

The extracted mesh does not have surface normal, so ``compute_vertex_normals`` is applied for computing surface normal.
``draw_geometries([mesh])`` displays extracted mesh like below:

.. image:: ../../_static/Advanced/rgbd_integration/integrated.png
	:width: 400px
