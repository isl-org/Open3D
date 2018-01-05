.. _rgbd_integration:

RGBD integration
-------------------------------------

Once a RGBD odometry is estimated, the next step is to make beautiful tight mesh from a noisy RGBD sequence.
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
This tutorial script uses `Log file format <http://redwood-data.org/indoor/fileformat.html>`_ to retrieve camera odometry as an example. A log file represents camera odometry has following format.

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

A log file has one line of meta data and 4x4 matrix for each frame. The function ``read_trajectory`` reads a list of camera poses. There is a function ``write_trajectory`` in src/Python/Tutorial/Advanced/trajectory_io.py for writing log files.


.. _tsdf_volume_integration:

TSDF volume integration
``````````````````````````````````````
The following script integrates five RGBD frames into a TSDF volume.

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

The script defines a volume using ``ScalableTSDFVolume``. It sets voxels of TSDF to be in 4.0/512.0m size. Note that ``ScalableTSDFVolume`` does not limited to specific cubic size. ``sdf_trunc = 0.04`` sets truncation value. This indicates TSDF having smaller than 0.04 is ignored for memory efficiency. Integrating depth maps into a TSDF volume is implementation of [Curless1996]_ and [Newcombe2011]_.

``with_color = True`` flag indicate that there are separate TSDF volumes made for integrating the color channels too. In this way extracted mesh from the volume has color. This color integration idea is from [Park2017]_ and can be actively used for :ref:`colored_point_registration`.


.. _extract_a_mesh:

Extract a mesh
``````````````````````````````````````
After integrating few frames, the mesh can be extracted from TSDF volume using marching cubes [LorensenAndCline1987]_. Below script does mesh extraction.

.. code-block:: python

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    draw_geometries([mesh])

The raw mesh extracted from a volume does not have surface normal, so ``compute_vertex_normals`` is applied for computing surface normal.
``draw_geometries([mesh])`` displays the extracted mesh like below. 

.. image:: ../../_static/Advanced/rgbd_integration/integrated.png
    :width: 400px
