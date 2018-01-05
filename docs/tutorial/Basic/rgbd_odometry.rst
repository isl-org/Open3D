.. _rgbd_odometry:

RGBD odometry
-------------------------------------

RGBD image sequence is interesting data that can be used for 3D scene reconstruction.
The basic idea for the scene reconstruction begins with estimating small movement between the sequential frames.
The small movement is often called odometry.

Open3D provides two ways to estimate RGBD odometry. The following tutorial shows basic usage of two methods.

.. code-block:: python

    import sys
    sys.path.append("../..")
    from py3d import *
    import numpy as np

    if __name__ == "__main__":
        pinhole_camera_intrinsic = read_pinhole_camera_intrinsic(
                "../../TestData/camera.json")
        print(pinhole_camera_intrinsic.intrinsic_matrix)

        source_color = read_image("../../TestData/RGBD/color/00000.jpg")
        source_depth = read_image("../../TestData/RGBD/depth/00000.png")
        target_color = read_image("../../TestData/RGBD/color/00001.jpg")
        target_depth = read_image("../../TestData/RGBD/depth/00001.png")
        source_rgbd_image = create_rgbd_image_from_color_and_depth(
                source_color, source_depth);
        target_rgbd_image = create_rgbd_image_from_color_and_depth(
                target_color, target_depth);
        target_pcd = create_point_cloud_from_rgbd_image(
                target_rgbd_image, pinhole_camera_intrinsic)

        option = OdometryOption()
        odo_init = np.identity(4)
        print(option)

        [success_color_term, trans_color_term, info] = compute_rgbd_odometry(
                source_rgbd_image, target_rgbd_image,
                pinhole_camera_intrinsic, odo_init,
                RGBDOdometryJacobianFromColorTerm(), option)
        [success_hybrid_term, trans_hybrid_term, info] = compute_rgbd_odometry(
                source_rgbd_image, target_rgbd_image,
                pinhole_camera_intrinsic, odo_init,
                RGBDOdometryJacobianFromHybridTerm(), option)

        if success_color_term:
            print("Using RGB-D Odometry")
            print(trans_color_term)
            source_pcd_color_term = create_point_cloud_from_rgbd_image(
                    source_rgbd_image, pinhole_camera_intrinsic)
            source_pcd_color_term.transform(trans_color_term)
            draw_geometries([target_pcd, source_pcd_color_term])
        if success_hybrid_term:
            print("Using Hybrid RGB-D Odometry")
            print(trans_hybrid_term)
            source_pcd_hybrid_term = create_point_cloud_from_rgbd_image(
                    source_rgbd_image, pinhole_camera_intrinsic)
            source_pcd_hybrid_term.transform(trans_hybrid_term)
            draw_geometries([target_pcd, source_pcd_hybrid_term])


.. _reading_camera_intrinsic:

Reading camera intrinsic
=====================================

Every RGBD camera has unique intrinsic matrix that can express how the 3D point is
projected onto image plane. This intrinsic matrix is an essential element for transforming
RGBD image into point cloud. The following script reads camera intrinsic parameter.

.. code-block:: python

    pinhole_camera_intrinsic = read_pinhole_camera_intrinsic(
            "../../TestData/camera.json")
    print(pinhole_camera_intrinsic.intrinsic_matrix)

This script prints following intrinsic matrix loaded from camera.json

.. code-block:: shell

    [[ 415.69219382    0.          319.5       ]
     [   0.          415.69219382  239.5       ]
     [   0.            0.            1.        ]]


.. _reading_rgbd_image:

Reading RGBD image
=====================================

Reading RGBD image is easy. It is required to read color and depth image independently, and
make RGBD image from the two images. Let's review following script.

.. code-block:: shell

    source_color = read_image("../../TestData/RGBD/color/00000.jpg")
    source_depth = read_image("../../TestData/RGBD/depth/00000.png")
    target_color = read_image("../../TestData/RGBD/color/00001.jpg")
    target_depth = read_image("../../TestData/RGBD/depth/00001.png")
    source_rgbd_image = create_rgbd_image_from_color_and_depth(
            source_color, source_depth)
    target_rgbd_image = create_rgbd_image_from_color_and_depth(
            target_color, target_depth)

The script reads two color and depth image pairs using ``read_image`` and makes
two RGBD image class using ``create_rgbd_image_from_color_and_depth``.
This is basic data format used for RGBD odometry or for transforming 3D point cloud.

.. note:: ``compute_rgbd_odometry`` assumes color and depth image are in the same image domain. To align the two image domain, it is necessary to do intrinsic and extrinsic camera calibration of two cameras. Please refer RGBD camera API to utilize factory calibration parameter, or use image domain alignment functions provided.


.. _compute_odometry:

Compute odometry from RGBD image pair
=====================================

The script calls ``compute_rgbd_odometry`` twice. Let's review code snippet.

.. code-block:: python

    [success, trans_color_term, info] = compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image,
            pinhole_camera_intrinsic, odo_init,
            RGBDOdometryJacobianFromColorTerm(), option)
    [success, trans_hybrid_term, info] = compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image,
            pinhole_camera_intrinsic, odo_init,
            RGBDOdometryJacobianFromHybridTerm(), option)

The only difference is to specify odometry estimation method with ``RGBDOdometryJacobianFromColorTerm()`` or ``RGBDOdometryJacobianFromHybridTerm()``.
The first one computes odometry using idea of [Steinbrucker2011]_. It minimizes photo consistency of aligned images. The corresponding points are detemined by depth image. The second method computes odometry using [Park2017]_. This method has additional cost term that also optimizes geometric alignment.


.. _visualize_rgbd_image:

Visualize RGBD image pair
=====================================

After computing alignment, it is useful to visualize aligned RGBD images. The idea is transform source and target RGBD images into point cloud and visualize together. The following script implements the idea.

.. code-block:: python

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = create_point_cloud_from_rgbd_image(
                source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        draw_geometries([target_pcd, source_pcd_color_term])
    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = create_point_cloud_from_rgbd_image(
                source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        draw_geometries([target_pcd, source_pcd_hybrid_term])

``create_point_cloud_from_rgbd_image`` is useful function that transform RGBD image into point cloud. The source point cloud is transformed using ``.transform()`` with estimated odometry. ``draw_geometries`` display two point clouds by taking a list of point cloud objects ``[target_pcd, source_pcd_color_term]``.

This script will show two windows and transformation matrix

.. image:: ../../_static/Basic/rgbd_odometry/color_term.png
    :width: 400px

.. image:: ../../_static/Basic/rgbd_odometry/hybrid_term.png
    :width: 400px

.. code-block:: shell

    Using RGB-D Odometry
    [[  9.99985131e-01  -2.26255547e-04  -5.44848980e-03  -4.68289761e-04]
     [  1.48026964e-04   9.99896965e-01  -1.43539723e-02   2.88993731e-02]
     [  5.45117608e-03   1.43529524e-02   9.99882132e-01   7.82593526e-04]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]
    Using Hybrid RGB-D Odometry
    [[  9.99994666e-01  -1.00290715e-03  -3.10826763e-03  -3.75410348e-03]
     [  9.64492959e-04   9.99923448e-01  -1.23356675e-02   2.54977516e-02]
     [  3.12040122e-03   1.23326038e-02   9.99919082e-01   1.88139799e-03]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]
