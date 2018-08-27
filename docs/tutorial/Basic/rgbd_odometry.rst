.. _rgbd_odometry:

RGBD odometry
-------------------------------------

An RGBD odometry finds the camera movement between two consecutive RGBD image pairs. The input are two instances of ``RGBDImage``. The output is the motion in the form of a rigid body transformation. Open3D has implemented two RGBD odometries: [Steinbrucker2011]_ and [Park2017]_.

.. literalinclude:: ../../../examples/Python/Basic/rgbd_odometry.py
   :language: python
   :lineno-start: 5
   :lines: 5-
   :linenos:


.. _reading_camera_intrinsic:

Read camera intrinsic
=====================================

We first read the camera intrinsic matrix from a json file.

.. literalinclude:: ../../../examples/Python/Basic/rgbd_odometry.py
   :language: python
   :lineno-start: 11
   :lines: 11-13
   :linenos:

This yields:

.. code-block:: sh

    [[ 525.     0.   319.5]
     [   0.   525.   239.5]
     [   0.     0.     1. ]]


.. Note:: Lots of small data structures in Open3D can be read from / written into ``json`` files. This includes camera intrinsics, camera trajectory, pose graph, etc.

.. _reading_rgbd_image:

Read RGBD image
=====================================

.. literalinclude:: ../../../examples/Python/Basic/rgbd_odometry.py
   :language: python
   :lineno-start: 15
   :lines: 15-24
   :linenos:

This code block reads two pairs of RGBD images in the Redwood format. We refer to :ref:`rgbd_redwood` for a comprehensive explanation.

.. note:: Open3D assumes the color image and depth image are synchronized and registered in the same coordinate frame. This can usually be done by turning on both the synchronization and registration features in the RGBD camera settings.

.. _compute_odometry:

Compute odometry from two RGBD image pairs
==================================================

.. literalinclude:: ../../../examples/Python/Basic/rgbd_odometry.py
   :language: python
   :lineno-start: 30
   :lines: 30-37
   :linenos:

This code block calls two different RGBD odometry methods. The first one is [Steinbrucker2011]_. It minimizes photo consistency of aligned images. The second one is [Park2017]_. In addition to photo consistency, it implements constraint for geometry. Both functions run in similar speed. But [Park2017]_ is more accurate in our test on benchmark datasets. It is recommended.

Several parameters in ``OdometryOption()``:

* ``minimum_correspondence_ratio`` : After alignment, measure the overlapping ratio of two RGBD images. If overlapping region of two RGBD image is smaller than specified ratio, the odometry module regards that this is a failure case.
* ``max_depth_diff`` : In depth image domain, if two aligned pixels have a depth difference less than specified value, they are considered as a correspondence. Larger value induce more aggressive search, but it is prone to unstable result.
* ``min_depth`` and ``max_depth`` : Pixels that has smaller or larger than specified depth values are ignored.

.. _visualize_rgbd_image:

Visualize RGBD image pairs
=====================================

.. literalinclude:: ../../../examples/Python/Basic/rgbd_odometry.py
   :language: python
   :lineno-start: 39
   :lines: 39-52
   :linenos:

The RGBD image pairs are converted into point clouds and rendered together. Note that the point cloud representing the first (source) RGBD image is transformed with the transformation estimated by the odometry. After this transformation, both point clouds are aligned.

Outputs:

.. code-block:: sh

    Using RGB-D Odometry
    [[  9.99985131e-01  -2.26255547e-04  -5.44848980e-03  -4.68289761e-04]
     [  1.48026964e-04   9.99896965e-01  -1.43539723e-02   2.88993731e-02]
     [  5.45117608e-03   1.43529524e-02   9.99882132e-01   7.82593526e-04]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]

.. image:: ../../_static/Basic/rgbd_odometry/color_term.png
    :width: 400px

.. code-block:: sh

    Using Hybrid RGB-D Odometry
    [[  9.99994666e-01  -1.00290715e-03  -3.10826763e-03  -3.75410348e-03]
     [  9.64492959e-04   9.99923448e-01  -1.23356675e-02   2.54977516e-02]
     [  3.12040122e-03   1.23326038e-02   9.99919082e-01   1.88139799e-03]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.00000000e+00]]

.. image:: ../../_static/Basic/rgbd_odometry/hybrid_term.png
    :width: 400px
