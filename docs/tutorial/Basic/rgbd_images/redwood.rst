.. _rgbd_redwood:

Redwood dataset
-------------------------------------
This tutorial reads and visualizes an ``RGBDImage`` from `the Redwood dataset <http://redwood-data.org/>`_ [Choi2015]_.

.. literalinclude:: ../../../../examples/Python/Basic/rgbd_redwood.py
   :language: python
   :lineno-start: 5
   :lines: 5-
   :linenos:


The Redwood format stored depth in a 16-bit single channel image. The integer value represents the depth measurement in millimeters. It is the default format for Open3D to parse depth images.

.. literalinclude:: ../../../../examples/Python/Basic/rgbd_redwood.py
   :language: python
   :lineno-start: 11
   :lines: 11-16
   :linenos:

The default conversion function ``create_rgbd_image_from_color_and_depth`` creates an ``RGBDImage`` from a pair of color and depth image. The color image is converted into a grayscale image, stored in ``float`` ranged in [0, 1]. The depth image is stored in ``float``, representing the depth value in meters. ``print(rgbd_image)`` yields:

.. code-block:: sh

    RGBDImage of size
    Color image : 640x480, with 1 channels.
    Depth image : 640x480, with 1 channels.
    Use numpy.asarray to access buffer data.

The converted images can be rendered as numpy arrays.

.. literalinclude:: ../../../../examples/Python/Basic/rgbd_redwood.py
   :language: python
   :lineno-start: 18
   :lines: 18-24
   :linenos:

Outputs:

.. image:: ../../../_static/Basic/rgbd_images/redwood_rgbd.png
    :width: 400px

The RGBD image can be converted into a point cloud, given a set of camera parameters.

.. literalinclude:: ../../../../examples/Python/Basic/rgbd_redwood.py
   :language: python
   :lineno-start: 26
   :lines: 26-32
   :linenos:

Here we use ``PinholeCameraIntrinsicParameters.PrimeSenseDefault`` as default camera parameter. It has image resolution 640x480, focal length (fx, fy) = (525.0, 525.0), and optical center (cx, cy) = (319.5, 239.5). An identity matrix is used as the default extrinsic parameter. ``pcd.transform`` applies an up-down flip transformation on the point cloud for better visualization purpose. This outputs:

.. image:: ../../../_static/Basic/rgbd_images/redwood_pcd.png
    :width: 400px
