.. _file_io:

File IO
-------------------------------------

This tutorial shows how basic geometries are read and written by Open3D.

.. literalinclude:: ../../../examples/Python/Basic/file_io.py
   :language: python
   :lineno-start: 5
   :lines: 5-
   :linenos:

.. _io_point_cloud:

Point cloud
=====================================

This script reads and writes a point cloud.

.. literalinclude:: ../../../examples/Python/Basic/file_io.py
   :language: python
   :lineno-start: 11
   :lines: 11-14
   :linenos:

``print()`` function can be used for displaying a summary of ``pcd``. Output message is below:

.. code-block:: sh

    Testing IO for point cloud ...
    PointCloud with 113662 points.


.. _io_mesh:

Mesh
=====================================

This script reads and writes a mesh.

.. literalinclude:: ../../../examples/Python/Basic/file_io.py
   :language: python
   :lineno-start: 16
   :lines: 16-19
   :linenos:

Compared to the data structure of point cloud, mesh has triangles that define surface.

.. code-block:: sh

    Testing IO for meshes ...
    TriangleMesh with 1440 points and 2880 triangles.


.. _io_image:

Image
=====================================

This script reads and writes an image.

.. literalinclude:: ../../../examples/Python/Basic/file_io.py
   :language: python
   :lineno-start: 21
   :lines: 21-24
   :linenos:

Size of image is readily displayed using ``print(img)``.

.. code-block:: sh

    Testing IO for images ...
    Image of size 512x512, with 3 channels.
    Use numpy.asarray to access buffer data.
