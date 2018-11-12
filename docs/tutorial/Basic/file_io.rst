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

By default, Open3D tries to infer point cloud file type by extension. Below is
a list of supported point cloud file types.

========== =======================================================================================
Format     Description
========== =======================================================================================
``xyz``    Each line contains ``[x, y, z]``, where ``x, y, z`` are the 3D coordinates
``xyzn``   Each line contains ``[x, y, z, nx, ny, nz]``, where ``nx, ny, nz``
           are the normals
``xyzrgb`` Each line contains ``[x, y, z, r, g, b]``,
           where ``r, g, b`` are in floats of range ``[0, 1]``
``pts``    | The first line is an integer representing the number of points
           | Each subsequent line contains ``[x, y, z, i, r, g, b]``,
             where ``r, g, b`` are in ```uint8```
``ply``    See `Polygon File Format <http://paulbourke.net/dataformats/ply>`_,
           the ``ply`` file can contain both point cloud and mesh
``pcd``    See `Point Cloud Data <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`_
========== =======================================================================================

It's also possible to specify the file type explicitly. In this case, the file
extension will be ignored.

.. code-block:: python

    pcd = read_point_cloud("my_points.txt", format='xyz')


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
