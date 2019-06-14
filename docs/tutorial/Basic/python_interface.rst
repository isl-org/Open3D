.. _python_interface_tutorial:

Python interface
----------------

For the C++ interface see :ref:`cplusplus_interface_tutorial`.


Install open3d Python package
=============================

For installing Open3D Python package, see :ref:`install_open3d_python`.


Install open3d from source
==========================

For installing from source, see :ref:`compilation`.

.. _import_open3d_module:

Import open3d module
====================

This tutorial shows how to import ``open3d`` module and print out help information.
For trouble shooting, see :ref:`compilation_ubuntu_python_binding`.

.. literalinclude:: ../../../examples/Python/Basic/python_binding.py
   :language: python
   :lineno-start: 5
   :lines: 5-
   :linenos:

This scripts has two functions: ``example_help_function`` and ``example_import_all``
that show very basic usage of Open3D Python module.

.. note:: Depending on environment, the name of Python library may not be ``open3d.so``. Regardless of the file name, ``import open3d`` should work.

.. literalinclude:: ../../../examples/Python/Basic/python_binding.py
   :language: python
   :lineno-start: 10
   :lines: 10-12
   :linenos:

This imports ``read_point_cloud`` function from ``open3d`` module. It reads a point cloud file and returns an instance of ``PointCloud`` class. ``print(pcd)`` prints brief information of the point cloud:

.. code-block:: sh

    PointCloud with 198835 points.


.. _using_builtin_help_function:

Using built-in help function
````````````````````````````

It is recommended to use Python built-in ``help`` function to get definitions and instructions of Open3D functions and classes. For example,

.. literalinclude:: ../../../examples/Python/Basic/python_binding.py
   :language: python
   :lineno-start: 15
   :lines: 15-18
   :linenos:


Browse open3d
`````````````

``help(open3d)`` prints documents of ``open3d`` module.

.. code-block:: sh

    Help on module open3d:

    NAME
        open3d - Python binding of Open3D

    FILE
        /Users/myaccount/Open3D/build/lib/open3d.so

    CLASSES
        __builtin__.object
            CorrespondenceChecker
                CorrespondenceCheckerBasedOnDistance
                CorrespondenceCheckerBasedOnEdgeLength
                CorrespondenceCheckerBasedOnNormal
            DoubleVector
            Feature
            Geometry
                Geometry2D
                    Image
                Geometry3D
                    PointCloud
                    TriangleMesh
    :


Description of a class in open3d
````````````````````````````````

``help(open3d.PointCloud)`` provides description of ``PointCloud`` class.

.. code-block:: sh

    Help on class PointCloud in module open3d:

    class PointCloud(Geometry3D)
     |  Method resolution order:
     |      PointCloud
     |      Geometry3D
     |      Geometry
     |      __builtin__.object
     |
     |  Methods defined here:
     |
     |  __add__(...)
     |      __add__(self: open3d.PointCloud, arg0: open3d.PointCloud) -> open3d.PointCloud
     |
    :


Description of a function in open3d
```````````````````````````````````

``help(open3d.read_point_cloud)`` provides description of input argument and return type of ``read_point_cloud`` function.

.. code-block:: sh

    Help on built-in function read_point_cloud in module open3d:

    read_point_cloud(...)
        read_point_cloud(filename: unicode) -> open3d.PointCloud

        Function to read PointCloud from file
