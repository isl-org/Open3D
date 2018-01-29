.. _python_binding_tutorial:

Python binding
-------------------------------------

If Open3D is successfully compiled with Python binding, it will create a Python library with name ``py3d``. Typically, you will find a file ``py3d.so`` in ``build/lib`` directory. This tutorial shows how to import ``py3d`` module and print out help information. For trouble shooing, see :ref:`python_binding`.

.. code-block:: python

    # src/Python/Tutorial/Basic/python_binding.py

    import sys
    import numpy as np
    sys.path.append("../..")

    def example_help_function():
        import py3d as py3d
        help(py3d)
        help(py3d.PointCloud)
        help(py3d.read_point_cloud)

    def example_import_function():
        from py3d import read_point_cloud
        pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        print(pcd)

    if __name__ == "__main__":
        example_help_function()
        example_import_function()

This scripts has two functions: ``example_help_function`` and ``example_import_all``
that show very basic usage of Open3D Python module.
In the heading, it uses ``sys.path.append()`` to refer the path where ``py3d.so`` is located.

.. note:: Depending on environment, the name of Python library may not ``py3d.so``. Regardless of the file name, ``import py3d`` should work.

.. _import_py3d_module:

Import py3d module
=====================================

.. code-block:: python

    def example_import_function():
        from py3d import read_point_cloud
        pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        print(pcd)

This imports ``read_point_cloud`` function from ``py3d`` module. It reads a point cloud file and returns an instance of ``PointCloud`` class. ``print(pcd)`` prints brief information of the point cloud:

.. code-block:: sh

    PointCloud with 198835 points.


.. _using_builtin_help_function:

Using built-in help function
=====================================

It is recommended to use Python built-in ``help`` function to get definitions and instructions of Open3D functions and classes. For example,

.. code-block:: python

    def example_help_function():
        import py3d as py3d
        help(py3d)
        help(py3d.PointCloud)
        help(py3d.read_point_cloud)


Browse py3d
``````````````````````````````````````

``help(py3d)`` prints documents of ``py3d`` module.

.. code-block:: sh

    Help on module py3d:

    NAME
        py3d - Python binding of Open3D

    FILE
        /Users/myaccount/Open3D/build/lib/py3d.so

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


Description of a class in py3d
``````````````````````````````````````

``help(py3d.PointCloud)`` provides description of ``PointCloud`` class.

.. code-block:: sh

    Help on class PointCloud in module py3d:

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
     |      __add__(self: py3d.PointCloud, arg0: py3d.PointCloud) -> py3d.PointCloud
     |
    :


Description of a function in py3d
``````````````````````````````````````

``help(py3d.read_point_cloud)`` provides description of input argument and return type of ``read_point_cloud`` function.

.. code-block:: sh

    Help on built-in function read_point_cloud in module py3d:

    read_point_cloud(...)
        read_point_cloud(filename: unicode) -> py3d.PointCloud

        Function to read PointCloud from file
