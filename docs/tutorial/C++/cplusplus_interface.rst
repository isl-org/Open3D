.. _cplusplus_interface_tutorial:

C++ interface
-------------

This page explains how to create a CMake based C++ project using the Open3D C++ interface.

.. note:: For help on the C++ interfaces, refer to examples in [examples/Cpp/Test], [examples/Cpp/Experimental] and [src/Tools] folder and check `Doxygen document for C++ API <http://open3d.org/cppapi>`_.

.. _install_open3d_from_source:

Install open3d from source
==========================

For installing from source see :ref:`getting_started_compilation`.

.. _create_cplusplus_project:

Create C++ project
==================

.. warning:: The following is supported at this time only on Ubuntu with all dependencies installed.

Let's create a basic C++ project based on CMake and Open3D installed libraries and headers.

1. Get the code from :download:`TestVisualizer.cpp <../../_static/C++/TestVisualizer.cpp>`
2. Get the CMake config file from :download:`CMakeLists.txt <../../_static/C++/CMakeLists.txt>`
3. Build the project using the following commands:

.. code-block:: bash

    mkdir -p build
    cd build
    cmake ..
    make -j

Highlights
``````````

The following fragment from ``CMakeLists.txt`` shows how to specify hints to CMake when looking for the Open3D installation.
This technique is required when installing Open3D to a user location rather than to a system wide location.

.. literalinclude:: ../../_static/C++/CMakeLists.txt
   :language: cmake
   :lineno-start: 5
   :lines: 5
   :linenos:

This section of the ``CMakeLists.txt`` specifies the installed Open3D include directories, libraries and library directories.

.. literalinclude:: ../../_static/C++/CMakeLists.txt
   :language: cmake
   :lineno-start: 21
   :lines: 21-41
   :linenos:
