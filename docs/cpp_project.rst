.. _cplusplus_example_project:

Link Open3D in C++ projects
===========================

CMake
-----

We provide two example CMake projects to demonstrate how to use Open3D in your
CMake projects.

* `Find Pre-Installed Open3D Package in CMake <https://github.com/isl-org/open3d-cmake-find-package>`_
  This option can be used if you'd like Open3D build and install Open3D first,
  then link your project to Open3D.
* `Use Open3D as a CMake External Project <https://github.com/isl-org/open3d-cmake-external-project>`_
  This option can be used if you'd like Open3D to build alongside with your
  project.

You may download Open3D library binaries for common platform and build
configurations from GitHub releases. For instructions on how to compile Open3D
from source, checkout :ref:`compilation`.

pkg-config
----------

If you don't use the CMake build system in your project, you can use the simpler
``pkg-config`` tool to get the build settings needed to link it with Open3D.
This is available on Linux and macOS, if you use Open3D shared libraries. Note
that we recommend using ``CMake`` over ``pkg-config``, since the latter cannot
properly account for complex build configurations.

For example, you can equivalently build the `Draw` executable from the above
example project with this command:

.. code:: sh

    export PKG_CONFIG_PATH="$PKG_CONFIG_PATH:<Open3D_install_path>/lib/pkgconfig"
    c++ Draw.cpp -o Draw $(pkg-config --cflags --libs Open3D)

``pkg-config`` reads ``.pc`` files included in the Open3D install and fills in the
required build options. Note that the ``pkg-config --libs`` options must appear
*after* your source files to avoid unrecognized symbol linker errors.
