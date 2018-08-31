.. _compilation:

Compilation Options
###################

This page shows advanced options to customize your Open3D build. For quick start, see :ref:`getting_started_compilation`.

.. _python_binding:

Python binding
==============

We use `pybind11 <https://github.com/pybind/pybind11>`_ to build the Python binding.
It tries to automatically detect the installed version of Python and link against that.
When this fails, or when there are multiple versions of Python and it finds the wrong one, delete CMakeCache.txt and then invoke CMake as follows:

.. code-block:: bash

    cmake -DPYTHON_EXECUTABLE:FILEPATH=<path-to-python-executable> ..

.. Note:: Python binding issues can also refer to `pybind11 document page <http://pybind11.readthedocs.io/en/stable/faq.html>`_.

If you do not want Python binding, you may turn off the following compilation options:

- ``BUILD_PYBIND11``
- ``BUILD_PYTHON_MODULE``
- ``BUILD_PYTHON_TUTORIALS``

Dependencies
============

For each dependent library, there is a corresponding CMake build option ``BUILD_<LIBRARY_NAME>``. If the option is ``ON``, the dependent library is forced to be compiled from the source code included in ``3rdparty`` folder. If it is ``OFF``, CMake will try to find system installed libraries and use it. If CMake fails to find the dependent library, it falls back to compiling the library from source code.

.. tip:: On Ubuntu and OSX it is recommended to link Open3D to system installed libraries. The dependencies can be installed via scripts ``util/scripts/install-deps-ubuntu.sh`` and ``util/scripts/install-deps-osx.sh``. On Windows it is recommended to compile everything from source since Windows lacks a package management software.

The following is an example of forcing building dependencies from source code:

.. code-block:: bash

    cmake -DBUILD_EIGEN3=ON  \
          -DBUILD_GLEW=ON    \
          -DBUILD_GLFW=ON    \
          -DBUILD_JPEG=ON    \
          -DBUILD_JSONCPP=ON \
          -DBUILD_PNG=ON     \
          ..

.. note:: Enabling these build options may increase the compilation time.

OpenMP
======

We automatically detect if the C++ compiler supports OpenMP and compile Open3D with it if the compilation option ``WITH_OPENMP`` is ``ON``.
OpenMP can greatly accelerate computation on a multi-core CPU.

The default LLVM compiler on OS X does not support OpenMP.
A workaround is to install a C++ compiler with OpenMP support, such as gcc, then use it to compile Open3D.
For example, starting from a clean build directory, run

.. code-block:: bash

    brew install gcc --without-multilib
    cmake -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 ..
    make -j

.. note:: This workaround has some compatibility issues with the source code of GLFW included in ``3rdparty``.
          Make sure Open3D is linked against GLFW installed on the OS.

Unit testing
============

.. warning:: Work in progress!

    - Unit test coverage: low.
    - Tested on: macOS and Ubuntu.

Unit testing is based on `Google Test <https://github.com/google/googletest>`_.
By default unit tests are turned off. In order to enable them follow the next steps:

    1. Download/Build/Install Google Test.
    2. Set the BUILD_UNIT_TESTS flag to ON.

.. code-block:: bash

    cd util/scripts
    ./install-gtest.sh

    cd <path_to_Open3D>
    mkdir build
    cd build
    cmake -DBUILD_UNIT_TESTS=ON ..
    make -j

In order to perform the unit tests:

.. code-block:: bash

    cd util/scripts
    ./runUnitTests.sh

Documentation
=============

Documentation is written in `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_ and compiled with `sphinx <http://www.sphinx-doc.org/>`_.
From ``docs`` folder, run

.. code-block:: bash

    pip install sphinx sphinx-autobuild sphinx-rtd-theme
    make html

Documentation for C++ API is made with `Doxygen <http://www.stack.nl/~dimitri/doxygen/>`_.
Follow the `Doxygen installation instruction <http://www.stack.nl/~dimitri/doxygen/manual/install.html>`_.
From Open3D ``docs`` folder, run

.. code-block:: bash

    doxygen Doxyfile
