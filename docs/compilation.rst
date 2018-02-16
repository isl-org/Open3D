.. _compilation:

Compilation Options
#######################

This page shows advanced options to customize your Open3D build. For quick start, see :ref:`getting_started_compilation`.

.. _python_binding:

Python binding
=================

We use `pybind11 <https://github.com/pybind/pybind11>`_ to build the Python binding. It tries to automatically detect the installed version of Python and link against that. When this fails, or when there are multiple versions of Python and it finds the wrong one, delete CMakeCache.txt and then invoke CMake as follows:

.. code-block:: bash

    cmake -DPYTHON_EXECUTABLE:FILEPATH=<path-to-python-executable> ../src

.. Note:: Python binding issues can also refer to `pybind11 document page <http://pybind11.readthedocs.io/en/stable/faq.html>`_.

If you do not want Python binding, turn off the compilation options ``Open3D_BUILD_PYTHON_BINDING`` and ``Open3D_BUILD_PYTHON_BINDING_TESTS``.

Dependencies
================

On Windows, all dependent libraries are built from source code. Dependencies are included in ``src/External`` folder.

On Linux and OS X, the default behavior of Open3D is to find packages installed on the OS and link to them. If the required package(s) is not found, it is built from source code. This avoids compilation of dependent libraries as much as possible, and thus significantly reduces compilation time. However, sometimes there can be compatibility issues due to version conflict. In this case, you can manual turn off the compilation option ``Open3D_USE_NATIVE_DEPENDENCY_BUILD`` to force building dependent libraries from source code:

.. code-block:: bash

    cmake -DOpen3D_USE_NATIVE_DEPENDENCY_BUILD=OFF ../src

OpenMP
==========

We automatically detect if the C++ compiler supports OpenMP and compile Open3D with it if the compilation option ``Open3D_USE_OPENMP`` is ``ON``. OpenMP can greatly accelerate computation on a multi-core CPU.

The default LLVM compiler on OS X does not support OpenMP. A workaround is to install a C++ compiler with OpenMP support, such as gcc, then use it to compile Open3D. For example, starting from a clean build directory, run

.. code-block:: bash

    brew install gcc --without-multilib
    cmake -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 ../src
    make -j

.. note:: This workaround has some compatibility issues with the source code of GLFW included in ``src/External``. Make sure Open3D is linked against GLFW installed on the OS.

Documentation
===============

Documentation is written in `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_ and compiled with `sphinx <http://www.sphinx-doc.org/>`_. From ``docs`` folder, run

.. code-block:: bash

    pip install sphinx sphinx-autobuild sphinx-rtd-theme
    make html

Documentation for C++ API is made with `Doxygen <http://www.stack.nl/~dimitri/doxygen/>`_. Follow the `Doxygen installation instruction <http://www.stack.nl/~dimitri/doxygen/manual/install.html>`_. From Open3D root folder, run

.. code-block:: bash

    doxygen Doxyfile
