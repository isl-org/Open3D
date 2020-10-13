.. _compilation:

Build from source
=====================

You may want to build Open3D from source if you are developing Open3D, want to
get the latest features in the ``master`` branch, or if the OS or Python
versions are not covered by Open3D's pre-built Python packages from PyPI and
Conda.

.. _compiler_version:

System requirements
-------------------

* Ubuntu 18.04+: GCC 5+, Clang 7+
* macOS 10.14+: XCode 8.0+
* Windows 10 (64-bit): Visual Studio 2019+
* CMake 3.12+

  * Ubuntu:

    * Install with ``apt-get``: see `official APT repository <https://apt.kitware.com/>`_
    * Install with ``snap``: ``sudo snap install cmake --classic``
    * Install with ``pip`` (run inside a Python virtual environment): ``pip install cmake``

  * macOS:

    * Install with Homebrew: ``brew install cmake``
    * Download from: `CMake download page <https://cmake.org/download/>`_

  * Windows:

    * Download from: `CMake download page <https://cmake.org/download/>`_

Cloning Open3D
--------------

Make sure to use the ``--recursive`` flag when cloning Open3D.

.. code-block:: bash

    git clone --recursive https://github.com/intel-isl/Open3D

.. _compilation_ubuntu:

Ubuntu
------

.. _compilation_ubuntu_dependencies:

1. Install dependencies
```````````````````````

.. code-block:: bash

    util/install_deps_ubuntu.sh

.. _compilation_ubuntu_python_binding:

2. Setup Python binding environments
````````````````````````````````````

Ensure that the default Python in the current ``PATH`` is the desired one.
Specifically,

- For pip virtualenv, activate it by ``source path_to_my_env/bin/activate``.
- For Conda virtualenv, activate it by ``conda activate my_env``.
- For the system's default Python (note: ``sudo`` may be required for installing
  Python packages), no action is required.

Finally, check

.. code-block:: bash

    which python
    python -V

Alternatively, you can also set CMake flag ``-DPYTHON_EXECUTABLE=/path/to/python``
to specify the python executable.

If Python binding is not needed, it can be turned off by setting
``-DBUILD_PYTHON_MODULE=OFF``.

.. _compilation_ubuntu_config:

3. Config
`````````
.. code-block:: bash

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=<open3d_install_directory> ..

The ``CMAKE_INSTALL_PREFIX`` argument is optional and can be used to install
Open3D to a user location. In the absence of this argument Open3D will be
installed to a system location (sudo required). For more customizations of the
build, please see :ref:`compilation_options`.

.. _compilation_ubuntu_build:

4. Build
````````

.. code-block:: bash

    # On Ubuntu
    make -j$(nproc)

    # On macOS
    make -j$(sysctl -n hw.physicalcpu)

.. _compilation_ubuntu_install:

5. Install
``````````

5.1 Install Open3D Python package
:::::::::::::::::::::::::::::::::

Inside the activated virtualenv (shall be activated before ``cmake``), run

.. code-block:: bash

    make install-pip-package

The above command is **compatible with both pip and Conda virtualenvs**. To
uninstall, run

.. code-block:: bash

    pip uninstall open3d

For more fine-grained controls, you can also run:

.. code-block:: bash

    # Create Python package in build/lib/python_package
    make python-package

    # Create pip wheel in build/lib/python_package/pip_package
    make pip-package

    # Create conda package in build/lib/python_package/conda_package
    make conda-package

    # Install pip wheel
    make install-pip-package

If the installation is successful, we shall now be able to import Open3D

.. code-block:: bash

    python -c "import open3d"

5.2 Install Open3D as a C++ library
:::::::::::::::::::::::::::::::::::

To Install/uninstall the Open3D as a C++ library (headers and binaries):

.. code-block:: bash

    cd build
    make install
    ...
    make uninstall

Note that ``sudo`` may be needed to install Open3D to a system location.

To link a C++ project against the Open3D C++ library, please refer to
:ref:`create_cplusplus_project`, starting from
`this example CMake file <https://github.com/intel-isl/Open3D/tree/master/docs/_static/C%2B%2B>`_.


.. tip:: You may also check out ``utils/scripts`` which contains scripts
    to build, install and verify the code. These scripts may help in subsequent
    builds when contributing to Open3D.

.. _compilation_osx:

MacOS
-----

The MacOS compilation steps are mostly identical with :ref:`compilation_ubuntu`.

1. Install dependencies
```````````````````````

Run ``util/install_deps_macos.sh``. We use `homebrew <https://brew.sh/>`_
to manage dependencies. Follow the instructions from the script.

2. Setup Python binding environments
````````````````````````````````````

Same as the steps for Ubuntu: :ref:`compilation_ubuntu_python_binding`.

3. Config
`````````

Same as the steps for Ubuntu: :ref:`compilation_ubuntu_config`.

Alternatively, to use Xcode IDE, run:

.. code-block:: bash

    mkdir build-xcode
    cd build-xcode
    cmake -G Xcode -DCMAKE_INSTALL_PREFIX=<open3d_install_directory> ..
    open Open3D.xcodeproj/

4. Build
````````

Same as the steps for Ubuntu: :ref:`compilation_ubuntu_build`.

5. Install
``````````

Same as the steps for Ubuntu: :ref:`compilation_ubuntu_install`.

.. _compilation_windows:

Windows
-------

1. Dependencies
```````````````
For easy compilation, we have included source code of all dependent libraries
in the ``3rdparty`` folder. Therefore, we don't need to install any dependencies.

2. Setup Python binding environments
````````````````````````````````````

Most steps are the steps for Ubuntu: :ref:`compilation_ubuntu_python_binding`.
Instead of ``which``, check the Python path with ``where python``, also pay
attention to the ``Found PythonInterp`` message printed by CMake.

3. Config (generate Visual Studio solution)
```````````````````````````````````````````

The CMake GUI is as shown in the following figure. Specify the
directories, click ``Configure`` and choose the correct Visual Studio
version (e.g., ``Visual Studio 16 2019 Win64``), then click ``Generate``.
This will create an ``Open3D.sln`` file in the build directory.

.. image:: _static/cmake_windows.png
    :width: 500px

Alternatively, this file can be generated by calling CMake from the console:

.. code-block:: bat

    mkdir build
    cd build

    :: Run one of the following lines based on your Visual Studio version
    cmake -G "Visual Studio 16 2019 Win64" ..

.. error:: If cmake fail to find ``PYTHON_EXECUTABLE``, follow the Ubuntu guide:
    :ref:`compilation_ubuntu_python_binding` to activate the Python virtualenv before running
    ``cmake`` or specify the Python path manually.

By default, CMake links with dynamic runtime (``/MD`` or ``/MDd``). To link with
static runtime (``/MT`` or ``/MTd``) set ``-DSTATIC_WINDOWS_RUNTIME=ON``.

4. Build
````````

Open ``Open3D.sln`` file with Visual Studio, change the build type to
``Release``, then rebuild the ``ALL_BUILD`` target.

.. image:: _static/open3d.vc_solution.hightlights.png
    :width: 250px

Alternatively, we can also build directly from the CMD terminal. Run

.. code-block:: bat

    cmake --build . --parallel %NUMBER_OF_PROCESSORS% --config Release --target ALL_BUILD

5. Install
``````````

Open3D can be installed as a C++ library or a Python package, by building the
corresponding targets with Visual Studio or from the terminal. E.g.

.. code-block:: bat

    cmake --build . --parallel %NUMBER_OF_PROCESSORS% --config Release --target the-target-name

Here's a list of installation related targets. Please refer to
:ref:`compilation_ubuntu_install` for more detailed documentation.

- ``install``
- ``python-package``
- ``pip-package``
- ``install-pip-package``

Sanity check
------------

For a quick sanity check, try importing the library from the Python interactive
shell:

.. code-block:: bash

    python

    >>> import open3d

.. error:: If there is an issue, check whether the Python version detected by
    CMake (see ``Found PythonInterp`` log from CMake, or check the value of the
    ``PYTHON_EXECUTABLE`` CMake variable) and the Python version for command
    line environment (type ``python -V``). They should match. If it is not,
    please follow :ref:`compilation_ubuntu_python_binding` in docs. In addition,
    `python binding issue  <https://github.com/intel-isl/Open3D/issues?utf8=%E2%9C%93&q=is%3Aissue+label%3A%22python+binding%22+>`_
    on Github repository has helpful notes from Open3D users.

.. _compilation_options:

Compilation Options
-------------------

This page shows advanced options to customize the Open3D build. For quick
start, see :ref:`compilation`.

OpenMP
``````

We automatically detect if the C++ compiler supports OpenMP and compile Open3D
with it if the compilation option ``WITH_OPENMP`` is ``ON``.
OpenMP can greatly accelerate computation on a multi-core CPU.

The default LLVM compiler on OS X does not support OpenMP.
A workaround is to install a C++ compiler with OpenMP support, such as ``gcc``,
then use it to compile Open3D. For example, starting from a clean build
directory, run

.. code-block:: bash

    brew install gcc --without-multilib
    cmake -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 ..
    make -j

.. note:: This workaround has some compatibility issues with the source code of
    GLFW included in ``3rdparty``.
    Make sure Open3D is linked against GLFW installed on the OS.

ML Module
`````````

The ML module consists of primitives like operators and layers as well as high
level code for models and pipelines. To build the operators and layers, set
``BUILD_PYTORCH_OPS=ON`` and/or ``BUILD_TENSORFLOW_OPS=ON``.  Don't forget to also
enable ``BUILD_CUDA_MODULE=ON`` for GPU support. To include the models and
pipelines form Open3D-ML in the python package, set ``BUNDLE_OPEN3D_ML=ON`` and
``OPEN3D_ML_ROOT`` to the Open3D-ML repository. You can directly download
Open3D-ML from GitHub during the build with
``OPEN3D_ML_ROOT=https://github.com/intel-isl/Open3D-ML.git``.

The following example shows the command for building the ops with GPU support
for all supported ML frameworks and bundling the high level Open3D-ML code.

.. code-block:: bash

    # In the build directory
    cmake -DBUILD_CUDA_MODULE=ON \
          -DBUILD_PYTORCH_OPS=ON \
          -DBUILD_TENSORFLOW_OPS=ON \
          -DBUNDLE_OPEN3D_ML=ON \
          -DOPEN3D_ML_ROOT=https://github.com/intel-isl/Open3D-ML.git \
          ..
    # Install the python wheel with pip
    make -j install-pip-package

.. note::
    Importing Python libraries compiled with different CXX ABI may cause segfaults
    in regex. https://stackoverflow.com/q/51382355/1255535. By default, PyTorch
    and TensorFlow Python releases use the older CXX ABI; while when they are
    compiled from source, newer ABI is enabled by default.

    When releasing Open3D as a Python package, we set
    ``-DGLIBCXX_USE_CXX11_ABI=OFF`` and compile all dependencies from source,
    in order to ensure compatibility with PyTorch and TensorFlow Python releases.

    If you build PyTorch or TensorFlow from source or if you run into ABI
    compatibility issues with them, please:

    1. Check PyTorch and TensorFlow ABI with

       .. code-block:: bash

           python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
           python -c "import tensorflow; print(tensorflow.__cxx11_abi_flag__)"

    2. Configure Open3D to compile all dependencies from source
       with the corresponding ABI version obtained from step 1.

    After installation of the Python package, you can check Open3D ABI version
    with:

    .. code-block:: bash

        python -c "import open3d; print(open3d.open3d_pybind._GLIBCXX_USE_CXX11_ABI)"

    To build Open3D with CUDA support, configure with:

    .. code-block:: bash

        cmake -DBUILD_CUDA_MODULE=ON -DCMAKE_INSTALL_PREFIX=<open3d_install_directory> ..

    Please note that CUDA support is work in progress and experimental. For building
    Open3D with CUDA support, ensure that CUDA is properly installed by running following commands:

    .. code-block:: bash

        nvidia-smi      # Prints CUDA-enabled GPU information
        nvcc -V         # Prints compiler version

    If you see an output similar to ``command not found``, you can install CUDA toolkit
    by following the `official
    documentation. <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_


Unit test
`````````

To build unit tests, set `BUILD_UNIT_TESTS=ON` at CMake config stage. The unit
test executable will be located at `bin/tests` in the `build` directory.

Please also refer to `googletest <https://github.com/google/googletest.git>`_ for
reference.

.. code-block:: bash

    # In the build directory
    cmake -DBUILD_UNIT_TESTS=ON ..
    make -j
    ./bin/tests
