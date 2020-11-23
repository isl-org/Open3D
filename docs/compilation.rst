.. _compilation:

Build from source
=====================

.. _compiler_version:

System requirements
-------------------

* Ubuntu 18.04+: GCC 5+, Clang 7+
* macOS 10.14+: XCode 8.0+
* Windows 10 (64-bit): Visual Studio 2019+
* CMake: 3.15+ for Ubuntu and macOS, 3.18+ for Windows

  * Ubuntu (18.04):

    * Install with ``apt-get``: see `official APT repository <https://apt.kitware.com/>`_
    * Install with ``snap``: ``sudo snap install cmake --classic``
    * Install with ``pip`` (run inside a Python virtualenv): ``pip install cmake``

  * Ubuntu (20.04+): Use the default OS repository: ``sudo apt-get install cmake``
  * macOS: Install with Homebrew: ``brew install cmake``
  * Windows: Download from: `CMake download page <https://cmake.org/download/>`_

* CUDA 10.1 (optional): Open3D supports GPU acceleration of an increasing number
  of operations through CUDA on Linux. Please see the `official documentation
  <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ to
  install the CUDA toolkit from Nvidia.


Cloning Open3D
--------------

Make sure to use the ``--recursive`` flag when cloning Open3D.

.. code-block:: bash

    git clone --recursive https://github.com/intel-isl/Open3D

    # You can also update the submodule manually
    git submodule update --init --recursive

.. _compilation_unix:

Ubuntu/macOS
------------

.. _compilation_unix_dependencies:

1. Install dependencies
```````````````````````

.. code-block:: bash

    # On Ubuntu
    util/install_deps_ubuntu.sh

    # On macOS
    # Install Homebrew first: https://brew.sh/
    util/install_deps_macos.sh

.. _compilation_unix_python:

2. Setup Python environments
````````````````````````````

Activate the python ``virtualenv`` or Conda ``virtualenv```. Check
``which python`` to ensure that it shows the desired Python executable.
Alternatively, set the CMake flag ``-DPYTHON_EXECUTABLE=/path/to/python``
to specify the python executable.

If Python binding is not needed, you can turn it off by ``-DBUILD_PYTHON_MODULE=OFF``.

.. _compilation_unix_config:

3. Config
`````````
.. code-block:: bash

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=<open3d_install_directory> ..

The ``CMAKE_INSTALL_PREFIX`` argument is optional and can be used to install
Open3D to a user location. In the absence of this argument Open3D will be
installed to a system location where ``sudo`` is required) For more
options of the build, see :ref:`compilation_options`.

.. _compilation_unix_build:

4. Build
````````

.. code-block:: bash

    # On Ubuntu
    make -j$(nproc)

    # On macOS
    make -j$(sysctl -n hw.physicalcpu)

.. _compilation_unix_install:

5. Install
``````````

To install Open3D C++ library:

.. code-block:: bash

    make install

To link a C++ project against the Open3D C++ library, please refer to
:ref:`create_cplusplus_project`.


To install Open3D Python library, build one of the following options:

.. code-block:: bash

    # Activate the virtualenv first
    # Install pip package in the current python environment
    make install-pip-package

    # Create Python package in build/lib
    make python-package

    # Create pip wheel in build/lib
    # This creates a .whl file that you can install manually.
    make pip-package

    # Create conda package in build/lib
    # This creates a .tar.bz2 file that you can install manually.
    make conda-package

Finally, verify the python installation with:

.. code-block:: bash

    python -c "import open3d"


.. _compilation_windows:

Windows
-------

1. Setup Python binding environments
````````````````````````````````````

Most steps are the steps for Ubuntu: :ref:`compilation_unix_python`.
Instead of ``which``, check the Python path with ``where python``.

2. Config
`````````

.. code-block:: bat

    mkdir build
    cd build

    :: Specify the generator based on your Visual Studio version
    :: If CMAKE_INSTALL_PREFIX is a system folder, admin access is needed for installation
    cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX="<open3d_install_directory>" ..

3. Build
````````

.. code-block:: bat

    cmake --build . --config Release --target ALL_BUILD

Alternatively, you can open the ``Open3D.sln`` project with Visual Studio and
build the same target.

4. Install
``````````

To install Open3D C++ library, build the ``INSTALL`` target in terminal or
in Visual Studio.

.. code-block:: bat

    cmake --build . --config Release --target INSTALL

To link a C++ project against the Open3D C++ library, please refer to
:ref:`create_cplusplus_project`.

To install Open3D Python library, build the corresponding python installation
targets in terminal or Visual Studio.

.. code-block:: bat

    :: Activate the virtualenv first
    :: Install pip package in the current python environment
    cmake --build . --config Release --target install-pip-package

    :: Create Python package in build/lib
    cmake --build . --config Release --target python-package

    :: Create pip package in build/lib
    :: This creates a .whl file that you can install manually.
    cmake --build . --config Release --target pip-package

    :: Create conda package in build/lib
    :: This creates a .tar.bz2 file that you can install manually.
    cmake --build . --config Release --target conda-package

Finally, verify the Python installation with:

.. code-block:: bash

    python -c "import open3d; print(open3d)"

.. _compilation_options:

Compilation options
-------------------

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
pipelines from Open3D-ML in the python package, set ``BUNDLE_OPEN3D_ML=ON`` and
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

        python -c "import open3d; print(open3d.pybind._GLIBCXX_USE_CXX11_ABI)"

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
---------

To build and run C++ unit tests:

.. code-block:: bash

    cmake -DBUILD_UNIT_TESTS=ON ..
    make -j
    ./bin/tests


To run Python unit tests:

.. code-block:: bash

    # Activate virtualenv first
    pip install pytest
    make install-pip-package
    pytest ../python/test
