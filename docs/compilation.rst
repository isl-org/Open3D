.. _compilation:

Build from source
=====================

.. _compiler_version:

System requirements
-------------------

* C++17 compiler:

  * Ubuntu 20.04+: GCC 5+, Clang 7+
  * macOS 10.15+: XCode 8.0+
  * Windows 10 (64-bit): Visual Studio 2019+

* CMake: 3.24+

  * Ubuntu (20.04+):

    * Install with ``apt-get``: see `official APT repository <https://apt.kitware.com/>`_
    * Install with ``snap``: ``sudo snap install cmake --classic``
    * Install with ``pip`` (run inside a Python virtualenv): ``pip install cmake``

  * macOS: Install with Homebrew: ``brew install cmake``
  * Windows: Download from: `CMake download page <https://cmake.org/download/>`_

* CUDA 11.5+ (optional): Open3D supports GPU acceleration through CUDA on Linux.
  Prebuilt wheels statically link the CUDA runtime libraries directly into
  ``libOpen3D``, so no separate NVIDIA runtime pip packages are required at
  import time. For building from source, install the CUDA toolkit
  (``nvidia-smi``, ``nvcc -V``) and configure with ``-DBUILD_CUDA_MODULE=ON``
  (``-DBUILD_WITH_CUDA_STATIC=ON`` by default). We recommend using CUDA 13+
  for the best compatibility with recent GPUs and optional external
  dependencies such as Tensorflow or PyTorch.

* Shared libraries: Open3D builds with ``BUILD_SHARED_LIBS=ON`` by default,
  producing a single ``libOpen3D.so`` / ``Open3D.dll`` that contains CPU and one
  of CUDA or SYCL code together. Python wheels ship one ``pybind`` extension
  module that links against this single library and picks the available
  device (CPU + CUDA or SYCL) at runtime.

* Ccache 4.0+ (optional, recommended): ccache is a compiler cache that can
  speed up the compilation process by avoiding recompilation of the same
  source code. Please refer to :ref:`ccache` for installation guides.

Cloning Open3D
--------------

.. code-block:: bash

    git clone https://github.com/isl-org/Open3D

.. _compilation_unix:

Ubuntu/macOS
------------

.. _compilation_unix_dependencies:

1. Install dependencies
```````````````````````

.. code-block:: bash

    # Only needed for Ubuntu
    util/install_deps_ubuntu.sh

.. _compilation_unix_python:

2. Setup Python environments
````````````````````````````

Activate the Python ``virtualenv`` or Conda environment. Check
``which python`` to ensure that it shows the desired Python executable.
Alternatively, set the CMake flag ``-DPython3_ROOT=/path/to/python``
to specify the path to the Python installation.

If Python binding is not needed, you can turn it off by ``-DBUILD_PYTHON_MODULE=OFF``.

.. _compilation_unix_config:

3. Config
`````````
.. code-block:: bash

    mkdir build
    cd build
    cmake ..

You can specify ``-DCMAKE_INSTALL_PREFIX=$HOME/open3d_install`` to control the
installation directory of ``make install``. In the absence of
``CMAKE_INSTALL_PREFIX``, Open3D will be installed to a system location where
``sudo`` may be required.

For more build options, see :ref:`compilation_options` and the root
``CMakeLists.txt``.

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
:ref:`cplusplus_example_project`.

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
:ref:`cplusplus_example_project`.

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

Filament
````````

The visualization module depends on the Filament rendering engine and, by default,
Open3D uses a prebuilt version of it. You can also build Filament from source
by setting ``BUILD_FILAMENT_FROM_SOURCE=ON``.

ML Module
`````````

The ML module consists of primitives like operators and layers as well as high
level code for models and pipelines. To build the operators and layers, set
``BUILD_PYTORCH_OPS=ON`` and/or ``BUILD_TENSORFLOW_OPS=ON``.  Don't forget to also
enable ``BUILD_CUDA_MODULE=ON`` for GPU support. To include the models and
pipelines from Open3D-ML in the python package, set ``BUNDLE_OPEN3D_ML=ON`` and
``OPEN3D_ML_ROOT`` to the Open3D-ML repository. You can directly download
Open3D-ML from GitHub during the build with
``OPEN3D_ML_ROOT=https://github.com/isl-org/Open3D-ML.git``.

.. warning:: Compiling PyTorch ops with PyTorch < 1.9 may have
    stability issues. See `Open3D issue #3324 <https://github.com/isl-org/Open3D/issues/3324>`_
    and `PyTorch issue #52663 <https://github.com/pytorch/pytorch/issues/52663>`_
    for more information on this problem. Official PyTorch wheels 1.9 and later
    do not have this problem.

    We recommend to compile Pytorch from source
    with compile flags ``-Xcompiler -fno-gnu-unique`` or use the `PyTorch 1.8.2
    wheels from Open3D.
    <https://github.com/isl-org/open3d_downloads/releases/tag/torch1.8.2>`_
    To reproduce the Open3D PyTorch 1.8.2 wheels see the builder repository `here.
    <https://github.com/isl-org/pytorch_builder>`_

The following example shows the command for building the ops with GPU support
for all supported ML frameworks and bundling the high level Open3D-ML code.

.. code-block:: bash

    # In the build directory
    cmake -DBUILD_CUDA_MODULE=ON \
          -DGLIBCXX_USE_CXX11_ABI=OFF \
          -DBUILD_PYTORCH_OPS=ON \
          -DBUILD_TENSORFLOW_OPS=ON \
          -DBUNDLE_OPEN3D_ML=ON \
          -DOPEN3D_ML_ROOT=https://github.com/isl-org/Open3D-ML.git \
          ..
    # Install the python wheel with pip
    make -j install-pip-package

.. note::
    On Linux, importing Python libraries compiled with different CXX ABI may
    cause segfaults in regex. https://stackoverflow.com/q/51382355/1255535. By
    default, PyTorch and TensorFlow Python releases use the older CXX ABI; while
    when compiled from source, the newer CXX11 ABI is enabled by default.

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

        cmake -DBUILD_CUDA_MODULE=ON -DBUILD_SHARED_LIBS=ON \
              -DCMAKE_INSTALL_PREFIX=<open3d_install_directory> ..

    CUDA runtime libraries are statically linked into ``libOpen3D`` by default
    (``-DBUILD_WITH_CUDA_STATIC=ON``), so CUDA wheels do not need any NVIDIA
    redistributable pip packages at import time. Pass
    ``-DBUILD_WITH_CUDA_STATIC=OFF`` to dynamically link the CUDA runtime
    instead. For development, ensure the CUDA toolkit is available:

    .. code-block:: bash

        nvidia-smi      # Prints CUDA-enabled GPU information
        nvcc -V         # Prints compiler version

    If you see an output similar to ``command not found``, you can install CUDA toolkit
    by following the `official
    documentation. <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_

.. _abi_dependency_compatibility:

ABI dependency and compatibility
---------------------------------

When compiling Open3D from source or using prebuilt wheels with machine learning
(ML) framework bindings (such as PyTorch or TensorFlow) and hardware
acceleration (such as CUDA or SYCL), maintaining Application Binary Interface
(ABI) compatibility across all dependencies is critical.  An ABI mismatch
between Open3D, the ML frameworks, and the underlying runtime libraries can lead
to compilation failures, linker errors, or runtime crashes (such as segmentation
faults).

ABI Dependency Tree
````````````````````

The diagram below illustrates how Open3D and its custom operators depend on the
underlying runtime ABI libraries:

.. code-block:: text

       +-----------------------------------------------------+
       |                 Runtime ABI Library                 |
       |     (glibc, libstdc++, cuda-runtime, sycl-runtime)  |
       +-----------+--------------+--------------+-----------+
                   ^              ^              ^
                   |              |              |
                   |        +-----+-----+  +-----+-----+
                   |        |  PyTorch  |  |TensorFlow |
                   |        +-----+-----+  +-----+-----+
                   |              ^              ^
                   |              |              |
                   |        +-----+-----+  +-----+-----+
                   |        | torch-ops |  |  tf-ops   |
                   |        +-----+-----+  +-----+-----+
                   |              ^              ^
                   |              |              |
             +-----+-----+--------+--------------+-----+
             |                 Open3D                  |
             +-----------------------------------------+

As shown in the diagram:

* **Open3D** directly links against the core runtime ABI libraries (such as ``glibc``, ``nvidia-rt``, or ``sycl-rt``). CPU, CUDA, and SYCL code all live in the same ``libOpen3D`` shared library.
* When custom ML operators are enabled (``BUILD_PYTORCH_OPS=ON`` or ``BUILD_TENSORFLOW_OPS=ON``), Open3D compiles custom operator libraries (``torch-ops`` and ``tf-ops``).
* These custom operators depend directly on the installed ML frameworks (``pytorch`` and ``tensorflow``).
* Both the ML frameworks and the custom operators must link against the exact same runtime ABI libraries.

Consequently, the dependency versions must be compatible. Typically, the major
and minor versions of the toolchains and runtime libraries used at build time
must match those used by the installed ML frameworks, and the runtime
environment must satisfy the compatibility guarantees of each library.

Runtime ABI Libraries and Compatibility Guarantees
`````````````````````````````````````````````````````

glibc (GNU C Library)
"""""""""""""""""""""

* **Backward Compatibility**: ``glibc`` guarantees strict backward compatibility. A binary compiled against an older version of ``glibc`` (e.g., ``glibc 2.31`` on Ubuntu 20.04) will run without issues on a system with a newer version of ``glibc`` (e.g., ``glibc 2.35`` on Ubuntu 22.04). It does **not** guarantee forward compatibility.
* **C++ Standard Library ABI (libstdc++)**:
  * Open3D compiles with ``GLIBCXX_USE_CXX11_ABI=ON`` by default.
  * Modern PyTorch and TensorFlow Linux wheel releases are also built with CXX11 ABI enabled (``_GLIBCXX_USE_CXX11_ABI=1``) by default.
  * Open3D's CMake configuration automatically queries and verifies that the ABI configuration of the installed PyTorch and TensorFlow matches Open3D's configuration to prevent linker and runtime errors.

nvidia-rt (NVIDIA CUDA Runtime and Driver)
""""""""""""""""""""""""""""""""""""""""""

* **Static Linking for Safety**:
  * To prevent runtime version mismatch issues between different CUDA runtimes, Open3D builds with ``BUILD_WITH_CUDA_STATIC=ON`` by default.
  * This statically links CUDA toolkit libraries (such as ``cudart_static``, ``cublas_static``, ``cusolver_static``, ``cusparse_static``, and ``npp*_static``) directly into ``libOpen3D``, isolating Open3D from external CUDA runtime version mismatches.
* **Compatibility Guarantees**:
  * **Driver Backward Compatibility**: Newer NVIDIA drivers support older CUDA Toolkit and CUDA Runtime versions.
  * **CUDA Minor Version Compatibility**: Starting with CUDA 11, NVIDIA guarantees binary compatibility within the same major version. An application compiled with any CUDA 11.x SDK can run on any driver that supports CUDA 11.0 or later. The same applies to CUDA 12.x and CUDA 13.x.

sycl-rt (Intel oneAPI DPC++ Runtime)
""""""""""""""""""""""""""""""""""""

* **Compatibility Guarantees**:
  * **oneAPI Runtime Compatibility**: Intel oneAPI guarantees backward compatibility for the DPC++ runtime (``dpcpp-cpp-rt``). A newer runtime can execute binaries compiled with an older oneAPI compiler. Forward compatibility is not supported.
  * **Strict Version Matching**: Due to rapid development in SYCL standards and implementations, it is highly recommended to use matching major and minor versions of the DPC++ compiler and runtime.
  * Open3D's SYCL wheels pin the runtime dependency in ``python/requirements_sycl.txt`` (e.g., ``dpcpp-cpp-rt==2025.3.1``). When compiling Open3D with SYCL support, ensure your Intel oneAPI Base Toolkit version is compatible with this runtime.

WebRTC remote visualization
```````````````````````````

We provide pre-built binaries of the `WebRTC library <https://webrtc.org/>`_ to
build Open3D with remote visualization. Currently, Linux, macOS and Windows are
supported for ``x86_64`` architecture. If you wish to use a different version of
WebRTC or build for a different configuration or platform, please see the
`official WebRTC documentation
<https://webrtc.googlesource.com/src/+/refs/heads/master/docs/native-code/development/index.md>`_
and the Open3D build scripts.

Linux and macOS
"""""""""""""""
Please see the build script ``3rdparty/webrtc/webrtc_build.sh``. For Linux, you
can also use the provided ``3rdparty/webrtc/Dockerfile.webrtc`` for building.

Windows
"""""""
We provide Windows MSVC static libraries built in Release and Debug mode built with
the static Windows runtime. This corresponds to building with the ``/MT`` and
``/MTd`` options respectively. For the build procedure, please see
``.github/workflows/webrtc.yml``. Other configurations are not supported.

Unit test
---------

To build and run C++ unit tests:

.. code-block:: bash

    cmake -DBUILD_UNIT_TESTS=ON ..
    make -j$(nproc)
    ./bin/tests

To run Python unit tests:

.. code-block:: bash

    # Activate virtualenv first
    pip install pytest
    make install-pip-package -j$(nproc)
    pytest ../python/test

.. _ccache:

Caching compilation with ccache
-------------------------------

ccache is a compiler cache that can speed up the compilation process by avoiding
recompilation of the same source code. It can significantly speed up
recompilation of Open3D on Linux/macOS, even if you clear the ``build``
directory. You'll need ccache 4.0+ to cache both C++ and CUDA compilations.

After installing ``ccache``, simply reconfigure and recompile the Open3D
library. Open3D's CMake script can detect and use it automatically. You don't
need to setup additional paths except for the ``ccache`` program itself.

Ubuntu 20.04+
`````````````

If you install ``ccache`` via ``sudo apt install ccache``, the 3.x version will
be installed. To cache CUDA compilations, you'll need the 4.0+ version. Here, we
demonstrate one way to setup ``ccache`` by compiling it from source, installing
it to ``${HOME}/bin``, and adding ``${HOME}/bin`` to ``${PATH}``.

.. code-block:: bash

    # Clone
    git clone https://github.com/ccache/ccache.git
    cd ccache
    git checkout v4.6 -b 4.6

    # Build
    mkdir build
    cd build
    cmake -DZSTD_FROM_INTERNET=ON \
          -DHIREDIS_FROM_INTERNET=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=${HOME} \
          ..
    make -j$(nproc)
    make install -j$(nproc)

    # Add ${HOME}/bin to ${PATH} in your ~/.bashrc
    echo "PATH=${HOME}/bin:${PATH}" >> ~/.bashrc

    # Restart the terminal now, or source ~/.bashrc
    source ~/.bashrc

    # Verify `ccache` has been installed correctly
    which ccache
    ccache --version

Ubuntu 22.04+
`````````````

.. code-block:: bash

    sudo apt install ccache

macOS
`````

.. code-block:: bash

    brew install ccache

Monitoring ccache statistics
````````````````````````````

.. code-block:: bash

    ccache -s
