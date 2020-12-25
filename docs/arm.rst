.. _arm:

ARM support
===========

Open3D provides experimental support for 64-bit ARM architecture (``arm64``
or ``aarch64``) on Linux. Open3D needs to be compiled from source to run on ARM.

System requirements
-------------------

* 64-bit ARM processor and 64-bit Linux operating system. Check the output of
  ``uname -p`` and it should show ``aarch64``.
* Full OpenGL (not OpenGL ES) is needed for Open3D GUI. If OpenGL is not
  available, the Open3D GUI will compile but it won't run. In this case, we
  recommend setting ``-DBUILD_GUI=OFF`` during the ``cmake`` configuration step.


Building Open3D on ARM64
------------------------

Note: If you encounter build issues, check the ``arm64`` section of
``.github/workflows/openblas.yml`` for the full CI build scripts on ARM64.


Install dependencies
````````````````````

Install the following system dependencies:

.. code-block:: bash

    sudo apt-get update -y
    sudo apt-get install -y apt-utils build-essential git cmake
    sudo apt-get install -y python3 python3-dev python3-pip
    sudo apt-get install -y xorg-dev libglu1-mesa-dev
    sudo apt-get install -y libblas-dev liblapack-dev liblapacke-dev
    sudo apt-get install -y libsdl2-dev libc++-7-dev libc++abi-7-dev libxi-dev
    sudo apt-get install -y clang-7

Optionally, ``virtualenv`` and ``ccache`` are recommended. Note that conda does
not support ARM.

.. code-block:: bash

    sudo apt-get install -y python3-virtualenv ccache

If the Open3D build system complains about ``CMake xxx or higher is required``,
refer to one of the following options:

* `Compile CMake from source <https://cmake.org/install/>`_
* Install with ``snap``: ``sudo snap install cmake --classic``
* Install with ``pip`` (run inside a Python virtual environment): ``pip install cmake``


Build
`````

.. code-block:: bash

    # Optional: create and activate virtual environment
    virtualenv --python=$(which python3) ${HOME}/venv
    source ${HOME}/venv/bin/activate

    # Clone
    git clone --recursive https://github.com/intel-isl/Open3D
    cd Open3D
    git submodule update --init --recursive
    mkdir build
    cd build

    # Configure
    # > Set -DBUILD_CUDA_MODULE=ON if CUDA is available (e.g. on Nvidia Jetson)
    # > Set -DBUILD_GUI=ON if OpenGL is available (e.g. on Nvidia Jetson)
    # > We don't support TensorFlow and PyTorch on ARM officially
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_GUI=OFF \
        -DBUILD_TENSORFLOW_OPS=OFF \
        -DBUILD_PYTORCH_OPS=OFF \
        -DBUILD_UNIT_TESTS=ON \
        -DCMAKE_INSTALL_PREFIX=~/open3d_install \
        -DPYTHON_EXECUTABLE=$(which python) \
        ..

    # Build C++ library
    make -j$(nproc)

    # Run tests (optional)
    make tests -j$(nproc)
    ./bin/tests --gtest_filter="-*Reduce*Sum*"

    # Install C++ package (optional)
    make install

    # Install Open3D python package (optional)
    make install-pip-package -j$(nproc)
    python -c "import open3d; print(open3d)"

    # Run Open3D GUI (optional, available on when -DBUILD_GUI=ON)
    ./bin/Open3D/Open3D


Nvidia Jetson
-------------

Nvidia Jetson computers with 64-bit processor and OS are supported. You can
compile Open3D with ``-DBUILD_CUDA_MODULE=ON`` and ``-DBUILD_GUI=ON`` and
the Open3D GUI app should be functional. We support CUDA v10.x, but other
versions should work as well.


Raspberry Pi 4
--------------

Raspberry Pi 4 has 64-bit processor and supports OpenGL ES (not OpenGL).
To build Open3D on Raspberry Pi 4, compile with ``-DBUILD_GUI=OFF``.
