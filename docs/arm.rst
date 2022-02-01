.. _arm:

ARM support
===========

Open3D provides experimental support for 64-bit ARM architecture (``arm64``
or ``aarch64``) on Linux and macOS (Apple Silicon). Starting from Open3D 0.14,
we provide pre-compiled ARM64 wheels for Linux and macOS. Install the wheel by:

.. code-block:: bash

    pip install open3d
    python -c "import open3d; print(open3d.__version__)"

    # Test the legacy visualizer
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw_geometries([c])"

    # Test the new GUI visualizer
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw(c)"

+------------------------+----------------+---------------------+------------+----------------+
|                        | Linux (OpenGL) | Linux (OpenGL ES)   | macOS      | Windows on ARM |
+========================+================+=====================+============+================+
| ``pip install open3d`` | Yes            | Yes                 | Yes        | No             |
+------------------------+----------------+---------------------+------------+----------------+
| Compile from source    | Yes            | Yes                 | Yes        | No             |
+------------------------+----------------+---------------------+------------+----------------+
| Visualizer and GUI     | Yes            | No                  | Yes        | No             |
+------------------------+----------------+---------------------+------------+----------------+
| Non-GUI features       | Yes            | Yes                 | Yes        | No             |
+------------------------+----------------+---------------------+------------+----------------+
| Special build flags    | Not needed     | ``-DBUILD_GUI=OFF`` | Not needed | N/A            |
+------------------------+----------------+---------------------+------------+----------------+
| Example device         | Nvidia Jetson  | Raspberry Pi 4      | M1 MacBook | Surface Pro X  |
+------------------------+----------------+---------------------+------------+----------------+

Additional notes:

* On Linux, check the output of ``uname -p`` and it should show ``aarch64``. On
  macOS, check the output of ``uname -m`` and it should show ``arm64``.
* Full OpenGL (not OpenGL ES) is needed for Open3D GUI. Open3D GUI is supported
  on Nvidia Jetson platforms and on Apple ARM64 devices.
* If the full OpenGL is not available (e.g. on Raspberry Pi devices), the Open3D
  GUI code  will compile but it won't run. In this case, we recommend setting
  ``-DBUILD_GUI=OFF`` during the ``cmake`` configuration step.
* For Windows on ARM devices, Open3D might work with the x86 emulation layer,
  but it is not official supported.
* Open3D installed via ``pip install open3d`` will not contain CUDA support on
  ARM64 platforms. To use CUDA, you need to compile Open3D with CUDA manually
  for Nvidia Jetson boards.

Compiling Open3D on ARM64 Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Building Open3D Python Wheel with Docker
----------------------------------------

This is recommended for Python users. By using Docker, the only dependency to
install is the Docker engine itself. This is especially useful since ARM64 Linux
has many variants and it could be difficult to configure all dependencies
manually.

To build Open3D Python wheel with Docker, you can run one of the following
commands:

.. code-block:: bash

    cd docker

    ./docker_build.sh openblas-arm64-py36  # Python 3.6
    ./docker_build.sh openblas-arm64-py37  # Python 3.7
    ./docker_build.sh openblas-arm64-py38  # Python 3.8
    ./docker_build.sh openblas-arm64-py39  # Python 3.9

After running ``docker_build.sh``, you shall see a ``.whl`` file generated the
current directly on the host. Then simply install the ``.whl`` file by:

.. code-block:: bash

    # Optional: activate your virtualenv
    conda activate your-virtual-env

    # Install and test
    pip install open3d-*.whl
    python -c "import open3d; print(open3d.__version__)"
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw_geometries([c])"
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw(c)"

The ``./docker_build.sh`` script works on both Linux and macOS ARM64 hosts.  You
can even cross-compile an ARM64 wheel on an x86-64 host. Install Docker and
Qemu:

.. code-block:: bash

    sudo apt-get --yes install qemu binfmt-support qemu-user-static

and follow the same steps as above.


Building Open3D directly
------------------------

You may run into issues building Open3D directly on your ARM64 machine due to
dependency conflicts or version incompatibilities. In general, we recommend
building from a clean OS and only install the required dependencies by Open3D.
It has been reported by users that some globally installed packages (e.g.
TBB, Parallel STL, BLAS, LAPACK) may cause compatibility issues if they are not
the same version as the one used by Open3D.

If you only need the Python wheel, consider using the Docker build method or
install Open3D via ``pip install open3d`` directly.

Install dependencies
````````````````````

.. code-block:: bash

    # Install dependencies
    ./util/install_deps_ubuntu.sh
    sudo apt-get install -y clang-7  # Or any >= 7 version of clang.

    # Optional: ccache is recommended to speed up subsequent builds
    sudo apt-get install -y ccache

    # Check cmake version, you should have 3.19+
    cmake --version


If the Open3D build system complains about ``CMake xxx or higher is required``,
refer to one of the following options:

* `Compile CMake from source <https://cmake.org/install/>`_
* Download the pre-compiled ``aarch64`` CMake from `CMake releases <https://github.com/Kitware/CMake/releases/>`_,
  and setup ``PATH`` accordingly.
* Install with ``pip`` (run inside a Python virtual environment): ``pip install cmake``

Build
`````

.. code-block:: bash

    # Optional: activate your virtualenv
    conda activate your-virtual-env

    # Configure
    # Set -DBUILD_CUDA_MODULE=ON if CUDA is available (e.g. on Nvidia Jetson)
    # Set -DBUILD_GUI=ON if full OpenGL is available (e.g. on Nvidia Jetson)
    cd Open3D && mkdir build && cd build
    cmake -DBUILD_CUDA_MODULE=OFF -DBUILD_GUI=OFF ..

    # Build
    make -j$(nproc)
    make install-pip-package -j$(nproc)

    # Test C++ viewer app (only available when -DBUILD_GUI=ON)
    ./bin/Open3D/Open3D

    # Test Python visualization (only available when -DBUILD_GUI=ON)
    python -c "import open3d; print(open3d.__version__)"
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw_geometries([c])"
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw(c)"


Compiling Open3D on ARM64 macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Dependencies
    brew install gfortran

    # Optional: ccache is recommended to speed up subsequent builds
    sudo apt-get install -y ccache

    # Optional: activate your virtualenv
    conda activate your-virtual-env

    # Configure
    cd Open3D && mkdir build && cd build
    cmake ..

    # Build
    make -j8
    make install-pip-package -j8

    # Test C++ viewer app
    ./bin/Open3D/Open3D

    # Test Python visualization
    python -c "import open3d; print(open3d.__version__)"
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw_geometries([c])"
    python -c "import open3d as o3d; c = o3d.geometry.TriangleMesh.create_box(); o3d.visualization.draw(c)"
