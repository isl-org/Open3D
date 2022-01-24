.. _arm:

ARM support
===========

Open3D provides experimental support for 64-bit ARM architecture (``arm64``
or ``aarch64``) on Linux.

* 64-bit ARM processor and 64-bit Linux operating system. Check the output of
  ``uname -p`` and it should show ``aarch64``.
* Full OpenGL (not OpenGL ES) is needed for Open3D GUI. If OpenGL is not
  available, the Open3D GUI will compile but it won't run. In this case, we
  recommend setting ``-DBUILD_GUI=OFF`` during the ``cmake`` configuration step.

Starting from Open3D 0.14, we provide pre-compiled ARM64 wheels for Linux.
Install by:

.. code-block:: bash

    pip install open3d
    python -c "import open3d; print(open3d.__version__)"

Building Open3D Python Wheel with Docker
----------------------------------------

This is recommended for Python users. By using Docker, the only dependency to
install is the Docker engine itself. This is especially useful since ARM64 Linux
has many variants and it could be difficult to configure all dependencies
manually.

First, install Docker following the `official guide <https://docs.docker.com/get-docker/>`_.
Also, complete the `post-installation steps for Linux https://docs.docker.com/engine/install/linux-postinstall/>`_.
Make sure that ``docker`` can be executed without root privileges. To verify
Docker is installed correctly, run:

.. code-block:: bash

    # You should be able to run this without sudo.
    docker run hello-world

The next step is to build Open3D Python wheel with Docker. You can run one of
the following commands:

.. code-block:: bash

    cd Docker

    ./docker_build.sh openblas-arm64-py36-dev  # Python 3.6 wheel, dev mode
    ./docker_build.sh openblas-arm64-py37-dev  # Python 3.7 wheel, dev mode
    ./docker_build.sh openblas-arm64-py38-dev  # Python 3.8 wheel, dev mode
    ./docker_build.sh openblas-arm64-py39-dev  # Python 3.9 wheel, dev mode
    ./docker_build.sh openblas-arm64-py36      # Python 3.6 wheel, release mode
    ./docker_build.sh openblas-arm64-py37      # Python 3.7 wheel, release mode
    ./docker_build.sh openblas-arm64-py38      # Python 3.8 wheel, release mode
    ./docker_build.sh openblas-arm64-py39      # Python 3.9 wheel, release mode

The release mode is pretty much the same as the dev mode, except that the
version number of the release mode does not contain the commit id. Typically,
using the dev mode is sufficient for most users.

After running ``docker_build.sh``, you shall see a ``.whl`` file generated the
current directly on the host. Then simply install the ``.whl`` file by:

.. code-block:: bash

    # (Activate the virtual environment first)
    pip install open3d-*.whl

You can even cross-compile an ARM64 wheel on an x86-64 host. Install Docker and
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

Install the following system dependencies:

.. code-block:: bash

    ./util/install_deps_ubuntu.sh
    sudo apt-get install -y clang-7  # Or any >= 7 version of clang.

``ccache`` is recommended to cache the compilation:

.. code-block:: bash

    sudo apt-get install -y ccache

If the Open3D build system complains about ``CMake xxx or higher is required``,
refer to one of the following options:

* `Compile CMake from source <https://cmake.org/install/>`_
* Download the pre-compiled ``aarch64`` CMake from `CMake releases <https://github.com/Kitware/CMake/releases/>`_,
  and setup ``PATH`` accordingly.
* Install with ``snap``: ``sudo snap install cmake --classic``
* Install with ``pip`` (run inside a Python virtual environment): ``pip install cmake``

Build
`````

.. code-block:: bash

    # Optional: create and activate virtual environment
    virtualenv --python=$(which python3) ${HOME}/venv
    source ${HOME}/venv/bin/activate

    # Clone
    git clone https://github.com/isl-org/Open3D
    cd Open3D
    mkdir build
    cd build

    # Configure
    # > Set -DBUILD_CUDA_MODULE=ON if CUDA is available (e.g. on Nvidia Jetson)
    # > Set -DBUILD_GUI=ON if full OpenGL is available (e.g. on Nvidia Jetson)
    cmake -DBUILD_CUDA_MODULE=OFF -DBUILD_GUI=OFF ..

    # Build C++ library
    make -j$(nproc)

    # Run Open3D C++ Viewer App (only available on when -DBUILD_GUI=ON)
    ./bin/Open3D/Open3D

    # Install Open3D python package
    make install-pip-package -j$(nproc)

    # Test import Open3D python package
    python -c "import open3d; print(open3d)"


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
