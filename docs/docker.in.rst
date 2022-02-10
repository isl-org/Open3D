.. _docker:

Docker
======

Docker provides a convenient way to build, install and run applications isolated
from the rest of your system. You do not need to change software versions on
your system or install new software, except the Docker engine itself.

First, install Docker following the `official guide <https://docs.docker.com/get-docker/>`_.
Also, complete the `post-installation steps for Linux <https://docs.docker.com/engine/install/linux-postinstall/>`_.
Make sure that ``docker`` can be executed without root privileges. To verify
Docker is installed correctly, run:

.. code-block:: bash

    # You should be able to run this without sudo.
    docker run hello-world


Install and run Open3D apps in docker
-------------------------------------

You can install and run Open3D applications from a docker container.

For Python application, you will need to install a minimum set of dependencies.
For more details please see `this issue
<https://github.com/isl-org/Open3D/issues/3388>`__. A minimal ``Dockerfile`` for
Python applications looks like this:

.. code-block:: dockerfile

    # This could also be another Ubuntu or Debian based distribution
    FROM ubuntu:latest

    # Install Open3D system dependencies and pip
    RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libgomp1 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

    # Install Open3D from the pypi repositories
    RUN python3 -m pip install --no-cache-dir --upgrade open3d

If you have an NVIDIA GPU and want to use it for computation (``CUDA``) or
visualization, follow these `directions.
<https://docs.docker.com/config/containers/resource_constraints/#gpu>`__

To run GUI applications from the docker container, add these options to the
``docker run`` command line to ensure that docker has access to the:

1. GPU:

  - Intel (Mesa drivers): ``--device=/dev/dri:/dev/dri``

  - NVIDIA: ``--gpus 'all,"capabilities=compute,utility,graphics"'``

2. X server: ``-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY``

For example, the following commands will build and run the docker container with
the Open3D viewer application, and provide access to the current directory as
``/root``.  Once the docker image is built, you can run the container from any
folder that contains data you wish to visualize.

.. code-block:: bash

    mkdir open3d-viewer-docker && cd open3d-viewer-docker
    # Download Open3D viewer deb package.
    wget https://github.com/isl-org/Open3D/releases/download/v@OPEN3D_VERSION@/open3d-app-@OPEN3D_VERSION@-Ubuntu.deb
    # Build docker image in folder containing Open3D deb package.
    docker build -t open3d-viewer -f- . <<EOF
    FROM ubuntu:latest
    COPY open3d*.deb /root/
    RUN apt-get update \
        && apt-get install --yes /root/open3d*.deb \
        && rm -rf /var/lib/apt/lists/*
    ENTRYPOINT ["Open3D"]
    EOF

    # Allow local X11 connections
    xhost local:root
    # Run Open3D viewer docker image with the Intel GPU
    docker run --device=/dev/dri:/dev/dri \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
        -v "$PWD":/root open3d-viewer:latest
    # Run Open3D viewer docker image with the NVIDIA GPU
    docker run  --gpus 'all,"capabilities=compute,utility,graphics"' \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
        -v "$PWD":/root open3d-viewer:latest

Also see the `docker tutorial for ROS
<http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration>`__ for more
information.

Building Open3D in Docker
-------------------------

If your current system does not support the minimum system requirements for
building Open3D or if you have different versions of Open3D dependencies
installed, you can build Open3D from source in docker without interfering with
your system. This may be the case for older OS such as Ubuntu 16.04 or CentOS 7.
We provide docker build scripts and dockerfiles to build Python wheels in
various configurations. You can choose between different versions of Python
(3.7-3.9), hardware architectures (AMD64, ARM64, CUDA) and developer vs release
modes. Some sample configuration options available are shown below.

.. code-block:: bash

    cd docker

    ./docker_build.sh cuda_wheel_py38_dev  # Python 3.8, AMD64, CUDA with MKL, developer mode
    ./docker_build.sh openblas-amd64-py36  # Python 3.6, AMD64 with OpenBLAS instead of MKL, release mode
    ./docker_build.sh openblas-arm64-py37  # Python 3.7, ARM64 with OpenBLAS, release mode

Run ``./docker_build.sh`` without arguments to get a list of all available build
configurations.
