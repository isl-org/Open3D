.. _docker-test:

Docker test
-----------

A number of Docker images have been designed for the purpose of testing Open3D.
These configurations are based on various Ubuntu versions, Python installations
and number of preinstalled Open3D dependencies and don't support a GUI.

Ubuntu:

- 14.04
- 16.04
- 18.04

Python installations:

- py2: python2.x
- py3: python3.x
- mc2: miniconda2
- mc3: miniconda3

Preinstalled dependencies (bundle type):

- base: minimal, dependencies will be built from source
- deps: all dependencies are installed from packages

The Python version will differ from one image to another based on the defaults for that configuration.

The images can be found at: `intelvcl/open3d-test <https://hub.docker.com/r/intelvcl/open3d-test>`_

.. _docker-test-usage-notes:

Usage notes
===========

The docker setup files and tools can be found under: `Open3d/util/docker/open3d-test <https://github.com/intel-isl/Open3D/tree/docker/util/docker/open3d-test>`_.

.. _docker-test-setup:

Setup files
```````````

The setup files are used, along with the dockerfiles, to build the images.

.. _docker-test-tools:

Tools
`````

The tools are scripts designed to:

- build the docker images
- upload the images to the online repository
- test Open3D
- delete the images from the system
- stop running containers

Most scripts, with the exception of those that are meant for all
configurations, accept the following arguments:

- Ubuntu version:   14.04 16.04 18.04
- Bundle type:      base deps
- Environment type: py2 py3 mc2 mc3
- Link type:        STATIC SHARED

The first two arguments, the Ubuntu version and the bundle type, are always required.
The last two arguments, the Python installation (aka environment type) and the library link type are only required by the test.sh script.

The scripts are designed to display help on the arguments if an error is detected.

**build.sh**

This script will build a single Open3D docker image based on the Ubuntu version, bundle type and Python version.

**build-all.sh**

This script will build all of the supported Open3D docker images.

**upload.sh**

This script will upload a single Open3D docker image. Accepts the same command line arguments as build.sh.

Requires docker login:

.. code-block:: sh

    $ docker login --username intelvcl

**upload-all.sh**

This script will upload all of the Open3D docker images.

Requires docker login:

.. code-block:: sh

    $ docker login --username intelvcl

**cleanup.sh**

This script will remove all of the Open3D docker images, if any is found on the system.

**prune.sh**

This script is useful at design time and helps remove any unfinished docker images.

**test.sh**

This script tests a single Open3D configuration.
Requires all four command line arguments.

**test-all.sh**

This script tests all Open3D test configurations.

**run.sh**

This script is useful for interacting with Open3D under a specific configuration.
Accepts the same command line arguments as build.sh.

**stop.sh**

This script stops a single Open3D container.

**stop-all.sh**

This script stops all, if any was found, Open3D containers.
