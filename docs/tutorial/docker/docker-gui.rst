.. _docker-gui:

Docker GUI
----------

The following document describes a solution for utilizing Open3D with GUI/visualization under Docker.

Utilizing this approach the user can:

- sandbox Open3D from other applications on a machine
- operate Open3D on a headless machine using VNC or the terminal
- edit the Open3D code on the host side but run it inside an Open3D container

This recipe was developed and tested on Ubuntu 18.04. Other distributions might need slightly different instructions.

.. _docker-gui-usage-notes:

Usage notes
===========

.. _docker-gui-files:

Docker files
````````````

The docker setup files and tools can be found under `Open3d/util/docker/open3d-xvfb <https://github.com/intel-isl/Open3D/tree/docker/util/docker/open3d-xvfb>`_.

    - Dockerfile
    - setup
        - entrypoint.s
        - docker_sample.sh
    - tools
        - attach.sh
        - build.sh
        - delete.sh
        - name.sh
        - prune.sh
        - run.sh
        - stop.sh

**Dockerfile**

``Dockerfile`` is the Docker script used to build the Open3D image.

**Tools**

We provide a number of Docker tools for convenience:

- ``attach.sh``
  Attach a terminal to a running Open3D docker container.
- ``build.sh``
  Build the Open3D docker image.
- ``delete.sh``
  Delete the Open3D image.
- ``prune.sh``
  Delete **all** stopped containers and **all** dangling images.
- ``run.sh``
  Run the Open3D container. Checkout Open3D, build and install.
- ``stop.sh``
  Stop the Open3D container.

A typical flow of events when working with docker looks like this:

.. code-block:: sh

    $ cd <open3d_path>/util/docker/open3d_xvfb/tools
    $ ./build.sh
    $ ./run.sh
    root@open3d-xvfb:~/open3d/build# <do some stuff>
    root@open3d-xvfb:~/open3d/build# exit
    $ ./attach.sh
    root@open3d-xvfb:~/open3d# <do some stuff>
    root@open3d-xvfb:~/open3d# exit
    $ ./stop.sh

Building the Open3D Docker image will take approximately 10-15 minutes to complete.
At the end the image will be ~1GB in size.

Running the Open3D Docker container will perform the following steps:

- git clone Open3D master to ``~/open3d_docker``
- run the container in interactive mode with the host path ``~/open3d_docker`` mounted inside the container at ``/root/Open3D``
- build Open3D and install inside the docker container.
- attach a terminal to the Open3D container for command line input from the host side

In order to disconnect from a running container type ``exit`` at the terminal.
You can still attach to a running container at a later time.

The Open3D container is automatically removed when stopped.
None of the Open3D files are removed as they in fact reside on the host due to the Docker bind mounting functionality.
In order to keep the container around (and not have to rebuild the Open3D binaries every time) remove the ``-rm`` option in ``run.sh``.

Prunning images/containers is useful when modifying/testing a new image.
It cleans up the docker workspace and frees up disk space.

.. _docker-gui-remote-access:

Remote access
`````````````

VNC can be used to remote into a running docker container.

A running Open3D container listens to port 5920 on the host.
The ``run.sh`` script redirects host port 5920 to container port 5900.

This allows remoting into the container using VNC to ``<host ip>:5920``.
The default password is ``1234`` and can be changed in ``Open3D/issue_17/util/docker/open3d-xvfb/setup/entrypoint.sh`` (requires rebuilding the Open3D Docker image with ``build.sh``).
Once connected you can use Open3D as usual.

.. _docker-gui-host-terminal:

Running at the host terminal
````````````````````````````

It is also possible to run Open3D from a host terminal attached to a running Open3D Docker container.
An example on how this can be perfomed:

.. code-block:: sh

    # at the host terminal
    $ sudo cp ~/open3d_docker/util/docker/open3d-xvfb/setup/docker_sample.sh \
              ~/open3d_docker/build/lib/Tutorial/Advanced
    $ cd ~/open3d_docker/utilities/docker/open3d-xvfb/tools
    $ ./attach.sh

    # at the container terminal
    $ cd ~/open3d/build/lib/Tutorial/Advanced
    $ sh docker_sample.sh

.. _docker-gui-limitations:

Limitations
===========

- | the ``lxde`` user interface needs more configuring.
  | Some things won't work as expected. For example the ``UXTerm`` doesn't start and ``lxterminal`` may crash occasionally.
- | the container screen resolution is set to 1280x1024x8.
  | The resolution will be increased in the future.
- | there are some rendering issues.
  | Some images may be saved incorrectly to the disk. For example, when running the ``headless_sample.py`` sample the color images saved to the disk are black.
- | ``run.sh`` clones Open3D to a hardcoded location: ``~/open3d_docker``
  | We are considering the following alternatives:

    - let the user specify the destination
    - reuse the current location of Open3D

