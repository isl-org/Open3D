.. _docker:

Docker
------

The following document describes a Docker CE based solution for headless rendering.

This recipe was developed and tested on Ubuntu 16.04. Other distributions might need slightly different instructions.

Docker installation
===================

The prefered installation mode is to use the official `Docker repository <https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository>`_.

Install dependencies

.. code-block:: sh

    $ sudo apt update && \
      sudo apt upgrade -y && \
      sudo apt install -y \
           curl \
           apt-transport-https \
           ca-certificates \
           software-properties-common

Add Docker’s official GPG key:

.. code-block:: sh

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

Manually verify that the key is ``9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88``:

.. code-block:: sh

    $ sudo apt-key fingerprint 0EBFCD88

Set up the stable repository:

.. code-block:: sh

    $ sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"

    $ sudo apt update

Install the latest version:

.. code-block:: sh

    $ sudo apt install -y docker-ce

Verify the installation:

.. code-block:: sh

    $ sudo docker run hello-world

The verification will timeout if running behind a proxy:
``Unable to find image 'hello-world:latest' locally``

Optional steps
==============

`Docker Post-installation steps for Linux <https://docs.docker.com/install/linux/linux-postinstall>`_

Proxy settings
``````````````

In order to solve the hello-world issue above follow the `Docker proxy settings <https://docs.docker.com/config/daemon/systemd/#httphttps-proxy>`_.

Create a systemd drop-in directory for the docker service:

.. code-block:: sh

    $ sudo mkdir -p /etc/systemd/system/docker.service.d

    $ sudo touch /etc/systemd/system/docker.service.d/http-proxy.conf
    $ sudo ne /etc/systemd/system/docker.service.d/http-proxy.conf
    [Service]
    Environment="HTTP_PROXY=server:port" "NO_PROXY=localhost;127.0.0.1"

    $ sudo touch /etc/systemd/system/docker.service.d/https-proxy.conf
    $ sudo ne /etc/systemd/system/docker.service.d/https-proxy.conf
    [Service]
    Environment="HTTPS_PROXY=server:port/" "NO_PROXY=localhost;127.0.0.1"

Flush changes and restart Docker:

.. code-block:: sh

    $ sudo systemctl daemon-reload
    $ sudo systemctl restart docker


Verify that the configuration has been loaded:

.. code-block:: sh

    $ systemctl show --property=Environment docker


DNS servers
```````````

In order to specify `DNS servers for docker <https://docs.docker.com/install/linux/linux-postinstall/#specify-dns-servers-for-docker>`_
edit ``/etc/docker/daemon.json`` on the host:

.. code-block:: sh

    $ sudo ne /etc/docker/daemon.json
    {
        "dns": ["xxx.xxx.xxx.xxx", "xxx.xxx.xxx.xxx"]
    }

Add user to “docker” group
``````````````````````````

This will eliminate the need to use sudo in order to run docker commands.

.. code-block:: sh

    $ sudo usermod -aG docker <user_name>

``Warning``
````````````
The docker group grants privileges equivalent to the root user.
For details on how this impacts security in your system, see
`Docker Daemon Attack Surface <https://docs.docker.com/engine/security/security/#docker-daemon-attack-surface>`_.

Usage notes
===========

Docker files
````````````````

The Docker files can be found under ``Open3D/util/docker/ubuntu-xvfb``::

    - Dockerfile
    - setup
        - build.sh
        - entrypoint.s
        - headless_sample.py
        - headless_sample.sh
    - tools
        - attach.sh
        - build.sh
        - delete.sh
        - it.sh
        - prune.sh
        - run.sh
        - stop.sh

Dockerfile
++++++++++

``Dockerfile`` is the Docker script used to build the Open3D image.

Tools
+++++

We provide a number of Docker tools for convenience:

- ``attach.sh``
  Start the Open3D docker container and attach to it using a terminal interface.
- ``build.sh``
  Build the Open3D docker image.
- ``delete.sh``
  Delete the Open3D image.
- ``it.sh``
  Start the Open3D docker container and display container stdout.
- ``prune.sh``
  Delete hanging containers and images.
- ``run.sh``
  Run the Open3D container.
- ``stop.sh``
  Stop the Open3D container.

Building the Open3D Docker image will take approximately 10-15 minutes to complete.
At the end the image will be ~2.1 GB in size.

Running the Open3D Docker container will perform the following steps:

- git clone Open3D master to ``~/Open3D_docker``
- copy the ``headless_sample.py`` and ``headless_sample.sh`` to ``~/Open3D_docker/build/lib/Tutorial/Advanced``
- run and detach the Open3D container with the host path ``~/Open3D_docker`` mounted inside the container at ``/root/Open3D``
- attach a terminal to the Open3D container for command line input from the host side

The Open3D container is automatically removed when stopped.
None of the Open3D files are removed as they in fact reside on the host due to the Docker bind mounting functionality.
In order to keep the container around remove the ``-rm`` option in ``it.sh`` and/or ``run.sh``.

Prunning images/containers is useful when modifying/testing a new image.

VNC
```
A running Open3D container listens to port 5920 on the host.
The ``it.sh``, ``run.sh`` and ``attach.sh`` scripts redirect host port 5920 to container port 5900.

This allows remoting into the container using VNC to ``<host ip>:5920``. Once connected you can use Open3D as usual.

Headless rendering in terminal
``````````````````````````````

Sometimes it may be necessary to perform rendering as part of some script automation.
In order to do this follow the next steps::

$ cd <Open3D path>/utilities/docker/ubuntu-xvfb/tools
$ ./build.sh
$ ./attach.sh
$ ./headless_sample.sh

The ``headless_sample.sh`` renders some images and saves them to disk.
The images can be accessed in real time on the host at ``~/Open3D_docker/build/lib/TestData/depth`` and won't go away when the container is stopped/deleted.

Limitations
```````````

- the ``lxde`` based interface employed in this Docker image needs more configuring.
  Some things won't work as expected. For example ``lxterminal`` crashes.
- the resolution is set to 1280x1024x8 when remoting into an Open3D container.
  Open3D windows are larger than this. The resolution will be increased in the future.
- the ``headless_sample.py`` sample does not return as it expects GUI user input.
  The sample will be redesigned in the future.
- for now running the Open3D docker container clones Open3D master to ``~/Open3D_docker``.
  We are considering the following options:

    - let the user specify the destination
    - reuse the current location of Open3D.

