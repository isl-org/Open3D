.. _docker:

Docker
------

The following document describes a Docker CE based solution for utilizing Open3D.
Utilizing this approach the user can:

- sandbox Open3D from other applications on a machine
- operate Open3D on a headless machine using VNC or the terminal
- edit the Open3D code on the host side but run it inside an Open3D container

This recipe was developed and tested on Ubuntu 18.04. Other distributions might need slightly different instructions.

.. _docker_installation:

Ubuntu installation
===================

.. warning:: For the latest and most accurate installation guide see the official documentation at `Docker-CE install <https://docs.docker.com/install/>`_.

The prefered installation mode is to use the official `Docker repository <https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository>`_.

Install dependencies:

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

In order to address this problem see the :ref:`docker_proxy_settings_daemon` proxy settings.

**Optional steps**

`Docker Post-installation steps for Linux <https://docs.docker.com/install/linux/linux-postinstall>`_

**Add user to “docker” group**

This will eliminate the need to use sudo in order to run docker commands.

.. code-block:: sh

    $ sudo usermod -aG docker <user_name>

.. warning:: | The docker group grants privileges equivalent to the root user.
             | For details on how this impacts security in your system, see `Docker Daemon Attack Surface <https://docs.docker.com/engine/security/security/#docker-daemon-attack-surface>`_.

.. _docker_proxy_settings:

Proxy settings
==============

.. _docker_proxy_settings_daemon:

Docker daemon
`````````````

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

.. _docker_proxy_settings_container:

Docker container
````````````````

The docker container proxy settings must be set in order to enable features like `apt install` in your container.
This can be accomplished either in the Dockerfile itself or by using a global mechanism like `chameleonsocks`.

**Dockerfile**

Take a look at the first few lines of the example :download:`Dockerfile <../../_static/docker/Dockerfile>`:

.. literalinclude:: ../../_static/docker/Dockerfile
   :language: docker
   :lineno-start: 3
   :lines: 3-10
   :linenos:

Uncommenting and settings these environment variables does the job.

**chameleonsocks**

Another way to set the proxy settings of the container is to use `chameleonsocks <https://github.com/crops/chameleonsocks>`_ which is a mechanism to redirect SOCKS or HTTP proxies.

`chameleonsocks` has the advantage that the Dockerfile doesn't need changing based on network configuration circumstances.
However it can conflict with other proxy settings on your host system. When that happens stopping the `chameleonsocks` container addresses the conflicts.

.. code-block:: sh

    $ cd ~
    $ mkdir chameleonsocks
    $ cd chameleonsocks
    $ https_proxy=<https://server:port \
      wget https://raw.githubusercontent.com/crops/chameleonsocks/master/chameleonsocks.sh && \
      chmod 755 chameleonsocks.sh

In order to properly set the proxy settings edit the proxy settings inside the `chameleonsocks.sh` file:

.. code-block:: sh

    # This is the domain name or ip address of your proxy server. It must
    # be defined or nothing will work. Do not specify a protocol type
    # such as http:// or https://.
    : ${PROXY:=server}
    # This is the port number of your proxy server
    : ${PORT:=port}
    # Possible PROXY_TYPE values: socks4, socks5, http-connect, http-relay
    : ${PROXY_TYPE:=socks5}
    # a file containing local company specific exceptions
    : ${EXCEPTIONS:=chameleon.exceptions}
    # Autoproxy url, this is often something like
    # http://autoproxy.server.com or http://wpad.server.com/wpad.out
    # ONLY additional exceptions are pulled from here. not the proxy
    : ${PAC_URL=http://wpad.server.com/wpad.dat}

Create the `chameleon.exceptions` file:

.. code-block:: sh

    $ touch chameleon.exceptions
    $ ne chameleon.exceptions

    server.com
    .server.com
    localhost
    127.0.0.1

Install `chameleonsocks`:

.. code-block:: sh

    $ ./chameleonsocks.sh --install

NOTE: the `chameleonsocks.exceptions` file disappears after the install step above.

DNS servers
```````````

In order to specify `DNS servers for docker <https://docs.docker.com/install/linux/linux-postinstall/#specify-dns-servers-for-docker>`_
edit ``/etc/docker/daemon.json`` on the host:

.. code-block:: sh

    $ sudo ne /etc/docker/daemon.json
    {
        "dns": ["xxx.xxx.xxx.xxx", "xxx.xxx.xxx.xxx"]
    }

.. _docker_usage_notes:

Usage notes
===========

.. _docker_files:

Docker files
````````````

The Docker files can be found under ``Open3D/util/docker/open3d-xvfb``::

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

.. _docker_remote_access:

Remote access
`````````````

VNC can be used to remote into a running docker container.

A running Open3D container listens to port 5920 on the host.
The ``run.sh`` script redirects host port 5920 to container port 5900.

This allows remoting into the container using VNC to ``<host ip>:5920``.
The default password is ``1234`` and can be changed in ``Open3D/issue_17/util/docker/open3d-xvfb/setup/entrypoint.sh`` (requires rebuilding the Open3D Docker image with ``build.sh``).
Once connected you can use Open3D as usual.

.. _docker_host_terminal:

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

.. _docker_limitations:

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

