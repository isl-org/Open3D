.. _headless_rendering:

Headless Rendering
------------------

Docker installation
===================

The prefered installation mode is using the official Docker repository.

https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository
https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

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

Proxy settings
``````````````

https://docs.docker.com/config/daemon/systemd/#httphttps-proxy

Solves the hello-world issue above.

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

https://docs.docker.com/install/linux/linux-postinstall/#specify-dns-servers-for-docker

On the host:

.. code-block:: sh

    $ sudo ne /etc/docker/daemon.json
    {
        "dns": ["10.248.2.1", "10.22.224.196", "10.3.86.116"]
    }

Add user to “docker” group
``````````````````````````

This will eliminate the need to use sudo in order to run docker commands.

.. code-block:: sh

    $ sudo usermod -aG docker <user_name>

https://docs.docker.com/install/linux/linux-postinstall/

``Warning``
````````````
The docker group grants privileges equivalent to the root user.
For details on how this impacts security in your system, see
`Docker Daemon Attack Surface <https://docs.docker.com/engine/security/security/#docker-daemon-attack-surface>`_.

Usage notes
===========

- clone my branch
- $ cd <Open3D path>/utilities/docker/ubuntu-xvfb/tools
- $ ./build.sh to build the image
- $ ./attach.sh to clone/build the master Open3D repo and start the Open3D container
- $ ./headless_sample.sh to run the sample that renders some images and saves them to disk. The sample will render some images which can be accessed on the host at ~/Open3D_docker. These files don't go away if the container is stopped.

Notes:

- the sample will not return. Ctrl+c to exit. Need to update the sample to exit on it's own.
- the cloning is done for now to ~/Open3D_docker. We have options: find a way to let the user specify the destination or just reuse the current location of Open3D.
- TODO: uncomment entries in the dependencies section inside the Dockerfile.

