.. _headless_rendering:

Headless rendering
-------------------------------------

This tutorial explains how to render and save images from a terminal without any display device.

.. Note:: This feature is experimental; it was only tested with Ubuntu 16.04 environment.

Install OSMesa
````````````````````````

To generate a headless context, it is necessary to install `OSMesa <https://www.mesa3d.org/osmesa.html>`_.

.. code-block:: shell

    $ sudo apt-get install libosmesa6-dev

Otherwise, a recent version of OSMesa can be built from source.

.. code-block:: shell

    # download OSMesa 2018 release
    $ cd
    $ wget ftp://ftp.freedesktop.org/pub/mesa/mesa-18.0.0.tar.gz
    $ tar xf mesa-18.0.0.tar.gz
    $ cd mesa-18.0.0/

    # configure compile option and build
    $ ./configure --enable-osmesa --disable-driglx-direct --disable-gbm --enable-dri --with-gallium-drivers=swrast
    $ make

    # add OSMesa to local path.
    $ export PATH=$PATH:~/mesa-18.0.0

.. _install_virtualenv:

Install virtualenv
````````````````````````

The next step is to make a virtual environment for Python.

.. code-block:: shell

    $ sudo apt-get install virtualenv python-pip
    $ virtualenv -p /usr/bin/python3 py3env
    $ source py3env/bin/activate
    (py3env) $ pip install numpy matplotlib

This script installs and activates ``py3env``. The necessary modules, ``numpy`` and ``matplotlib``, are installed in ``py3env``.

.. Error:: Anaconda users are recommended to use this configuration as ``conda install matplotlib`` installs additional modules that are not based on OSMesa.
           This will result in **segmentation fault error** at runtime.


Build Open3D with OSMesa
````````````````````````

Let's move to build a folder.

.. code-block:: shell

    (py3env) $ cd ~/Open3D/
    (py3env) $ mkdir build && cd build

In the next step, there are two cmake flags that need to be specified.

- ``-DENABLE_HEADLESS_RENDERING=ON``: this flag informs glew and glfw should use **OSMesa**.
- ``-DBUILD_GLEW=ON -DBUILD_GLFW=ON``: note that headless rendering only works with the **glew 2.1** and **glfw 3.3-dev** version.
  In most cases, these versions are not installed in vanilla Ubuntu systems.
  Use these CMake options to force to build glew 2.1 and glfw 3.3-dev from source.

As a result, the cmake command is the following

.. code-block:: shell

    (py3env) $ cmake -DENABLE_HEADLESS_RENDERING=ON \
                     -DBUILD_GLEW=ON \
                     -DBUILD_GLFW=ON \
                     -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 \
                     ..

Note that ``-DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3`` is the same path that was used for :ref:`install_virtualenv`.

If cmake successfully generates makefiles, build Open3D.

.. code-block:: shell

    (py3env) $ make # or make -j in multi-core machine



Test headless rendering
````````````````````````

As a final step, test a Python script that saves depth and surface normal sequences.

.. code-block:: shell

    (py3env) $ cd ~/Open3D/build/lib/Tutorial/Advanced/
    (py3env) $ python headless_rendering.py

This should print the following:

.. code-block:: shell

    Capture image 00000
    Capture image 00001
    Capture image 00002
    Capture image 00003
    Capture image 00004
    Capture image 00005
    :
    Capture image 00030

Rendered images are at ~/Open3D/build/lib/TestData/depth and the image folder.

.. Note:: | ``headless_rendering.py`` saves png files.
          | This may take some time, so try to tweak the script for your purpose.

.. Error:: | If glew and glfw did not correctly link with OSMesa, it may crash with the following error.
           | **GLFW Error: X11: The DISPLAY environment variable is missing. Failed to initialize GLFW**
           | Try ``cmake`` with ``-DBUILD_GLEW=ON`` and ``-DBUILD_GLFW=ON`` flags.
