.. _headless_rendering:

Headless rendering
-------------------------------------

This tutorial explains how to render and save images from a terminal without any display device.

.. Note:: This feature is experimental; it was only tested with an Ubuntu 18.04 environment.

.. Note:: Although Ubuntu 16.04 is no longer supported for Open3D, additional instructions are under :ref:`headless_ubuntu1604`.

Install OSMesa
````````````````````````

To generate a headless context, it is necessary to install `OSMesa <https://www.mesa3d.org/osmesa.html>`_.

.. code-block:: shell

    $ sudo apt-get install libosmesa6-dev

.. _install_virtualenv:

Install virtualenv
````````````````````````

Create a virtual environment for Python.

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
- ``-DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF``: note that headless rendering only works with the **glew 2.1** and **glfw 3.3-dev** version.
  In most cases, these versions are not installed in vanilla Ubuntu systems.
  Use these CMake options to force to build glew 2.1 and glfw 3.3-dev from source included with Open3D.
- The Filament-based GUI implementation is not compatible with headless rendering, please set ``-DBUILD_GUI=OFF``.
- With ``-DBUILD_GUI=OFF`` webRTC support must also be disabled ``-DBUILD_WEBRTC=OFF``.

As a result, the cmake command is the following

.. code-block:: shell

    (py3env) $ cmake -DENABLE_HEADLESS_RENDERING=ON \
                     -DBUILD_GUI=OFF \
                     -DBUILD_WEBRTC=OFF \
                     -DUSE_SYSTEM_GLEW=OFF \
                     -DUSE_SYSTEM_GLFW=OFF \
                     ..

If cmake successfully generates makefiles, build Open3D.

.. code-block:: shell

    (py3env) $ make -j$(nproc)
    (py3env) $ make install-pip-package

.. _test_headless_rendering:

Test headless rendering
````````````````````````

As a final step, test a Python script that saves depth and surface normal sequences.

.. code-block:: shell

    (py3env) $ cd ~/Open3D/examples/python/visualization
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

Rendered images are at ~/Open3D/examples/test_data/depth and the image folder.

.. Note:: | ``headless_rendering.py`` saves png files.
          | This may take some time, so try to tweak the script for your purpose.

Possible Issues
````````````````````````

.. Error:: | If glew and glfw did not correctly link with OSMesa, it may crash with the following error.
           | **GLFW Error: X11: The DISPLAY environment variable is missing. Failed to initialize GLFW**

Try ``cmake`` with ``-DUSE_SYSTEM_GLEW=OFF`` and ``-DUSE_SYSTEM_GLFW=OFF`` flags.

.. Error:: | If OSMesa does not support GL 3.3 Core you will get the following error:
           | **GLFW Error: OSMesa: Failed to create context**

Open3D currently uses GL 3.3 Core Profile, if that is not supported you will get the above error.
You can run

.. code-block:: shell

    $ cd ~/Open3D/build
    $ bin/GLInfo

to get GL information for your environment (with or without a screen).
It will try and print various configurations, the second one is the one we use,
it should look something like

.. code-block:: shell

    [Open3D INFO] TryGLVersion: 3.3  GLFW_OPENGL_CORE_PROFILE
    [Open3D DEBUG] GL_VERSION:	3.3 (Core Profile) Mesa 19.2.8
    [Open3D DEBUG] GL_RENDERER:	llvmpipe (LLVM 9.0, 256 bits)
    [Open3D DEBUG] GL_VENDOR:	VMware, Inc.
    [Open3D DEBUG] GL_SHADING_LANGUAGE_VERSION:	3.30

If instead you get

.. code-block:: shell

    [Open3D INFO] TryGLVersion: 3.3  GLFW_OPENGL_CORE_PROFILE
    [Open3D WARNING] GLFW Error: OSMesa: Failed to create context
    [Open3D DEBUG] Failed to create window

Then your OSMesa version might be too old.  Try to follow instructions below to :ref:`compile_osmesa` to build a newer version and see if that resolves your issue.

.. _headless_ubuntu1604:

Headless Ubuntu 16.04
``````````````````````````````````````

For Ubuntu 16.04, a version of OSMesa needs to be built from source.
First follow :ref:`install_virtualenv` instructions above, then follow :ref:`compile_osmesa` instructions below.

.. _compile_osmesa:

Compile OSMesa from source
``````````````````````````````````````

Here are instructions for compiling mesa-19.0.8, last version that still supported ./configure:

.. code-block:: shell

    # install llvm-8
    (py3env) $ sudo apt install llvm-8

    # download OSMesa 19.0.8 release
    (py3env) $ curl -O https://mesa.freedesktop.org/archive/mesa-19.0.8.tar.xz
    (py3env) $ tar xvf mesa-19.0.8.tar.xz
    (py3env) $ cd mesa-19.0.8
    (py3env) $ LLVM_CONFIG="/usr/bin/llvm-config-8" ./configure --prefix=$HOME/osmesa \
        --disable-osmesa --disable-driglx-direct --disable-gbm --enable-dri \
        --with-gallium-drivers=swrast --enable-autotools --enable-llvm --enable-gallium-osmesa
    (py3env) $ make -j$(nproc)
    (py3env) $ make install
    # this installed OSMesa libraries to $HOME/osmesa/lib; in order for Open3D to pick it up
    # LD_LIBRARY_PATH needs to be updated to include it:
    (py3env) $ export LD_LIBRARY_PATH="$HOME/osmesa/lib:$LD_LIBRARY_PATH"
    # this needs to be done for every shell, or you can add it to your .bashrc
    (py3env) $ cd ~/Open3D
    (py3env) $ mkdir build&&cd build
    (py3env) $ cmake -DENABLE_HEADLESS_RENDERING=ON -DUSE_SYSTEM_GLEW=OFF -DUSE_SYSTEM_GLFW=OFF \
        -DOSMESA_INCLUDE_DIR=$HOME/osmesa/include -DOSMESA_LIBRARY="$HOME/osmesa/lib/libOSMesa.so" \
        ..
    (py3env) $ make -j$(nproc)
    (py3env) $ make install-pip-package

Now you can follow the instructions under :ref:`test_headless_rendering`.
