.. _headless_rendering:

Headless rendering
-------------------------------------

This tutorial explains how to render and save images from a terminal without any display device,
using the legacy ``open3d.visualization.Visualizer``.

On Linux, ``Visualizer.create_window()`` automatically detects whether a windowing system display
(``DISPLAY`` or ``WAYLAND_DISPLAY``) is available. If none is found, it transparently falls back to
a GPU-accelerated offscreen rendering context created with `EGL <https://www.khronos.org/egl>`_,
instead of creating a visible GLFW window. This works with the **standard Open3D binary
(Python wheel or C++ shared library)** - no special build flags or separate headless build are
required.

.. Note:: Headless rendering requires a GPU with EGL support (e.g. Mesa on Linux with a supported
          GPU, or NVIDIA's EGL driver). It is only supported on Linux; on Windows and macOS,
          offscreen rendering with the legacy visualizer instead relies on a hidden window and
          requires an active desktop/window-server session.

Install EGL
````````````````````````

Most Linux systems with a GPU driver already have EGL available. If needed, install the
development package:

.. code-block:: shell

    $ sudo apt-get install libegl1-mesa-dev

Test headless rendering
````````````````````````

Test with a Python script that saves depth and surface normal sequences, without a display
attached (e.g. over SSH without X forwarding, or on a headless CI runner):

.. code-block:: shell

    $ cd ~/Open3D/examples/python/visualization
    $ env -u DISPLAY -u WAYLAND_DISPLAY python headless_rendering.py

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

You can check whether your environment can create an offscreen GPU context, and which GPU/driver
is used, with:

.. code-block:: shell

    $ cd ~/Open3D/build
    $ env -u DISPLAY -u WAYLAND_DISPLAY bin/GLInfo

It should print something like:

.. code-block:: shell

    [Open3D INFO] GLInfo: using EGL 1.5 offscreen context
    GL_VERSION:     3.3 (Core Profile) Mesa 23.2.1
    GL_RENDERER:    NVIDIA GeForce RTX 3080/PCIe/SSE2
    GL_VENDOR:      NVIDIA Corporation

.. Error:: | If no EGL-capable GPU driver is available, ``Visualizer::CreateVisualizerWindow()``
           | will log: **Failed to create EGL offscreen context. Headless rendering requires an
           EGL-capable GPU driver (e.g. Mesa or NVIDIA).**

In that case, verify that a GPU and its driver (with EGL support) are visible inside your
container/VM (e.g. ``nvidia-smi``, or ``eglinfo`` from the ``mesa-utils`` package), and that
``libegl1`` is installed.
