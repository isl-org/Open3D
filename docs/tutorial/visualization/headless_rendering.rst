.. _headless_rendering:

Headless rendering
------------------

This tutorial shows how to render and save images from a terminal with no
display attached, using ``open3d.visualization.Visualizer``.

On Linux, ``Visualizer.create_window()`` checks whether a windowing system
display (``DISPLAY`` or ``WAYLAND_DISPLAY``) is available. If none is found,
it transparently creates a GPU-accelerated offscreen rendering context with
`EGL <https://www.khronos.org/egl>`_ instead of a visible GLFW window. No
special build flag or environment variable is needed; the same Open3D binary
and Python wheel support both windowed and headless use. If no GPU is
available, this falls back to software rendering, as long as Mesa's EGL
implementation is installed. On Windows and macOS, the visualizer still
requires an active desktop/window-server session and a hidden window.

Install EGL
```````````

Most Linux systems with a GPU driver already provide EGL. On systems without
a GPU (e.g. servers or Docker containers), install Mesa's software EGL
implementation:

.. code-block:: shell

    $ sudo apt-get install libegl1

Test headless rendering
````````````````````````

Run an example script that renders a camera trajectory and saves depth and
color images, with no display attached (e.g. over SSH without X forwarding,
or on a headless CI runner):

.. code-block:: shell

    $ cd ~/Open3D/examples/python/visualization
    $ env -u DISPLAY -u WAYLAND_DISPLAY python headless_rendering.py

This should print output similar to:

.. code-block:: shell

    Customized visualization playing a camera trajectory. Press ctrl+z to terminate.
    Saving color images in ~/Open3D/examples/python/visualization/HeadlessRenderingOutput/image
    Saving depth images in ~/Open3D/examples/python/visualization/HeadlessRenderingOutput/depth
    Capture image 00000
    Capture image 00001
    Capture image 00002
    :
    Capture image 00030

.. Note:: | Rendered images are saved as PNG files under
          | ``HeadlessRenderingOutput/image`` and ``HeadlessRenderingOutput/depth``,
          | relative to the current working directory. Rendering many frames can
          | take a while; adjust the script to your needs.

Possible issues
````````````````

To check whether your environment can create an offscreen GPU context, and
which GPU/driver is used, run the ``GLInfo`` tool with no display attached:

.. code-block:: shell

    $ cd ~/Open3D/build
    $ env -u DISPLAY -u WAYLAND_DISPLAY bin/GLInfo

On a machine with a working EGL driver, this prints something like:

.. code-block:: shell

    [Open3D INFO] EGLOffscreenContext: created 640x480 offscreen GPU context (EGL 1.5, vendor: NVIDIA).
    GL_VERSION:     3.3.0 NVIDIA 550.54.14
    GL_RENDERER:    NVIDIA GeForce RTX 3080/PCIe/SSE2
    GL_VENDOR:      NVIDIA Corporation

.. Warning:: | If no EGL-capable GPU driver is found, ``Visualizer::CreateVisualizerWindow()``
             | logs: **Failed to create EGL offscreen context. Headless rendering requires an
             EGL-capable GPU driver (e.g. Mesa or NVIDIA).**

In that case, verify that a GPU and its driver (with EGL support) are visible
inside your container/VM (e.g. with ``nvidia-smi``, or with ``eglinfo`` from
the ``mesa-utils`` package), and that ``libegl1`` is installed.
