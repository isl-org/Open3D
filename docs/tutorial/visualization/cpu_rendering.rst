.. _cpu_rendering:

CPU (Software) Rendering
========================

Open3D's new visualization functionality (:class:`.O3DVisualizer` class,
:func:`.draw()` function and :mod:`open3d.visualization.gui` and
:mod:`open3d.visualization.rendering` modules) requires a recent GPU with
support for OpenGL 4.1 or higher. This is not available in certain situations:

    - GPU is too old to support OpenGL 4.1.
    - No GPU is available, for example on cloud servers that do not have any GPU
      (integrated or discrete) installed, or in a docker container that does not
      have access to the host GPU. This is often the case for many cloud based
      Jupyter notebooks such as Google Colab, Kaggle, etc.
    - A GPU is available, but it only supports computation, not graphics. This
      is a common scenario for cloud based Jupyter notebooks deployed in docker
      containers.

Open3D supports CPU or software rendering in such situations. Note that this
usually produces slower and less responsive rendering, so a GPU is recommended.
Currently, this is available only for Linux. There are two separate ways to
use CPU rendering depending on whether interactive or headless rendering is
desired. Both methods are described below.

Headless CPU Rendering
----------------------

For Python code, you can enable CPU rendering for headless rendering when using
the :class: `.OffscreenRenderer` for a process by setting the environment
variable ``OPEN3D_CPU_RENDERING=true`` before importing Open3D. Here are the
different ways to do that:

.. code:: bash

    # from the command line
    OPEN3D_CPU_RENDERING=true python
    examples/python/visualization/render_to_image.py

.. code:: python

    # In Python code
    import os
    os.environ['OPEN3D_CPU_RENDERING'] = 'true'
    import open3d as o3d

    # In a Jupyter notebook
    %env OPEN3D_CPU_RENDERING true
    import open3d as o3d

.. note:: Seeting the environment variable after importing ``open3d`` will not work,
    even if ``open3d`` is re-imported. In this case, if no usable GPU is present, the
    Python interpreter or Jupyter kernel will crash when visualization functions are
    used.

.. note:: This method will **not** work for interactive rendering scripts such
   as ``examples/python/visualization/draw.py``. For interactive rendering see
   the next section.

Interactive CPU Rendering
-------------------------

The method for enabling interactive CPU rendering depends on your system:

1.  **You use Mesa drivers v20.2 or higher.** This is the case for all
    Intel GPUs and some AMD and Nvidia GPUs. You should be running a recent Linux
    OS, such as Ubuntu 20.04. Check your Mesa version from your package manager
    (e.g. run ``dpkg -s libglx-mesa0 | grep Version`` in Debian or Ubuntu). In this
    case, you can switch to CPU rendering by simply setting an environment
    variable before starting your application. For example, start the Open3D
    visualizer app in CPU rendering mode with:

    .. code:: bash

        LIBGL_ALWAYS_SOFTWARE=true Open3D

    Or for Python code:

    .. code:: bash

        LIBGL_ALWAYS_SOFTWARE=true python examples/python/visualization/draw.py

.. note:: Mesa drivers must be in use for this method to work; just having
   them installed is not sufficient. You can check the drivers in use with the
   ``glxinfo`` command.

2.  **You use Nvidia or AMD drivers or old Mesa drivers (< v20.2).**  We provide
    the Mesa software rendering library binary for download `here
    <https://github.com/isl-org/open3d_downloads/releases/download/mesa-libgl/mesa_libGL_22.0.tar.xz>`__.
    This is automatically downloaded to
    `build/_deps/download_mesa_libgl-src/libGL.so.1.5.0` when you build Open3D
    from source. If you want to use CPU rendering all the time, install this
    library to ``/usr/local/lib`` or ``$HOME/.local/lib`` and *prepend* it to your
    ``LD_LIBRARY_PATH``:

    .. code:: bash

        export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH

    For occasional use, you can instead launch a program with CPU rendering with:

    .. code:: bash

        LD_PRELOAD=$HOME/.local/lib/libGL.so.1.5.0 Open3D

    Or with Python code:

    .. code:: bash

        LD_PRELOAD=$HOME/.local/lib/libGL.so.1.5.0 python
        examples/python/visualization/draw.py
