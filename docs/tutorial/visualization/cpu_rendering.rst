.. _cpu_rendering:

CPU (Software) Rendering
========================

Open3D supports CPU or software rendering on Linux when a GPU is unavailable
or GPU rendering is not suitable. Software rendering is slower and less
responsive than GPU rendering.

On Linux and Windows, the Filament-based renderer uses Vulkan by default. When
Vulkan reports a CPU device such as Mesa's llvmpipe, Open3D automatically uses
Filament's OpenGL backend instead of its Vulkan backend. This avoids a known
llvmpipe crash in Filament's Vulkan path.

Select Software Rendering
--------------------------

Set ``VK_DRIVER_FILES`` before starting Open3D to select Mesa's llvmpipe Vulkan
driver. The path to ``lvp_icd.json`` may differ across distributions.

For an interactive application:

.. code-block:: bash

    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json Open3D

For Python, set the variable before importing ``open3d``:

.. code-block:: python

    import os
    os.environ['VK_DRIVER_FILES'] = '/usr/share/vulkan/icd.d/lvp_icd.json'
    import open3d as o3d

Headless or Offscreen Rendering
-------------------------------

For headless or offscreen rendering, also set ``EGL_PLATFORM=surfaceless``.
This variable is optional when a display server is available.

.. code-block:: bash

    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
        EGL_PLATFORM=surfaceless \
        python examples/python/visualization/render_to_image.py

The ``VK_DRIVER_FILES`` path is handled by the Vulkan loader during
initialization. If the Mesa ICD is installed elsewhere, use the path to that
system's ``lvp_icd.json`` file.
