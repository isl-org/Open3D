.. _getting_started:

Getting started
###############

.. _install_open3d_python:

Viewer
======

Use the Open3D viewer application to visualize 3D data in various formats and
interact with it.  You can download the latest stable release app from `Github
releases <https://github.com/isl-org/Open3D/releases>`__. The latest development
version (``HEAD`` of ``main`` branch) viewer app is provided here [#]_:

* `Linux (Ubuntu 22.04+ or glibc 2.35+) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-viewer-@OPEN3D_VERSION_FULL@-Linux.deb>`__ [#]_
* `MacOSX v10.15+ (Intel or Apple Silicon) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-app-macosx-10_15-universal2.zip>`__
* `Windows 10+ (64-bit) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-app-windows-amd64.zip>`__

.. [#] Please use these links from the `latest version of this page <https://www.open3d.org/docs/latest/getting_started.html>`__ only.
.. [#] To check the `glibc` version on your system, run :code:`ldd --version`.

Python
======

Open3D Python packages are distributed via
`PyPI <https://pypi.org/project/open3d/>`_.

Supported Python versions:

* 3.10
* 3.11
* 3.12
* 3.13

Supported operating systems:

* Ubuntu 22.04+
* macOS 10.15+
* Windows 10+ (64-bit)

If you have other Python versions or operating systems, please refer to
:ref:`compilation` and compile Open3D from source.

Pip (PyPI)
----------

.. code-block:: bash

    pip install open3d        # or
    pip install open3d-cpu    # Smaller CPU only wheel on x86_64 Linux (since v0.17+)

.. warning::

   Versions of ``numpy>=2.0.0`` require ``Open3D>0.18.0`` or the latest development
   version of Open3D. If you are using an older version of Open3D, downgrade ``numpy``
   with

   .. code-block:: bash

        pip install "numpy<2.0.0"

.. warning::

   Please upgrade your ``pip`` to a version >=20.3 to install Open3D in Linux,
   e.g. with

   .. code-block:: bash

        pip install -U "pip>=20.3"

.. note::
    In general, we recommend using a
    `virtual environment <https://docs.python-guide.org/dev/virtualenvs/>`_
    or `conda environment <https://docs.conda.io/en/latest/miniconda.html>`_.
    Otherwise, depending on the configurations, you may need ``pip3``  for
    Python 3, or the ``--user`` option to avoid permission issues. For example:

    .. code-block:: bash

        pip3 install open3d
        # or
        pip install --user open3d
        # or
        python3 -m pip install --user open3d

Development version (pip)
-------------------------

To test the latest features in Open3D, download and install the development
version (``HEAD`` of ``main`` branch):

.. list-table::
    :stub-columns: 1
    :widths: auto

    * - Linux
      - `Python 3.10 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp310-cp310-manylinux_2_35_x86_64.whl>`__
      - `Python 3.11 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp311-cp311-manylinux_2_35_x86_64.whl>`__
      - `Python 3.12 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp312-cp312-manylinux_2_35_x86_64.whl>`__
      - `Python 3.13 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp313-cp313-manylinux_2_35_x86_64.whl>`__

    * - Linux (CPU)
      - `Python 3.10 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_cpu-@OPEN3D_VERSION_FULL@-cp310-cp310-manylinux_2_35_x86_64.whl>`__
      - `Python 3.11 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_cpu-@OPEN3D_VERSION_FULL@-cp311-cp311-manylinux_2_35_x86_64.whl>`__
      - `Python 3.12 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_cpu-@OPEN3D_VERSION_FULL@-cp312-cp312-manylinux_2_35_x86_64.whl>`__
      - `Python 3.13 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_cpu-@OPEN3D_VERSION_FULL@-cp313-cp313-manylinux_2_35_x86_64.whl>`__

    * - MacOS
      - `Python 3.10 (x86_64+arm64) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp310-cp310-macosx_11_0_universal2.whl>`__
      - `Python 3.11 (x86_64+arm64) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp311-cp311-macosx_10_15_universal2.whl>`__
      - `Python 3.12 (x86_64+arm64) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp312-cp312-macosx_10_15_universal2.whl>`__
      - `Python 3.13 (x86_64+arm64) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp312-cp313-macosx_10_15_universal2.whl>`__

    * - Windows
      - `Python 3.10 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp310-cp310-win_amd64.whl>`__
      - `Python 3.11 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp311-cp311-win_amd64.whl>`__
      - `Python 3.12 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp312-cp312-win_amd64.whl>`__
      - `Python 3.13 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-@OPEN3D_VERSION_FULL@-cp313-cp313-win_amd64.whl>`__

Please use these links from the `latest version of this page
<https://www.open3d.org/docs/latest/getting_started.html>`__ only. You can also
install the latest development version directly with pip:

.. code-block:: bash

    pip install -U -f https://www.open3d.org/docs/latest/getting_started.html --only-binary open3d open3d

.. warning::
   The development wheels for Linux are named according to PEP600. Please
   use ``pip`` version >=20.3 to install them. The wheels are not yet fully
   PEP600 compliant.

Try it
------

.. code-block:: bash

    # Verify installation
    python -c "import open3d as o3d; print(o3d.__version__)"

    # Python API
    python -c "import open3d as o3d; \
               mesh = o3d.geometry.TriangleMesh.create_sphere(); \
               mesh.compute_vertex_normals(); \
               o3d.visualization.draw(mesh, raw_mode=True)"

    # Open3D CLI
    open3d example visualization/draw

If everything works, congratulations, now Open3D has been successfully installed!

Troubleshooting:
^^^^^^^^^^^^^^^^

If you get an error when importing Open3D, enable detailed Python warnings to
help troubleshoot the issue:

.. code-block:: bash

    python -W default -c "import open3d as o3d"

Running Open3D tutorials
------------------------

A complete set of Python tutorials and testing data will also be copied to
demonstrate the usage of Open3D Python interface. See ``examples/python`` for
all Python examples.

.. note:: Open3D's Python tutorial utilizes some external packages: ``numpy``,
    ``matplotlib``, ``opencv-python``.

.. _install_open3d_c++:

C++
===

To get started with using Open3D in your C++ applications, you can download a
binary package archive from `Github releases
<https://github.com/isl-org/Open3D/releases>`__ (since `v0.15`). These binary
package archives contain the Open3D shared library, include headers and GUI /
rendering resources. These are built with all supported features and are
available for the main supported platforms. Also, the latest development version
(``HEAD`` of ``main`` branch) binary package archives are provided here [#]_:

:Linux (Ubuntu 22.04+ or glibc 2.35+ [#]_):
    .. hlist::
        :columns: 2

        * `x86_64 (CXX11 ABI) <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-linux-x86_64-cxx11-abi-@OPEN3D_VERSION_FULL@.tar.xz>`__
        * `x86_64 (CXX11 ABI) with CUDA 12.6 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-linux-x86_64-cxx11-abi-cuda-@OPEN3D_VERSION_FULL@.tar.xz>`__

:MacOSX v10.15+:
    .. hlist::
        :columns: 2

        * `x86_64 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-darwin-x86_64-@OPEN3D_VERSION_FULL@.tar.xz>`__
        * `arm64 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-darwin-arm64-@OPEN3D_VERSION_FULL@.tar.xz>`__

:Windows 10+:
    .. hlist::
        :columns: 2

        * `x86_64 Release <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-windows-amd64-@OPEN3D_VERSION_FULL@.zip>`__
        * `x86_64 Debug <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d-devel-windows-amd64-@OPEN3D_VERSION_FULL@-dbg.zip>`__

.. [#] Please use these links from the `latest version of this page <https://www.open3d.org/docs/latest/getting_started.html>`__
    only.
.. [#] To check the `glibc` version on your system, run :code:`ldd --version`.

.. warning:: In Linux, do not link code with different CXX11 ABIs, since this will
    most likely cause linker errors or crashes. Most system libraries in recent
    Linux versions (e.g. if the OS came with GCC versions 5+) use the CXX11 ABI,
    while PyTorch and Tensorflow libraries typically use the pre CXX11 ABI.

If you need a subset of features, or a custom build configuration, please refer
to :ref:`compilation` and compile Open3D from source.

Try it
------

Extract the archive and move the contents to a local folder (such as
``$HOME/Documents/Open3D_install``):

.. code-block::

    Linux / MacOSX:                       Windows:
    Open3D_install                        Open3D_install
    ├── include                           ├── bin
    │   └── open3d                        │   ├── Open3D.dll
    │       ├── core                      │   └── resources
    │       ├── ...                       │       ├── brightday_ibl.ktx
    │       ├── Open3DConfig.h            │       ├── ...
    │       ├── Open3D.h                  │
    │       ├── ...                       ├── CMake
    ├── lib                               │   ├── Open3DConfig.cmake
    │   ├── cmake                         │   ├── ...
    │   │   └── Open3D                    ├── include
    │   │        ├── ...                  │   └── open3d
    │   ├── pkgconfig                     │       ├── core
    │   │   ├── Open3D.pc                 │       ├── ...
    │   │   ├── ...                       │       ├── Open3DConfig.h
    |   |                                 │       ├── Open3D.h
    │   ├── libOpen3D.so                  │       ├── ...
    │   ├── open3d_tf_ops.so              └── lib
    │   └── open3d_torch_ops.so               └── Open3D.lib
    └── share
        └── resources
            ├── html
            │    ├── ...
            ├── brightday_ibl.ktx
            ├── ...


Some files may be absent in the case of unsupported functionality. To use Open3D
with your programs through `cmake`, add ``-D
Open3D_ROOT=$HOME/Documents/Open3D_install`` to your CMake configure command
line. See the following example CMake projects for reference:

* `Find Pre-Installed Open3D Package in CMake <https://github.com/isl-org/open3d-cmake-find-package>`__
* `Use Open3D as a CMake External Project <https://github.com/isl-org/open3d-cmake-external-project>`__

The C++ code examples in the ``examples/cpp`` folder of the repository illustrate
a lot of the functionality available in Open3D and are a good place to start
using Open3D in your projects.
