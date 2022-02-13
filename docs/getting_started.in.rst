.. _getting_started:

Getting started
###############

.. _install_open3d_python:

Python
======

Open3D Python packages are distributed via
`PyPI <https://pypi.org/project/open3d/>`_.

Supported Python versions:

.. hlist::
    :columns: 4

    * 3.6
    * 3.7
    * 3.8
    * 3.9

Supported operating systems:

* Ubuntu 18.04+
* macOS 10.15+
* Windows 10 (64-bit)

If you have other Python versions or operating systems, please refer to
:ref:`compilation` and compile Open3D from source.

Pip (PyPI)
----------

.. code-block:: bash

    pip install open3d

.. note::
   Please upgrade your ``pip`` to a version >=20.3 to install Open3D in Linux,
   e.g. with

        ``pip install -U pip>=20.3``

.. note::
    In general, we recommend using a
    `virtual environment <https://docs.python-guide.org/dev/virtualenvs/>`_
    or `conda environment <https://docs.conda.io/en/latest/miniconda.html>`_.
    Otherwise, depending on the configurations, ``pip3`` may be needed for
    Python 3, or the ``--user`` option may need to be used to avoid permission
    issues. For example:

    .. code-block:: bash

        pip3 install open3d
        # or
        pip install --user open3d
        # or
        python3 -m pip install --user open3d

Development version (pip)
-------------------------

To test the latest features in Open3D, download and install the development
version (``HEAD`` of ``master`` branch):

.. list-table::
    :stub-columns: 1
    :widths: auto

    * - Linux
      - `Python 3.6 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp36-cp36m-manylinux_2_27_x86_64.whl>`__
      - `Python 3.7 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp37-cp37m-manylinux_2_27_x86_64.whl>`__
      - `Python 3.8 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp38-cp38-manylinux_2_27_x86_64.whl>`__
      - `Python 3.9 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp39-cp39-manylinux_2_27_x86_64.whl>`__

    * - MacOS
      - `Python 3.6 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp36-cp36m-macosx_10_15_x86_64.whl>`__
      - `Python 3.7 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp37-cp37m-macosx_10_15_x86_64.whl>`__
      - `Python 3.8 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp38-cp38-macosx_10_15_x86_64.whl>`__
      - `Python 3.9 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp39-cp39-macosx_10_15_x86_64.whl>`__

    * - Windows
      - `Python 3.6 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp36-cp36m-win_amd64.whl>`__
      - `Python 3.7 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp37-cp37m-win_amd64.whl>`__
      - `Python 3.8 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp38-cp38-win_amd64.whl>`__
      - `Python 3.9 <https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp39-cp39-win_amd64.whl>`__

Please use these links from the `latest version of this page
<http://www.open3d.org/docs/latest/getting_started.html>`__ only. For example,
to install the latest development version on Linux for Python 3.9:

.. code-block:: bash

    pip install --user --pre \
        https://storage.googleapis.com/open3d-releases-master/python-wheels/open3d-@OPEN3D_VERSION_FULL@-cp39-cp39-linux_x86_64.whl

.. note::
   The development wheels for Linux are named according to PEP600. Please
   use ``pip`` version >=20.3 to install them. The wheels are not yet fully
   PEP600 compliant.

Try it
------

Now, try importing Open3D.

.. code-block:: bash

    python -c "import open3d as o3d"

If this works, congratulations, now Open3D has been successfully installed!


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
    ``matplotlib``, ``opencv-python``. OpenCV is only used for reconstruction
    system. Please read ``util/install-deps-python.sh`` for installing these
    packages.


.. _install_open3d_c++:

C++
===

To get started with using Open3D in your C++ applications, you can download a
binary package archive from `Github releases
<https://github.com/isl-org/Open3D/releases>`__ (since `v0.15`). These binary
package archives contain the Open3D shared library built with all supported
features and are available for the main supported platforms. Also, the latest
development version (``HEAD`` of ``master`` branch) binary package archives are
provided here [#]_:

:Linux (Ubuntu 18.04+ or glibc 2.27+ [#]_):
    .. hlist::
        :columns: 2

        * `x86_64 (CXX11 ABI) <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-linux-x86_64-cxx11-abi-@OPEN3D_VERSION_FULL@.tar.xz>`__
        * `x86_64 (CXX11 ABI) with CUDA 11.x <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-linux-x86_64-cxx11-abi-cuda-@OPEN3D_VERSION_FULL@.tar.xz>`__
        * `x86_64 (pre CXX11 ABI) <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-linux-x86_64-pre-cxx11-abi-@OPEN3D_VERSION_FULL@.tar.xz>`__
        * `x86_64 (pre CXX11 ABI) with CUDA 11.x <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-linux-x86_64-pre-cxx11-abi-cuda-@OPEN3D_VERSION_FULL@.tar.xz>`__

:MacOSX v10.15+:
    .. hlist::
        :columns: 2

        * `x86_64 <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-darwin-x86_64-@OPEN3D_VERSION_FULL@.tar.xz>`__

:Windows 10+:
    .. hlist::
        :columns: 2

        * `x86_64 Release <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-windows-amd64-@OPEN3D_VERSION_FULL@.zip>`__
        * `x86_64 Debug <https://storage.googleapis.com/open3d-releases-master/devel/open3d-devel-windows-amd64-@OPEN3D_VERSION_FULL@-dbg.zip>`__

.. [#] Please use these links from the `latest version of this page <http://www.open3d.org/docs/latest/getting_started.html>`__
    only.
.. [#] To check the `glibc` version on your system, run :code:`ldd --version`.

.. note:: In Linux, do not link code with different CXX11 ABIs, since this will
    most likely cause linker errors or crashes. Most system libraries in recent
    Linux versions (e.g. if the OS came with GCC versions 5+) use the CXX11 ABI,
    while PyTorch and Tensorflow libraries typically use the pre CXX11 ABI.

If you need only a subset of features, or a custom build configuration, please
refer to :ref:`compilation` and compile Open3D from source.

Try it
------

Extract the archive and move the contents to a local folder (such as
``$HOME/Documents/Open3D_install``):

.. code-block::

    Linux / MacOSX:                       Windows:
    Open3D_install                        Open3D_install
    ├── include                           ├── bin
    │   └── open3d                        │   └── Open3D.dll
    │       ├── core                      ├── CMake
    │       ├── ...                       │   ├── Open3DConfig.cmake
    │       ├── Open3DConfig.h            │   ├── ...
    │       ├── Open3D.h                  ├── include
    │       ├── ...                       │   └── open3d
    └── lib                               │       ├── core
        ├── cmake                         │       ├── ...
        │   └── Open3D                    │       ├── Open3DConfig.h
        │        ├── ...                  │       ├── Open3D.h
        ├── libOpen3D.so                  │       ├── ...
        ├── open3d_tf_ops.so              └── lib
        └── open3d_torch_ops.so               └── Open3D.lib


Some files may be absent in the case of unsupported functionality. To use Open3D
with your programs through `cmake`, add ``-D
Open3D_ROOT=$HOME/Documents/Open3D_install`` to your CMake configure command
line. See the following example CMake projects for reference:

* `Find Pre-Installed Open3D Package in CMake <https://github.com/isl-org/open3d-cmake-find-package>`__
* `Use Open3D as a CMake External Project <https://github.com/isl-org/open3d-cmake-external-project>`__

The C++ code examples in the ``examples/cpp`` folder of the repository illustrate
a lot of the functionality available in Open3D and are a good place to start
using Open3D in your projects.
