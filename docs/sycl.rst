.. _sycl:

Cross-platform GPU support (SYCL)
=================================

From v0.19, Open3D provides an experimental SYCL backend for cross-platform GPU
support. This backend allows Open3D operations to run on many different GPUs,
including integrated GPUs and discrete GPUs from Intel, Nvidia and AMD. We
provide pre-built C++ binaries and Python wheels for Linux (Ubuntu 22.04+).

Enabled features
-----------------

Many Tensor API operations and Tensor Geometry operations without custom kernels
can now be offloaded to SYCL devices. In addition, HW accelerated raycasting
queries in :py:class:`open3d.t.geometry.RayCastingScene` are also supported. You
will get an error if an operation is not supported. The implementation is tested
on Linux on Intel integrated and discrete GPUs. Currently, a single GPU
(`SYCL:0`, if available) and the CPU (`SYCL:1` if a GPU is available, else
`SYCL:0`) are supported.

Installation
-------------

Both C++ binaries and Python wheels can be downloaded
from the Open3D GitHub releases page. For C++, install the `OneAPI runtime
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_
and (optionally) SYCL runtime for your `Nvidia
<https://developer.codeplay.com/products/oneapi/nvidia/download>`_ or `AMD
<https://developer.codeplay.com/products/oneapi/amd/download>`_ GPU.

For Python, the wheels will automatically install the DPC++ runtime package
(`dpcpp-cpp-rt`).  Make sure to have the `correct drivers installed 
<https://dgpu-docs.intel.com/driver/client/overview.html>`_ for your GPU. For
raycasting on Intel GPUs, you will also need the
`intel-level-zero-gpu-raytracing` package.


.. list-table::
    :stub-columns: 1
    :widths: auto

    * - Linux SYCL (Ubuntu 22.04+)
      - `Python 3.9 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_xpu-0.19.0-cp39-cp39-manylinux_2_31_x86_64.whl>`__
      - `Python 3.10 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_xpu-0.19.0-cp310-cp310-manylinux_2_31_x86_64.whl>`__
      - `Python 3.11 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_xpu-0.19.0-cp311-cp311-manylinux_2_31_x86_64.whl>`__
      - `Python 3.12 <https://github.com/isl-org/Open3D/releases/download/main-devel/open3d_xpu-0.19.0-cp312-cp312-manylinux_2_31_x86_64.whl>`__
      - `C++ x86_64 <https://github.com/isl-org/Open3D/releases/download/v0.19.0/open3d_xpu-devel-linux-x86_64-0.19.0.tar.xz>`__

Usage
------

The SYCL backend requires the new CXX11 ABI (Linux, gcc, libstdc++ only). If you
need to use the Open3D PyTorch extension, you should use cxx11_abi wheels for
PyTorch:

.. code-block:: shell

    pip install torch==2.2.2+cpu.cxx11.abi -i https://download.pytorch.org/whl/cpu/

PyTorch v2.7+ uses the new CXX11 ABI by default.

Some GPUs do not have native double precision support. For Intel GPUs, you can
emulate support with these environment variables:

.. code-block:: shell

    export IGC_EnableDPEmulation=1          # Enable float64 emulation during compilation 
    export OverrideDefaultFP64Settings=1    # Enable double precision emulation at runtime.

The binaries only contain kernels compiled to SPIR-V IR. At runtime, they will
be JIT compiled to your target GPU's native ISA. This means that the first run
of a kernel on a new device will be slower than subsequent runs.  Use this
environment variable to cache the JIT compiled kernels to your home directory:

.. code-block:: shell

    export SYCL_CACHE_PERSISTENT=1          # Cache SYCL kernel binaries.

.. code-block:: python

    import open3d as o3d
    o3d.core.sycl.enable_persistent_jit_cache() # Cache SYCL kernel binaries.

For multi-GPU systems (e.g. with both integrated and discrete GPUs), the more
powerful GPU is automatically selected, as long as the correct GPU drivers and
SYCL runtime are installed. You can select a specific device with the
`ONEAPI_DEVICE_FILTER` or `SYCL_DEVICE_ALLOWLIST`  `environment variables
<https://intel.github.io/llvm/EnvironmentVariables.html>`_.


.. code-block:: shell

    # Print all available devices (command line):
    sycl-ls
    # Examples:
    export ONEAPI_DEVICE_SELECTOR="opencl:1"    # Select the 2nd OpenCL device


.. code-block:: python

    # Print all available devices (Python):
    import os os.environ["SYCL_DEVICE_ALLOWLIST"] = "BackendName:cuda"  # Select CUDA GPU
    import open3d as o3d
    o3d.core.sycl.print_sycl_devices(print_all=True)

    # Return a list of available devices.
    o3d.core.sycl.get_available_devices() 

    # Check if a device is available
    o3d.core.sycl.is_available(o3d.core.Device("SYCL:0"))  


Building from source
---------------------

You can build the binaries from source as shown below. To build for a different
Python version, set the `PYTHON_VERSION` variable in `docker/docker_build.sh`.

.. code-block:: shell

    cd docker 
    ./docker_build.sh sycl-shared

This will create the Python wheel and C++ binary archive in the current
directory.

You can directly compile for a specific target device (i.e. ahead of time or AOT
compilation) using the OPEN3D_SYCL_TARGETS (`-fsycl-target` compiler option) and
OPEN3D_SYCL_TARGET_BACKEND_OPTIONS (`-Xs` compiler option) CMake variables in
Open3D. See the `compiler documentation
<https://github.com/intel/llvm/blob/sycl/sycl/doc/UsersManual.md>`_ for
information about building for specific hardware.

if you want to use different settings (e.g. AOT compilation for a specific
device, or build a wheel for a different Python version), you can update the
``docker_build.sh`` script, or build directly on host after installing the
``intel-basekit`` or ``intel-cpp-essentials`` Debian packages from the Intel
OneAPI repository.