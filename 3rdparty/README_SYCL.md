# Open3D SYCL

Note: this file will be moved to the `docs` folder once SYCL support is mature.

## Concepts

- **OpenCL**: Low-level programming interface for heterogeneous platforms.
- **SYCL**: High-level programming interface for heterogeneous platforms. SYCL
  was originally based on OpenCL but now SYCL is an independent standard.
- **DPC++**: Intel's implementation of the SYCL standard.
- **oneAPI**: A collection of open standards, including DPC++, oneMKL, oneTBB, etc.

Open3D's SYCL support runs on DPC++ and requires oneAPI dependencies. For
convenience, in the source code, we use the term "SYCL" although we may be
referring to "DPC++".

## Setup

1. Install oneAPI for Linux: [install via apt-get](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html)
2. Prepare environment
   ```bash
   # Source environments
   source /opt/intel/oneapi/setvars.sh

   # We'll be using oneAPI's distribution of conda and Python
   # Python 3.6+ will work
   conda create -n sycl python=3.8
   conda activate sycl
   ```
3. Check your environment
   ```bash
   which icx        # /opt/intel/oneapi/compiler/<version>/linux/bin/icx
   which icpx       # /opt/intel/oneapi/compiler/<version>/linux/bin/icpx
   which python     # ${HOME}/.conda/envs/sycl/bin/python
   python --version # Python 3.8.12 :: Intel Corporation
   ```
4. Config and build
   ```bash
   cmake -DBUILD_SYCL_MODULE=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx ..
   make -j$(nproc)
   ```
5. Run demos
   We provide several ways to run SYCL demos. This ensures the linking are
   correct and all run-time dependencies are satisfied.
   ```bash
   # C++ test
   make tests -j$(nproc)
   ./bin/tests --gtest_filter="*SYCLDemo*"

   # C++ example
   make SYCLDemo -j$(nproc)
   ./bin/examples/SYCLDemo

   # Python
   make install-pip-package -j$(nproc)
   pytest ../python/test/core/test_sycl_utils.py -s
   ```

## Available SYCL devices

After activating the SYCL environment, you may run `sycl-ls` to list available
SYCL devices, for example:

```bash
$ sycl-ls
[opencl:acc:0] Intel(R) FPGA Emulation Platform for OpenCL(TM)
[opencl:cpu:1] Intel(R) OpenCL, 12th Gen Intel(R) Core(TM) i9-12900K
[opencl:gpu:2] Intel(R) OpenCL HD Graphics, Intel(R) UHD Graphics 770
[ext_oneapi_level_zero:gpu:0] Intel(R) Level-Zero, Intel(R) UHD Graphics 770
[host:host:0] SYCL host platform, SYCL host device
```

Open3D can make use of the host and GPU SYCL devices.
- `core::Device("SYCL_CPU:0")` maps to the device returned by
  `sycl::host_selector()`.
- `core::Device("SYCL_GPU:0")` maps the default SYCL GPU returned by
  `sycl::gpu_selector()`.
  - Currently we only support one SYCL GPU. Multi-GPU support will be added in
    the future.
  - For Intel GPUs, the "level-zero GPU" (`ext_oneapi_level_zero:gpu:0` in the
    example above) backend will be used by default instead of the "OpenCL GPU"
    (`opencl:gpu:2` in the example above).

## Known limitations/requirement

- Limitations (not implemented yet)
  - Only supports Linux
  - Only supports `BUILD_CUDA_MODULE=OFF`
  - ML ops not supported (C++17 cannot be enabled for TF)
- Requirements (required by DPC++)
  - Only supports `GLIBCXX_USE_CXX11_ABI=ON`
  - Only supports `set(CMAKE_CXX_STANDARD 17)`


## List of oneAPI Python packages

To make `pip install open3d` works out-of-the box on SYCL-enabled platforms,
we can utilize runtime libraries released via PyPI. This feature needs to be
implemented.

User:
- https://pypi.org/user/IntelAutomationEngineering/

Libraries:
- dpcpp-cpp-rt     https://pypi.org/project/dpcpp-cpp-rt/#history
- mkl              https://pypi.org/project/mkl/#history
- mkl-static       https://pypi.org/project/mkl-static/#history
- mkl-include      https://pypi.org/project/mkl-include/#history
- mkl-dpcpp        https://pypi.org/project/mkl-dpcpp/#history
- mkl-devel-dpcpp  https://pypi.org/project/mkl-devel-dpcpp/#history
- ipp              https://pypi.org/project/ipp/#history
- ipp-static       https://pypi.org/project/ipp-static/#history
- tbb              https://pypi.org/project/tbb/#history
