# Open3D SYCL

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
4. Config
   ```bash
   # Add the following flag to CMake
   cmake -DBUILD_SYCL_MODULE=ON ..
   ```

## List of oneAPI Python packages

https://pypi.org/user/IntelAutomationEngineering/
- dpcpp-cpp-rt     https://pypi.org/project/dpcpp-cpp-rt/#history
- mkl              https://pypi.org/project/mkl/#history
- mkl-static       https://pypi.org/project/mkl-static/#history
- mkl-include      https://pypi.org/project/mkl-include/#history
- mkl-dpcpp        https://pypi.org/project/mkl-dpcpp/#history
- mkl-devel-dpcpp  https://pypi.org/project/mkl-devel-dpcpp/#history
- ipp              https://pypi.org/project/ipp/#history
- ipp-static       https://pypi.org/project/ipp-static/#history
- tbb              https://pypi.org/project/tbb/#history

## Known limitations/requirement

- Limitations (not implemented yet)
  - Only supports Linux
  - Only supports `BUILD_CUDA_MODULE=OFF`
  - ML ops not supported (C++17 cannot be enabled for TF)
- Requirements (required by DPC++)
  - Only supports `GLIBCXX_USE_CXX11_ABI=ON`
  - Only supports `set(CMAKE_CXX_STANDARD 17)`
