# SYCL Readme

Open3D's SYCL support runs on Intel oneAPI with DPC++. DPC++ is Intel's
implementation of the SYCL standard.

## TODO

- PR #0
  - [ ] Docker file
  - [ ] Decide on namespace name
  - [ ] Build CUDA together
- PR #1
  - [ ] Allow `BUILD_SHARED_LIBS=OFF`

## Setup

Under construction.

## Known limitations/requirement

Limitation == not implemented yet; Requirement == required by DPC++.

- Limitation: only supports `BUILD_SHARED_LIBS=ON`
- Limitation: only supports `BUILD_CUDA_MODULE=OFF`
- Requirement: only supports `GLIBCXX_USE_CXX11_ABI=ON`
- Requirement: only supports `set(CMAKE_CXX_STANDARD 17)`

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
