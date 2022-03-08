# OneAPI Readme

## Known limitations
- (fixed) TBB issue: `cpp/open3d/utility/ParallelScan.h` is disabled.
- Does not work with `-DBUILD_SHARED_LIBS=OFF`: need to refactor installation mechanism.
- Does not work with `-DBUILD_CUDA_MODULE=ON`: Faiss does not support icpx compiler.
  - Simlify to two binaries.
- Python: dynamic library loading.
- ABI=1 currently required.

## Python packages (not used for now)

https://pypi.org/user/IntelAutomationEngineering/
- dpcpp-cpp-rt     https://pypi.org/project/dpcpp-cpp-rt/#history     2021.4.0
- mkl              https://pypi.org/project/mkl/#history              2021.4.0
- mkl-static       https://pypi.org/project/mkl-static/#history       2021.4.0
- mkl-include      https://pypi.org/project/mkl-include/#history      2021.4.0
- mkl-dpcpp        https://pypi.org/project/mkl-dpcpp/#history        2021.4.0
- mkl-devel-dpcpp  https://pypi.org/project/mkl-devel-dpcpp/#history  2021.4.0
- ipp              https://pypi.org/project/ipp/#history              2021.4.0
- ipp-static       https://pypi.org/project/ipp-static/#history       2021.4.0
- tbb              https://pypi.org/project/tbb/#history              2021.4.0

## Use cases
- Run example and tests inside the `build/` directory.
- Imported by another C++ library, after `make install`.
- Distribute with Python (pip).

## Decisions
- MKL: use shared version
- IPP: use static version
- TBB: for ARM
  - C++: compile TBB from source, static or dynamic?
  - Python release: TODO
