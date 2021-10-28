# OneAPI Readme

## Python packages

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

## Plan (2021-10-27)
- MKL: use static version
- IPP: use static version
- TBB: we want TBB to work on ARM
