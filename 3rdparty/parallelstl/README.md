# Parallel STL
[![Stable release](https://img.shields.io/badge/version-20190522-green.svg)](https://github.com/intel/parallelstl/releases/tag/20190522)

Parallel STL is an implementation of the C++ standard library algorithms with support for execution policies,
as specified in ISO/IEC 14882:2017 standard, commonly called C++17.
The implementation also supports the unsequenced execution policy specified in Parallelism TS version 2
and proposed for the next version of the C++ standard in the C++ working group paper P1001R1.

Parallel STL offers a portable implementation of threaded and vectorized execution of standard C++ algorithms, optimized and validated for Intel(R) 64 processors.
For sequential execution, it relies on an available implementation of the C++ standard library.

The source code in this repository corresponds to the releases of Parallel STL with Intel(R) C++ Compiler or with Threading Building Blocks.
The upstream source code repository for development has moved to LLVM, with the GitHub mirror at https://github.com/llvm-mirror/pstl.

## Prerequisites
To use Parallel STL, you must have the following software installed:
* C++ compiler with:
  * Support for C++11
  * Support for OpenMP* 4.0 SIMD constructs
* Threading Building Blocks (TBB) which is available to download in the GitHub [repository](https://github.com/01org/tbb/)

## Release Information
Here are the latest [Changes](CHANGES) and [Release Notes](Release_Notes.txt) (contains system requirements and known issues).

## License
Parallel STL is licensed under [Apache License Version 2.0 with LLVM exceptions](LICENSE).

## Documentation
See [Getting Started](https://software.intel.com/en-us/get-started-with-pstl) with Parallel STL.

## Support and contribution
Please report issues and suggestions via [LLVM Bugzilla](https://bugs.llvm.org/),
[GitHub issues](https://github.com/intel/parallelstl/issues), or start a topic on the
[TBB forum](http://software.intel.com/en-us/forums/intel-threading-building-blocks/).

If you want to contribute to the development, please do it via the upstream repository at LLVM.
Read [the LLVM Developer Policy](https://llvm.org/docs/DeveloperPolicy.html) for additional details.
Pull requests to this repository are no more accepted.

## Contacts
* [libc++ developers mailing list](https://lists.llvm.org/mailman/listinfo/libcxx-dev)
* [E-mail the TBB team](mailto:inteltbbdevelopers@intel.com)

------------------------------------------------------------------------
Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

\* Other names and brands may be claimed as the property of others.
