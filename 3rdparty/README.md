# Open3D third-party libraries

This folder contains a set of external libraries that are used in Open3D. Open3D
prefers building third-party dependencies from source to detecting pre-installed
system dependencies.

## List of third-party libraries

```txt
--------------------------------------------------------------------------------
Eigen                       3.4                              Mainly MPL2 license
A high-level C++ library of template headers for linear algebra, matrix and
vector operations, numerical solvers and related algorithms
http://eigen.tuxfamily.org/
--------------------------------------------------------------------------------
zlib                        1.2.8                                   zlib license
A lossless data-compression library used by libpng
http://www.zlib.net/
--------------------------------------------------------------------------------
jsoncpp                     1.8.4                                    MIT license
A C++ library that allows manipulating JSON values
https://github.com/open-source-parsers/jsoncpp
--------------------------------------------------------------------------------
flann                       1.8.4                                    BSD license
A C++ library for performing fast approximate nearest neighbor searches in high
dimensional spaces
http://www.cs.ubc.ca/research/flann/
--------------------------------------------------------------------------------
dirent                      1.21                                     MIT license
https://github.com/tronkko/dirent
A C/C++ programming interface for cross-platform filesystem
--------------------------------------------------------------------------------
pybind11                    v2.6.2                                   BSD license
Python binding for C++11
https://github.com/pybind/pybind11
--------------------------------------------------------------------------------
Parallel STL                20190522                            Apache-2 license
An implementation of the C++ standard library algorithms with support for
execution policies
https://github.com/oneapi-src/oneDPL
--------------------------------------------------------------------------------
CUB                         1.8.0                                    BSD license
A flexible library of cooperative threadblock primitives and other utilities for
CUDA kernel programming
https://github.com/NVlabs/cub
--------------------------------------------------------------------------------
nanoflann                   1.3.1                                    BSD license
A C++11 header-only library for Nearest Neighbor (NN) search with KD-trees
https://github.com/jlblancoc/nanoflann
--------------------------------------------------------------------------------
CUTLASS                     1.3.3                                    BSD license
CUDA Templates for Linear Algebra Subroutines
https://github.com/NVIDIA/cutlass
--------------------------------------------------------------------------------
benchmark                   1.5.0                               Apache-2 license
A microbenchmark support library
https://github.com/google/benchmark
--------------------------------------------------------------------------------
curl                        7.79.1                                  Curl license
Curl is a command-line tool for transferring data specified with URL syntax.
https://github.com/curl/curl
--------------------------------------------------------------------------------
boringssl:                  edfe413            Dual OpenSSL, SSLeay, ISC license
BoringSSL is a fork of OpenSSL that is designed to meet Google's needs.
https://github.com/google/boringssl
--------------------------------------------------------------------------------
DirectX-Headers           v1.606.3                                   MIT license
Official DirectX headers available under an open source license
https://github.com/microsoft/DirectX-Headers
--------------------------------------------------------------------------------
DirectXMath                may2022                                   MIT license
DirectXMath is an all inline SIMD C++ linear algebra library for use in games
and graphics apps
https://github.com/microsoft/DirectXMath
--------------------------------------------------------------------------------
```

## Patching a third-party library

Using `assimp` as an example.

```bash
# Do this outside of a git directory
cd /tmp

# Download the tar.gz
wget https://github.com/assimp/assimp/archive/refs/tags/v5.1.3.tar.gz
tar xzf v5.1.3.tar.gz
cd assimp-5.1.3

# Init git and add all source
git init
git add .
git commit -am "Init commit"

# Make changes to the files, commit
cp /my/new/ObjFileData.h          code/AssetLib/Obj/ObjFileData.h
cp /my/new/ObjFileImporter.cpp    code/AssetLib/Obj/ObjFileImporter.cpp
cp /my/new/ObjFileMtlImporter.cpp code/AssetLib/Obj/ObjFileMtlImporter.cpp
git add .
git commit -am "Patch Assimp Obj importer"

# Create patch file to HEAD~1
git format-patch HEAD~1

# Test the patch
git reset --hard HEAD~1
git apply --ignore-space-change --ignore-whitespace 0001-Patch-Assimp-Obj-importer.patch
git status
```

Finally, this patch can be used in CMake `ExternalProject_Add` by specifying:

```cmake
find_package(Git QUIET REQUIRED)

ExternalProject_Add(
    ...
    PATCH_COMMAND ${GIT_EXECUTABLE} init
    COMMAND       ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
                  /path/to/0001-Patch-Assimp-Obj-importer.patch
    ...
)
```
