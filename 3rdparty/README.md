# Open3D third-party libraries

This folder contains a set of external libraries that are used in Open3D. Open3D
prefers building third-party dependencies from source to detecting pre-installed
system dependencies.

## List of third-party libraries

```txt
--------------------------------------------------------------------------------
benchmark                   1.5.5                               Apache-2 license
A microbenchmark support library
https://github.com/google/benchmark
--------------------------------------------------------------------------------
boringssl:                  edfe413            Dual OpenSSL, SSLeay, ISC license
BoringSSL is a fork of OpenSSL that is designed to meet Google's needs.
https://github.com/google/boringssl
--------------------------------------------------------------------------------
CUB                         1.8.0                                    BSD license
A flexible library of cooperative threadblock primitives and other utilities for
CUDA kernel programming
https://github.com/NVlabs/cub
--------------------------------------------------------------------------------
cppzmq                      4.6.0                                    MIT license
Header-only C++ binding for libzmq
https://github.com/zeromq/cppzmq
As an alternative, you can modify 3rdparty/zeromq/zeromq_build.cmake to fetch
zeromq from our fork
https://github.com/isl-org/libzmq
--------------------------------------------------------------------------------
curl                        7.88.0                                  Curl license
Curl is a command-line tool for transferring data specified with URL syntax.
https://github.com/curl/curl
--------------------------------------------------------------------------------
CUTLASS                     1.3.3                                    BSD license
CUDA Templates for Linear Algebra Subroutines
https://github.com/NVIDIA/cutlass
--------------------------------------------------------------------------------
dirent                      1.21                                     MIT license
https://github.com/tronkko/dirent
A C/C++ programming interface for cross-platform filesystem
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
Eigen                       3.4                              Mainly MPL2 license
A high-level C++ library of template headers for linear algebra, matrix and
vector operations, numerical solvers and related algorithms
http://eigen.tuxfamily.org/
--------------------------------------------------------------------------------
embree                      4.3.1                               Apache-2 license
Embree is a collection of high-performance ray tracing kernels
https://github.com/embree/embree
--------------------------------------------------------------------------------
flann                       1.8.4                                    BSD license
A C++ library for performing fast approximate nearest neighbor searches in high
dimensional spaces
http://www.cs.ubc.ca/research/flann/
--------------------------------------------------------------------------------
GLEW                        2.1.0                                    MIT License
A cross-platform open-source C/C++ extension loading library
http://glew.sourceforge.net/
--------------------------------------------------------------------------------
GLFW                        3.3.0 (dev)                      zlib/libpng license
A cross-platform library for creating windows with OpenGL contexts and receiving
input and events
http://www.glfw.org/
--------------------------------------------------------------------------------
jsoncpp                     1.8.4                                    MIT license
A C++ library that allows manipulating JSON values
https://github.com/open-source-parsers/jsoncpp
--------------------------------------------------------------------------------
libjpeg-turbo                2.1.5.1                           BSD-style license
A widely used C library for reading and writing JPEG image files
https://github.com/libjpeg-turbo/libjpeg-turbo
--------------------------------------------------------------------------------
libpng                      1.6.37                                libpng license
The free reference library for reading and writing PNGs
http://www.libpng.org/
--------------------------------------------------------------------------------
librealsense                2.44.0                               Apache-2 license
A cross-platform library for capturing data from the Intel RealSense F200,
SR300, R200 and L500 cameras
https://github.com/IntelRealSense/librealsense
--------------------------------------------------------------------------------
libzmq                      4.3.3         LGPLv3 + static link exception license
ZeroMQ is a high-performance asynchronous messaging library
https://github.com/zeromq/libzmq
--------------------------------------------------------------------------------
msgpack-c                   3.3.0                     Boost Software License 1.0
MessagePack implementation for C and C++
https://github.com/msgpack/msgpack-c/tree/cpp_master
--------------------------------------------------------------------------------
nanoflann                   1.3.1                                    BSD license
A C++11 header-only library for Nearest Neighbor (NN) search with KD-trees
https://github.com/jlblancoc/nanoflann
--------------------------------------------------------------------------------
PoissonReco                 12.0                                     BSD license
Poisson Surface Reconstruction
https://github.com/mkazhdan/PoissonRecon
--------------------------------------------------------------------------------
pybind11                    v2.13.1                                  BSD license
Python binding for C++11
https://github.com/pybind/pybind11
--------------------------------------------------------------------------------
RPly                        1.1.3                                    MIT license
A library to read and write PLY files
http://w3.impa.br/~diego/software/rply/
--------------------------------------------------------------------------------
stdgpu                       1b6a3319                         Apache-2.0 license
Efficient STL-like Data Structures on the GPU
https://github.com/stotko/stdgpu/
--------------------------------------------------------------------------------
tinyfiledialogs             2.7.2                                   zlib license
A lightweight cross-platform file dialog library
https://sourceforge.net/projects/tinyfiledialogs/
--------------------------------------------------------------------------------
tinygltf                    72f4a55                                  MIT license
Header only C++11 tiny glTF 2.0 library
https://github.com/syoyo/tinygltf
--------------------------------------------------------------------------------
tinyobjloader                v1.0.0                                  MIT license
Tiny but powerful single file wavefront obj loader
https://github.com/syoyo/tinyobjloader
--------------------------------------------------------------------------------
UVAtlas                    may2022                                   MIT license
UVAtlas isochart texture atlas
https://github.com/microsoft/uvatlas
--------------------------------------------------------------------------------
vtk                            9.1                                   BSD license
The Visualization Toolkit (VTK)
https://gitlab.kitware.com/vtk/vtk
--------------------------------------------------------------------------------
zlib                        1.2.13                                   zlib license
A lossless data-compression library used by libpng
http://www.zlib.net/
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
