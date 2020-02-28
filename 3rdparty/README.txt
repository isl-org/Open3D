This folder contains a set of external libraries that are used in Open3D.

We want to make Open3D self-contained and cross-platformly consistent. So
Open3DV does not detect system installed libraries and always link to these
external libraries.

Some external libraries rely on basic system level libraries such as OpenGL and
libusb. Run corresponding script file under "script" directory to automatically
config them.

--------------------------------------------------------------------------------
Eigen                       3.3.2                            Mainly MPL2 license
A high-level C++ library of template headers for linear algebra, matrix and
vector operations, numerical solvers and related algorithms
http://eigen.tuxfamily.org/
--------------------------------------------------------------------------------
GLFW                        3.3.0 (dev)                      zlib/libpng license
A cross-platform library for creating windows with OpenGL contexts and receiving
input and events
http://www.glfw.org/
--------------------------------------------------------------------------------
GLEW                        2.1.0                                    MIT License
A cross-platform open-source C/C++ extension loading library
http://glew.sourceforge.net/
--------------------------------------------------------------------------------
RPly                        1.1.3                                    MIT license
A library to read and write PLY files
http://w3.impa.br/~diego/software/rply/
--------------------------------------------------------------------------------
zlib                        1.2.8                                   zlib license
A lossless data-compression library used by libpng
http://www.zlib.net/
--------------------------------------------------------------------------------
libpng                      1.6.18                                libpng license
The free reference library for reading and writing PNGs
http://www.libpng.org/
--------------------------------------------------------------------------------
libjpeg                     9a                                   libjpeg license
A widely used C library for reading and writing JPEG image files
http://libjpeg.sourceforge.net/
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
librealsense                0.9.1                               Apache-2 license
A cross-platform library for capturing data from the Intel RealSense F200, SR300
and R200 cameras
https://github.com/IntelRealSense/librealsense
--------------------------------------------------------------------------------
tinyfiledialogs             2.7.2                                   zlib license
A lightweight cross-platform file dialog library
https://sourceforge.net/projects/tinyfiledialogs/
--------------------------------------------------------------------------------
tinygltf                    v2.2.0                                   MIT license
Header only C++11 tiny glTF 2.0 library
https://github.com/syoyo/tinygltf
--------------------------------------------------------------------------------
tinyobjloader                v1.0.0                                  MIT license
Tiny but powerful single file wavefront obj loader
https://github.com/syoyo/tinyobjloader
--------------------------------------------------------------------------------
pybind11                    2.2                                      BSD license
Python binding for C++11
https://github.com/pybind/pybind11
--------------------------------------------------------------------------------
PoissonReco                 12.0                                     BSD license
Poisson Surface Reconstruction
https://github.com/mkazhdan/PoissonRecon
--------------------------------------------------------------------------------
CUTLASS                     1.3.2                                    BSD license
CUDA Templates for Linear Algebra Subroutines
https://github.com/NVIDIA/cutlass
