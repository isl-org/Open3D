# Open3D

[![Build Status](https://travis-ci.com/IntelVCL/Open3D.svg?token=J6RafDafqG2bAk9tQXMU&branch=master)](https://travis-ci.com/IntelVCL/Open3D)

A modern cross-platform C++ library for 3d modeling and geometry processing.

## Compilation

Open3D is compiled using [CMake](https://cmake.org/). It uses C++11 features which are supported by any un-ancient compilers such as gcc 4.8+, Visual Studio 2015+, Clang 8.0+.

### Ubuntu

The compilation has been tested on Ubuntu 16.04 (gcc 5.4) and Ubuntu 15.10 (gcc 4.9):

```
> scripts/install-deps-ubuntu.sh
> mkdir build
> cd build
> cmake ../src/
> make
```

### OS X

The compilation has been tested with El Capitan (Clang or Xcode) and Sierra (Clang or Xcode). Follow the instructions in the Ubuntu section to compile from console with Clang. If you want to use Xcode:
```
> scripts/install-deps-osx.sh
> mkdir build-xcode
> cd build-xcode
> cmake -G Xcode ../src/
> open Open3D.xcodeproj/
```

### Windows

The compilation has been tested with Windows 8 and 10 (Visual Studio 2015). All dependencies for Windows have been included in the codebase for easy compilation. You can use the CMake GUI as follows. Click **Configure** and choose the correct Visual Studio version, then click **Generate**. Open the solution file with Visual Studio, change the build type to **Release**, then **rebuild** the **ALL_BUILD** target.
