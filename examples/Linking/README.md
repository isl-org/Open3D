# Open3D Linking Example

This examples shows how you and link your project to the installed Open3D.

## Usage

Specify `CMAKE_INSTALL_PREFIX` to the Open3D installation directory. If
`CMAKE_INSTALL_PREFIX` is not specified, CMake's default installation directory
will be used.

Example for Ubuntu/Mac:

```bash
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/open3d-installation
make
./ExampleLinking
```

Example for Windows:

```bat
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX="C:/Program Files/Open3D" -G "Visual Studio 15 2017 Win64" ..
make --build . --config Release
Release\ExampleLinking.exe
```
