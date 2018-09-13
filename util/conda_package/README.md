# Building Conda package for Open3D

## Prerequisites

Install `conda`, following
[this instruction](https://conda.io/docs/user-guide/install/index.html).

## Build Conda package

We use the same `setup.py` and shared library objects as when we're building
the `pip_package`.

### 1. Compile Open3D with different platforms with different Python versions

Example of building with Python 2.7 on Mac -- other platforms and Python
versions shall be similar:

```bash
# Create a clean build folder, you'll need different build folders for
# different configurations
mkdir build_macos_27
cd build_macos_27

# Create build environment with a specific Python version
conda create -n clean_build_env2.7 python=2.7
source activate clean_build_env2.7

# On Mac/Ubuntu, we need to build all dependencies as static libs from source
# On Windows, this is enabled by default
cmake -DWITH_OPENMP=ON -DBUILD_EXPERIMENTAL=ON -DBUILD_EIGEN3=ON \
      -DBUILD_GLEW=ON -DBUILD_GLFW=ON -DBUILD_JPEG=ON -DBUILD_JSONCPP=ON \
      -DBUILD_PNG=ON -DBUILD_PYBIND11=ON -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_TINYFILEDIALOGS=ON \
      -DPYTHON_EXECUTABLE=`which python2.7` ..
make -j

# Copy shared library to the corresponding platform folder
# The `open3d.so` name will be different depending on your Python version
cp lib/Python/open3d.so ../util/pip_package/open3d/macos

# Deactivate
source deactivate
cd ..
```

Note: On windows, in order to build 32-bit binaries, we'll need to run

```
set CONDA_FORCE_32BIT=1
```
before creating and activating the conda environment.

When configuring the projects, instead of `Visual Studio 15 2017 Win64`, we
use

```
cmake -DPYTHON_EXECUTABLE=C:\path_to_the_32bit_python_env\python.exe \
      -G "Visual Studio 15 2017" ..
```

After copying the file, the directory shall contain the following files:

```bash
➜  ~/repo/Open3D tree util/pip_package/open3d/
util/pip_package/open3d/
├── __init__.py
├── linux
│   ├── __init__.py
│   ├── open3d.cpython-35m-x86_64-linux-gnu.so
│   ├── open3d.cpython-36m-x86_64-linux-gnu.so
│   ├── open3d.so
│   └── readme.txt
├── macos
│   ├── __init__.py
│   ├── open3d.cpython-35m-darwin.so
│   ├── open3d.cpython-36m-darwin.so
│   ├── open3d.so
│   └── readme.txt
└── win32
    ├── 32b
    │   ├── __init__.py
    │   ├── open3d.cp35-win32.pyd
    │   ├── open3d.cp36-win32.pyd
    │   └── open3d.pyd
    ├── 64b
    │   ├── __init__.py
    │   ├── open3d.cp35-win_amd64.pyd
    │   ├── open3d.cp36-win_amd64.pyd
    │   └── open3d.pyd
    ├── __init__.py
    └── readme.txt
```

### 2. Build Conda package

For each OS, we need to run the following commands to build the conda package:

```bash
cd Open3D/util/conda_package

# Create a fresh Conda virtualenv
conda create -n build_env python=3.6
source activate build_env

# Install conda-build
conda install conda-build

# Build Conda package
conda-build open3d --output-folder dist
```

After building, `conda-build` automatically tries to run `import open3d`. If the
build is successful, the Conda package tar files shall be located in
`Open3D/util/conda_package/dist`.

### 3. Install Conda package locally for testing (optional)

We could test the conda package locally. Example for macOS with Python 3.6:

```bash
# Activate the new environment that we just created
conda create -n test_env3.6 python=3.6
source activate test_env3.6

# Install the Conda packages
conda install numpy # For local conda packages, we need to install deps manually
conda install Open3D/util/conda_package/dist/osx-64/open3d-0.3.0-py36_0.tar.bz2

# Try importing
python -c "from open3d import *"
```

### 4. Upload to Anaconda cloud

The last step is to upload it to Anaconda cloud. Example for macOS with Python
3.6:

```bash
# Install Anaconda client
source activate build_env
conda install anaconda-client

# Example: upload macOS python3.6 package
anaconda upload dist/osx-64/open3d-0.3.0-py36_0.tar.bz2
```

Now we can try installing it by:

```bash
conda create -n new_env3.6 python=3.6
source activate new_env3.6

conda install -c open3d-admin open3d
```
