# Building Conda package for Open3D

## Prerequisites

- Install `conda`
- Install `conda-build` by `conda install conda-build`

## Build Conda package

We use the same `setup.py` and shared library objects as when we're building 
the `pip_package`.

### 1. Copy shared libraries

Copy the shared libraries to the corresponding directory inside `Open3D/util/pip_package/open3d`. For example, on macOS with Python 3.6, `open3d.cpython-36m-darwin.so` shall be
copied to `Open3D/util/pip_package/open3d/macOS`

### 2. Build Conda package

```bash
cd Open3D/util/conda_package
conda-build open3d --output-folder dist
```

After building, `conda-build` automatically tries to run `import open3d`. If the 
build is successful, the Conda package tar shall be located in 
`Open3D/util/conda_package/dist`. 

For example, on macOS with Python 3.6, the package is located at
`/Open3D/util/conda_package/dist/osx-64/open3d-0.2.0-py36_0.tar.bz2`

### 3. Install Conda package (optional)

Example for macOS with Python 3.6.

```bash
# Create a fresh Conda virtualenv, the Python version shall be consistent
conda create -n newenv3.6 python=3.6

# Activate
conda activate newenv3.6

# Install the Conda packages
conda install numpy matplotlib # TODO: conda is not picking up the deps yet
conda install Open3D/util/conda_package/dist/osx-64/open3d-0.2.0-py36_0.tar.bz2

# Try importing
python -c "from open3d import *"
```
