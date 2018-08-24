# Building Conda package for Open3D

## Prerequisites

Install `conda`, following
[this instruction](https://conda.io/docs/user-guide/install/index.html).

## Build Conda package

We use the same `setup.py` and shared library objects as when we're building
the `pip_package`.

### 1. Copy shared libraries

Copy the shared libraries to the corresponding directory inside
`Open3D/util/pip_package/open3d`. For example, on macOS with Python 3.6,
`open3d.cpython-36m-darwin.so` shall be copied to
`Open3D/util/pip_package/open3d/macOS`

### 2. Build Conda package

```bash
cd Open3D/util/conda_package

# Create a fresh Conda virtualenv, the Python version shall be consistent
# to the Python version you used to build Open3D
conda create -n newenv3.6 python=3.6

# Activate
source activate newenv3.6

# Install conda-build. Notice that conda-build shall be installed inside of
# the same Conda environment to make sure that the Python versions match
conda install conda-build

# Build Conda package
conda-build open3d --output-folder dist
```

After building, `conda-build` automatically tries to run `import open3d`. If the
build is successful, the Conda package tar shall be located in
`Open3D/util/conda_package/dist`.

For example, on macOS with Python 3.6, the package is located at
`/Open3D/util/conda_package/dist/osx-64/open3d-0.2.0-py36_0.tar.bz2`

### 3. Install Conda package

Example for macOS with Python 3.6.

```bash
# Activate the new environment that we just created
source activate newenv3.6

# Install the Conda packages
conda install numpy matplotlib # TODO: conda is not picking up the deps yet
conda install Open3D/util/conda_package/dist/osx-64/open3d-0.2.0-py36_0.tar.bz2

# Try importing
python -c "from open3d import *"
```
