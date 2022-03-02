#!/usr/bin/env bash
set -euo pipefail

# The following environment variables are required:
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs 2>/dev/null || true)"} # Empty in macOS

DEVELOPER_BUILD="${DEVELOPER_BUILD:-ON}"
if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
    DEVELOPER_BUILD="ON"
fi
SHARED=${SHARED:-OFF}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)} # POSIX: MacOS + Linux
if [ -z "${BUILD_CUDA_MODULE:+x}" ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        BUILD_CUDA_MODULE=ON
    else
        BUILD_CUDA_MODULE=OFF
    fi
fi
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-ON}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-ON}
LOW_MEM_USAGE=${LOW_MEM_USAGE:-OFF}

# Dependency versions:
# CUDA
if [[ $BUILD_TENSORFLOW_OPS == ON || $BUILD_PYTORCH_OPS == ON || \
    $UBUNTU_VERSION != bionic ]]; then
    # CUDA version in sync with PyTorch and Tensorflow
    CUDA_VERSION=("11-0" "11.0")
    CUDNN_MAJOR_VERSION=8
    CUDNN_VERSION="8.0.5.39-1+cuda11.0"
    GCC_MAX_VER=9
else
    # Without MLOps, ensure Open3D works with the lowest supported CUDA version
    # Not available in Nvidia focal repos
    CUDA_VERSION=("10-1" "10.1")
    CUDNN_MAJOR_VERSION=8
    CUDNN_VERSION="8.0.5.39-1+cuda10.1"
    GCC_MAX_VER=7
fi
# ML
TENSORFLOW_VER="2.5.2"
TENSORBOARD_VER="2.5"
TORCH_CPU_GLNX_VER="1.8.2+cpu"
# TORCH_CUDA_GLNX_VER="1.8.2+cu111"
PYTHON_VER=$(python -c 'import sys; ver=f"{sys.version_info.major}{sys.version_info.minor}"; print(f"cp{ver}-cp{ver}{sys.abiflags}")' 2>/dev/null || true)
TORCH_CUDA_GLNX_URL="https://github.com/isl-org/open3d_downloads/releases/download/torch1.8.2/torch-1.8.2-${PYTHON_VER}-linux_x86_64.whl"
TORCH_MACOS_VER="1.8.2"
TORCH_REPO_URL="https://download.pytorch.org/whl/lts/1.8/torch_lts.html"
# Python
CONDA_BUILD_VER="3.21.4"
PIP_VER="21.1.1"
WHEEL_VER="0.35.1"
STOOLS_VER="50.3.2"
PYTEST_VER="6.0.1"
PYTEST_RANDOMLY_VER="3.8.0"
SCIPY_VER="1.5.4"
YAPF_VER="0.30.0"

OPEN3D_INSTALL_DIR=~/open3d_install
OPEN3D_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

install_python_dependencies() {

    echo "Installing Python dependencies"
    options="$(echo "$@" | tr ' ' '|')"
    if [[ "with-conda" =~ ^($options)$ ]]; then
        conda install conda-build="$CONDA_BUILD_VER" -y
    fi
    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER"
    if [[ "with-unit-test" =~ ^($options)$ ]]; then
        python -m pip install -U scipy=="$SCIPY_VER" pytest=="$PYTEST_VER" \
            pytest-randomly=="$PYTEST_RANDOMLY_VER"
    fi
    if [[ "with-cuda" =~ ^($options)$ ]]; then
        TF_ARCH_NAME=tensorflow-gpu
        TF_ARCH_DISABLE_NAME=tensorflow-cpu
        TORCH_GLNX="$TORCH_CUDA_GLNX_URL"
    else
        TF_ARCH_NAME=tensorflow-cpu
        TF_ARCH_DISABLE_NAME=tensorflow-gpu
        TORCH_GLNX="torch==$TORCH_CPU_GLNX_VER"
    fi

    # TODO: modify other locations to use requirements.txt
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements.txt"
    if [[ "with-jupyter" =~ ^($options)$ ]]; then
        python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements_jupyter.txt"
    fi

    echo
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        # TF happily installs both CPU and GPU versions at the same time, so remove the other
        python -m pip uninstall --yes "$TF_ARCH_DISABLE_NAME"
        python -m pip install -U "$TF_ARCH_NAME"=="$TENSORFLOW_VER"
    fi
    if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            python -m pip install -U "${TORCH_GLNX}" -f "$TORCH_REPO_URL"

        elif [[ "$OSTYPE" == "darwin"* ]]; then
            python -m pip install -U torch=="$TORCH_MACOS_VER" -f "$TORCH_REPO_URL"
        else
            echo "unknown OS $OSTYPE"
            exit 1
        fi
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        python -m pip install -U yapf=="$YAPF_VER"
    fi
    if [[ "purge-cache" =~ ^($options)$ ]]; then
        echo "Purge pip cache"
        python -m pip cache purge 2>/dev/null || true
    fi
}

build_all() {

    echo "Using cmake: $(command -v cmake)"
    cmake --version

    mkdir -p build
    cd build
    GLIBCXX_USE_CXX11_ABI=ON
    if [ "$BUILD_PYTORCH_OPS" == ON ] || [ "$BUILD_TENSORFLOW_OPS" == ON ]; then
        GLIBCXX_USE_CXX11_ABI=OFF
    fi

    cmakeOptions=(
        -DDEVELOPER_BUILD=$DEVELOPER_BUILD
        -DBUILD_SHARED_LIBS="$SHARED"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_LIBREALSENSE=ON
        -DBUILD_CUDA_MODULE="$BUILD_CUDA_MODULE"
        -DBUILD_COMMON_CUDA_ARCHS=ON
        -DBUILD_COMMON_ISPC_ISAS=ON
        -DGLIBCXX_USE_CXX11_ABI="$GLIBCXX_USE_CXX11_ABI"
        -DBUILD_TENSORFLOW_OPS="$BUILD_TENSORFLOW_OPS"
        -DBUILD_PYTORCH_OPS="$BUILD_PYTORCH_OPS"
        -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR"
        -DBUILD_UNIT_TESTS=ON
        -DBUILD_BENCHMARKS=ON
        -DBUILD_EXAMPLES=OFF
    )

    echo
    echo Running cmake "${cmakeOptions[@]}" ..
    cmake "${cmakeOptions[@]}" ..
    echo
    echo "Build & install Open3D..."
    make VERBOSE=1 -j"$NPROC"
    make VERBOSE=1 install -j"$NPROC"
    if [[ "$SHARED" == "ON" ]]; then
        make package
    fi
    make VERBOSE=1 install-pip-package -j"$NPROC"
    echo
}

build_pip_conda_package() {
    # Usage:
    #   build_pip_conda_package            # Default, build both pip and conda
    #   build_pip_conda_package both       # Build both pip and conda
    #   build_pip_conda_package pip        # Build pip only
    #   build_pip_conda_package conda      # Build conda only
    echo "Building Open3D wheel"
    options="$(echo "$@" | tr ' ' '|')"

    BUILD_FILAMENT_FROM_SOURCE=OFF
    set +u
    if [ -f "${OPEN3D_ML_ROOT}/set_open3d_ml_root.sh" ]; then
        echo "Open3D-ML available at ${OPEN3D_ML_ROOT}. Bundling Open3D-ML in wheel."
        # the build system of the main repo expects a master branch. make sure master exists
        git -C "${OPEN3D_ML_ROOT}" checkout -b master || true
        BUNDLE_OPEN3D_ML=ON
    else
        echo "Open3D-ML not available."
        BUNDLE_OPEN3D_ML=OFF
    fi
    if [[ "$DEVELOPER_BUILD" == "OFF" ]]; then
        echo "Building for a new Open3D release"
    fi
    if [[ "build_azure_kinect" =~ ^($options)$ ]]; then
        echo "Azure Kinect enabled in Python wheel."
        BUILD_AZURE_KINECT=ON
    else
        echo "Azure Kinect disabled in Python wheel."
        BUILD_AZURE_KINECT=OFF
    fi
    if [[ "build_jupyter" =~ ^($options)$ ]]; then
        echo "Building Jupyter extension in Python wheel."
        BUILD_JUPYTER_EXTENSION=ON
    else
        echo "Jupyter extension disabled in Python wheel."
        BUILD_JUPYTER_EXTENSION=OFF
    fi
    set -u

    echo
    echo Building with CPU only...
    mkdir -p build
    pushd build # PWD=Open3D/build
    cmakeOptions=("-DBUILD_SHARED_LIBS=OFF"
        "-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DBUILD_COMMON_ISPC_ISAS=ON"
        "-DBUILD_AZURE_KINECT=$BUILD_AZURE_KINECT"
        "-DBUILD_LIBREALSENSE=ON"
        "-DGLIBCXX_USE_CXX11_ABI=OFF"
        "-DBUILD_TENSORFLOW_OPS=ON"
        "-DBUILD_PYTORCH_OPS=ON"
        "-DBUILD_FILAMENT_FROM_SOURCE=$BUILD_FILAMENT_FROM_SOURCE"
        "-DBUILD_JUPYTER_EXTENSION=$BUILD_JUPYTER_EXTENSION"
        "-DCMAKE_INSTALL_PREFIX=$OPEN3D_INSTALL_DIR"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DBUILD_UNIT_TESTS=OFF"
        "-DBUILD_BENCHMARKS=OFF"
        "-DBUNDLE_OPEN3D_ML=$BUNDLE_OPEN3D_ML"
    )
    set -x # Echo commands on
    cmake -DBUILD_CUDA_MODULE=OFF "${cmakeOptions[@]}" ..
    set +x # Echo commands off
    echo
    make VERBOSE=1 -j"$NPROC" pybind open3d_tf_ops open3d_torch_ops

    if [ "$BUILD_CUDA_MODULE" == ON ]; then
        echo
        echo Installing CUDA versions of TensorFlow and PyTorch...
        install_python_dependencies with-cuda purge-cache
        echo
        echo Building with CUDA...
        rebuild_list=(bin lib/Release/*.a lib/_build_config.py cpp lib/ml)
        echo
        echo Removing CPU compiled files / folders: "${rebuild_list[@]}"
        rm -r "${rebuild_list[@]}" || true
        set -x # Echo commands on
        cmake -DBUILD_CUDA_MODULE=ON \
            -DBUILD_COMMON_CUDA_ARCHS=ON \
            "${cmakeOptions[@]}" ..
        set +x # Echo commands off
    fi
    echo

    options="$(echo "$@" | tr ' ' '|')"
    if [[ "pip" =~ ^($options)$ ]]; then
        echo "Packaging Open3D pip package..."
        make VERBOSE=1 -j"$NPROC" pip-package
    elif [[ "conda" =~ ^($options)$ ]]; then
        echo "Packaging Open3D conda package..."
        make VERBOSE=1 -j"$NPROC" conda-package
    else
        echo "Packaging Open3D pip and conda package..."
        make VERBOSE=1 -j"$NPROC" pip-conda-package
    fi
    popd # PWD=Open3D
}

# Test wheel in blank virtual environment
# Usage: test_wheel wheel_path
test_wheel() {
    wheel_path="$1"
    python -m venv open3d_test.venv
    # shellcheck disable=SC1091
    source open3d_test.venv/bin/activate
    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER"
    echo "Using python: $(command -v python)"
    python --version
    echo -n "Using pip: "
    python -m pip --version
    echo "Installing Open3D wheel $wheel_path in virtual environment..."
    python -m pip install "$wheel_path"
    python -c "import open3d; print('Installed:', open3d)"
    python -c "import open3d; print('CUDA enabled: ', open3d.core.cuda.is_available())"
    echo
    # echo "Dynamic libraries used:"
    # DLL_PATH=$(dirname $(python -c "import open3d; print(open3d.cpu.pybind.__file__)"))/..
    # if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    #     find "$DLL_PATH"/{cpu,cuda}/ -type f -print -execdir ldd {} \;
    # elif [[ "$OSTYPE" == "darwin"* ]]; then
    #     find "$DLL_PATH"/cpu/ -type f -execdir otool -L {} \;
    # fi
    echo
    # FIXME: Needed because Open3D-ML master TF and PyTorch is older than dev.
    if [ $BUILD_CUDA_MODULE == ON ]; then
        install_python_dependencies with-cuda
    else
        install_python_dependencies
    fi
    if [ "$BUILD_PYTORCH_OPS" == ON ]; then
        # python -m pip install -r "$OPEN3D_ML_ROOT/requirements-torch.txt"
        python -c \
            "import open3d.ml.torch; print('PyTorch Ops library loaded:', open3d.ml.torch._loaded)"
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == ON ]; then
        # python -m pip install -r "$OPEN3D_ML_ROOT/requirements-tensorflow.txt"
        python -c \
            "import open3d.ml.tf.ops; print('TensorFlow Ops library loaded:', open3d.ml.tf.ops)"
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == ON ] && [ "$BUILD_PYTORCH_OPS" == ON ]; then
        echo "importing in the reversed order"
        python -c "import tensorflow as tf; import open3d.ml.torch as o3d"
        echo "importing in the normal order"
        python -c "import open3d.ml.torch as o3d; import tensorflow as tf"
    fi
    deactivate open3d_test.venv # argument prevents unbound variable error
}

# Run in virtual environment
run_python_tests() {
    # shellcheck disable=SC1091
    source open3d_test.venv/bin/activate
    python -m pip install -U pytest=="$PYTEST_VER" \
        pytest-randomly=="$PYTEST_RANDOMLY_VER" \
        scipy=="$SCIPY_VER" \
        tensorboard=="$TENSORBOARD_VER"
    echo Add --randomly-seed=SEED to the test command to reproduce test order.
    pytest_args=("$OPEN3D_SOURCE_ROOT"/python/test/)
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] || [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        echo Testing ML Ops disabled
        pytest_args+=(--ignore "$OPEN3D_SOURCE_ROOT"/python/test/ml_ops/)
    fi
    python -m pytest "${pytest_args[@]}"
    deactivate open3d_test.venv # argument prevents unbound variable error
}

# Use: run_unit_tests
run_cpp_unit_tests() {
    unitTestFlags=--gtest_shuffle
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    echo "Run ./bin/tests $unitTestFlags --gtest_random_seed=SEED to repeat this test sequence."
    ./bin/tests "$unitTestFlags"
    echo
}

# test_cpp_example runExample
# Need variable OPEN3D_INSTALL_DIR
test_cpp_example() {
    # Now I am in Open3D/build/
    cd ..
    git clone https://github.com/isl-org/open3d-cmake-find-package.git
    cd open3d-cmake-find-package
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
    make -j"$NPROC" VERBOSE=1
    runExample="$1"
    if [ "$runExample" == ON ]; then
        ./Draw --skip-for-unit-test
    fi
    # Now I am in Open3D/open3d-cmake-find-package/build/
    cd ../../build
}

# Install dependencies needed for building documentation (on Ubuntu 18.04)
# Usage: install_docs_dependencies "${OPEN3D_ML_ROOT}"
install_docs_dependencies() {
    echo
    echo Install ubuntu dependencies
    echo Update cmake needed in Ubuntu 18.04
    sudo apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
    sudo apt-add-repository --yes 'deb https://apt.kitware.com/ubuntu/ bionic main'
    ./util/install_deps_ubuntu.sh assume-yes
    sudo apt-get install --yes cmake
    sudo apt-get install --yes doxygen
    sudo apt-get install --yes texlive
    sudo apt-get install --yes texlive-latex-extra
    sudo apt-get install --yes ghostscript
    sudo apt-get install --yes pandoc
    sudo apt-get install --yes ccache
    echo
    echo Install Python dependencies for building docs
    command -v python
    python -V
    python -m pip install -U -q "wheel==$WHEEL_VER" \
        "pip==$PIP_VER"
    python -m pip install -U -q "yapf==$YAPF_VER"
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/docs/requirements.txt"
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements.txt"
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements_jupyter.txt"
    echo
    if [[ -d "$1" ]]; then
        OPEN3D_ML_ROOT="$1"
        echo Installing Open3D-ML dependencies from "${OPEN3D_ML_ROOT}"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements.txt"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements-torch.txt"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements-tensorflow.txt" ||
            python -m pip install tensorflow # FIXME: Remove after Open3D-ML update
    else
        echo OPEN3D_ML_ROOT="$OPEN3D_ML_ROOT" not specified or invalid. Skipping ML dependencies.
    fi
}

# Build documentation
# Usage: build_docs $DEVELOPER_BUILD
build_docs() {
    NPROC=$(nproc)
    echo NPROC="$NPROC"
    mkdir -p build
    cd build
    set +u
    DEVELOPER_BUILD="$1"
    set -u
    if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
        DEVELOPER_BUILD=ON
        DOC_ARGS=""
    else
        DOC_ARGS="--is_release"
        echo "Building docs for a new Open3D release"
        echo
        echo "Building Open3D with ENABLE_HEADLESS_RENDERING=ON for Jupyter notebooks"
        echo
    fi
    cmakeOptions=("-DDEVELOPER_BUILD=$DEVELOPER_BUILD"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DWITH_OPENMP=ON"
        "-DBUILD_AZURE_KINECT=ON"
        "-DBUILD_LIBREALSENSE=ON"
        "-DGLIBCXX_USE_CXX11_ABI=OFF"
        "-DBUILD_TENSORFLOW_OPS=ON"
        "-DBUILD_PYTORCH_OPS=ON"
        "-DBUNDLE_OPEN3D_ML=ON"
        "-DBUILD_EXAMPLES=OFF"
    )
    set -x # Echo commands on
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=ON \
        -DBUILD_GUI=OFF \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        ..
    make install-pip-package -j$NPROC
    make -j$NPROC
    bin/GLInfo
    python -c "from open3d import *; import open3d; print(open3d)"
    cd ../docs # To Open3D/docs
    python make_docs.py $DOC_ARGS --clean_notebooks --execute_notebooks=always --py_api_rst=never
    python -m pip uninstall --yes open3d
    cd ../build
    set +x # Echo commands off
    echo
    echo "Building Open3D with BUILD_GUI=ON for visualization.{gui,rendering} documentation"
    echo
    set -x # Echo commands on
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=OFF \
        -DBUILD_GUI=ON \
        -DBUILD_WEBRTC=ON \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        ..
    make install-pip-package -j$NPROC
    make -j$NPROC
    bin/GLInfo || echo "Expect failure since HEADLESS_RENDERING=OFF"
    python -c "from open3d import *; import open3d; print(open3d)"
    cd ../docs # To Open3D/docs
    python make_docs.py $DOC_ARGS --py_api_rst=always --execute_notebooks=never --sphinx --doxygen
    set +x # Echo commands off
}

maximize_ubuntu_github_actions_build_space() {
    df -h
    $SUDO rm -rf /usr/share/dotnet
    $SUDO rm -rf /usr/local/lib/android
    $SUDO rm -rf /opt/ghc
    $SUDO rm -rf "$AGENT_TOOLSDIRECTORY"
    df -h
}
