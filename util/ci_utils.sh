#!/usr/bin/env bash
set -euo pipefail

# The following environment variables are required:
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs 2>/dev/null || true)"} # Empty in macOS

DEVELOPER_BUILD="${DEVELOPER_BUILD:-ON}"
if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
    DEVELOPER_BUILD="ON"
fi
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
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
# CUDA: see docker/docker_build.sh
# ML
TENSORFLOW_VER="2.16.2"
TORCH_VER="2.2.2"
TORCH_REPO_URL="https://download.pytorch.org/whl/torch/"
# Python
PIP_VER="23.2.1"
WHEEL_VER="0.38.4"
STOOLS_VER="67.3.2"
YAPF_VER="0.30.0"
PROTOBUF_VER="4.24.0"

OPEN3D_INSTALL_DIR=~/open3d_install
OPEN3D_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

install_python_dependencies() {

    echo "Installing Python dependencies"
    options="$(echo "$@" | tr ' ' '|')"
    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER"
    if [[ "with-unit-test" =~ ^($options)$ ]]; then
        python -m pip install -U -r python/requirements_test.txt
    fi
    if [[ "with-cuda" =~ ^($options)$ ]]; then
        TF_ARCH_NAME=tensorflow
        TF_ARCH_DISABLE_NAME=tensorflow-cpu
        CUDA_VER=$(nvcc --version | grep "release " | cut -c33-37 | sed 's|[^0-9]||g') # e.g.: 117, 118, 121, ...
        TORCH_GLNX="torch==${TORCH_VER}+cu${CUDA_VER}"
    else
        # tensorflow-cpu wheels for macOS arm64 are not available
        if [[ "$OSTYPE" == "darwin"* ]]; then
            TF_ARCH_NAME=tensorflow
            TF_ARCH_DISABLE_NAME=tensorflow
        else
            TF_ARCH_NAME=tensorflow-cpu
            TF_ARCH_DISABLE_NAME=tensorflow
        fi
        TORCH_GLNX="torch==${TORCH_VER}+cpu"
    fi

    # TODO: modify other locations to use requirements.txt
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements.txt"
    if [[ "with-jupyter" =~ ^($options)$ ]]; then
        python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements_jupyter_build.txt"
    fi

    echo
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        # TF happily installs both CPU and GPU versions at the same time, so remove the other
        python -m pip uninstall --yes "$TF_ARCH_DISABLE_NAME"
        python -m pip install -U "$TF_ARCH_NAME"=="$TENSORFLOW_VER" # ML/requirements-tensorflow.txt
    fi
    if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then # ML/requirements-torch.txt
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            python -m pip install -U "${TORCH_GLNX}" -f "$TORCH_REPO_URL" tensorboard

        elif [[ "$OSTYPE" == "darwin"* ]]; then
            python -m pip install -U torch=="$TORCH_VER" -f "$TORCH_REPO_URL" tensorboard
        else
            echo "unknown OS $OSTYPE"
            exit 1
        fi
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        python -m pip install -U yapf=="$YAPF_VER"
        # Fix Protobuf compatibility issue
        # https://stackoverflow.com/a/72493690/1255535
        # https://github.com/protocolbuffers/protobuf/issues/10051
        python -m pip install -U protobuf=="$PROTOBUF_VER"
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

    cmakeOptions=(
        -DDEVELOPER_BUILD="$DEVELOPER_BUILD"
        -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_LIBREALSENSE=ON
        -DBUILD_CUDA_MODULE="$BUILD_CUDA_MODULE"
        -DBUILD_COMMON_CUDA_ARCHS=ON
        -DBUILD_COMMON_ISPC_ISAS=ON
        # TODO: PyTorch still use old CXX ABI, remove this line when PyTorch is updated
        -DGLIBCXX_USE_CXX11_ABI=OFF
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
    if [[ "$BUILD_SHARED_LIBS" == "ON" ]]; then
        make package
    fi
    make VERBOSE=1 install-pip-package -j"$NPROC"
    echo
}

build_pip_package() {
    echo "Building Open3D wheel"
    options="$(echo "$@" | tr ' ' '|')"

    BUILD_FILAMENT_FROM_SOURCE=OFF
    set +u
    if [ -f "${OPEN3D_ML_ROOT}/set_open3d_ml_root.sh" ]; then
        echo "Open3D-ML available at ${OPEN3D_ML_ROOT}. Bundling Open3D-ML in wheel."
        # the build system of the main repo expects a main branch. make sure main exists
        git -C "${OPEN3D_ML_ROOT}" checkout -b main || true
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
    CXX11_ABI=ON
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        CXX11_ABI=$(python -c "import tensorflow as tf; print('ON' if tf.__cxx11_abi_flag__ else 'OFF')")
    elif [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        CXX11_ABI=$(python -c "import torch; print('ON' if torch._C._GLIBCXX_USE_CXX11_ABI else 'OFF')")
    fi
    echo Building with GLIBCXX_USE_CXX11_ABI="$CXX11_ABI"
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
        "-DGLIBCXX_USE_CXX11_ABI=$CXX11_ABI"
        "-DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS"
        "-DBUILD_PYTORCH_OPS=$BUILD_PYTORCH_OPS"
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

    echo "Packaging Open3D CPU pip package..."
    make VERBOSE=1 -j"$NPROC" pip-package
    mv lib/python_package/pip_package/open3d*.whl . # save CPU wheel

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

    echo "Packaging Open3D full pip package..."
    make VERBOSE=1 -j"$NPROC" pip-package
    mv open3d*.whl lib/python_package/pip_package/ # restore CPU wheel
    popd                                           # PWD=Open3D
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
    echo -n "Using python: $(command -v python)"
    python --version
    echo -n "Using pip: "
    python -m pip --version
    echo "Installing Open3D wheel $wheel_path in virtual environment..."
    python -m pip install "$wheel_path"
    python -W default -c "import open3d; print('Installed:', open3d); print('BUILD_CUDA_MODULE: ', open3d._build_config['BUILD_CUDA_MODULE'])"
    python -W default -c "import open3d; print('CUDA available: ', open3d.core.cuda.is_available())"
    echo
    # echo "Dynamic libraries used:"
    # DLL_PATH=$(dirname $(python -c "import open3d; print(open3d.cpu.pybind.__file__)"))/..
    # if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    #     find "$DLL_PATH"/{cpu,cuda}/ -type f -print -execdir ldd {} \;
    # elif [[ "$OSTYPE" == "darwin"* ]]; then
    #     find "$DLL_PATH"/cpu/ -type f -execdir otool -L {} \;
    # fi
    echo
    if python -c "import sys, open3d; sys.exit(not open3d._build_config['BUILD_PYTORCH_OPS'])"; then
        BUILD_PYTORCH_OPS=ON
        python -m pip install -r "$OPEN3D_ML_ROOT/requirements-torch.txt"
        python -W default -c \
            "import open3d.ml.torch; print('PyTorch Ops library loaded:', open3d.ml.torch._loaded)"
    fi
    if python -c "import sys, open3d; sys.exit(not open3d._build_config['BUILD_TENSORFLOW_OPS'])"; then
        BUILD_TENSORFLOW_OPS=ON
        python -m pip install -r "$OPEN3D_ML_ROOT/requirements-tensorflow.txt"
        python -W default -c \
            "import open3d.ml.tf.ops; print('TensorFlow Ops library loaded:', open3d.ml.tf.ops)"
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == ON ] && [ "$BUILD_PYTORCH_OPS" == ON ]; then
        echo "Importing TensorFlow and torch in the reversed order"
        python -W default -c "import tensorflow as tf; import torch; import open3d.ml.torch as o3d"
        echo "Importing TensorFlow and torch in the normal order"
        python -W default -c "import open3d.ml.torch as o3d; import tensorflow as tf; import torch"
    fi
    deactivate open3d_test.venv # argument prevents unbound variable error
}

# Run in virtual environment
run_python_tests() {
    # shellcheck disable=SC1091
    source open3d_test.venv/bin/activate
    python -m pip install -U -r python/requirements_test.txt
    echo Add --randomly-seed=SEED to the test command to reproduce test order.
    pytest_args=("$OPEN3D_SOURCE_ROOT"/python/test/)
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] && [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        echo Testing ML Ops disabled
        pytest_args+=(--ignore "$OPEN3D_SOURCE_ROOT"/python/test/ml_ops/)
    fi
    python -m pytest "${pytest_args[@]}"
    deactivate open3d_test.venv # argument prevents unbound variable error
    rm -rf open3d_test.venv     # cleanup for testing the next wheel
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
    pushd ../examples/cmake/open3d-cmake-find-package
    mkdir build
    pushd build
    echo Testing build with cmake
    cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
    make -j"$NPROC" VERBOSE=1
    runExample="$1"
    if [ "$runExample" == ON ]; then
        ./Draw --skip-for-unit-test
    fi
    if [ $BUILD_SHARED_LIBS == ON ]; then
        rm -r ./*
        echo Testing build with pkg-config
        export PKG_CONFIG_PATH=${OPEN3D_INSTALL_DIR}/lib/pkgconfig
        echo Open3D build options: $(pkg-config --cflags --libs Open3D)
        c++ ../Draw.cpp -o Draw $(pkg-config --cflags --libs Open3D)
        if [ "$runExample" == ON ]; then
            ./Draw --skip-for-unit-test
        fi
    fi
    popd
    popd
    # Now I am in Open3D/build/
}

# Install dependencies needed for building documentation (on Ubuntu 20.04)
# Usage: install_docs_dependencies "${OPEN3D_ML_ROOT}"
install_docs_dependencies() {
    echo
    echo Install ubuntu dependencies
    echo Update cmake needed in Ubuntu 20.04
    sudo apt-key adv --fetch-keys https://apt.kitware.com/keys/kitware-archive-latest.asc
    sudo apt-add-repository --yes 'deb https://apt.kitware.com/ubuntu/ focal main'
    ./util/install_deps_ubuntu.sh assume-yes
    sudo apt-get install --yes cmake
    sudo apt-get install --yes libxml2-dev libxslt-dev python3-dev
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
    if [[ -d "$1" ]]; then
        OPEN3D_ML_ROOT="$1"
        echo Installing Open3D-ML dependencies from "${OPEN3D_ML_ROOT}"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements.txt"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements-torch.txt"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements-tensorflow.txt"
    else
        echo OPEN3D_ML_ROOT="$OPEN3D_ML_ROOT" not specified or invalid. Skipping ML dependencies.
    fi
    echo
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements.txt"
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/python/requirements_jupyter_build.txt"
    python -m pip install -r "${OPEN3D_SOURCE_ROOT}/docs/requirements.txt"
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
        # TODO: PyTorch still use old CXX ABI, re-enable Tensorflow when PyTorch is updated to use new ABI
        "-DBUILD_TENSORFLOW_OPS=OFF"
        "-DBUILD_PYTORCH_OPS=ON"
        "-DBUILD_EXAMPLES=OFF"
    )
    set -x # Echo commands on
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=ON \
        -DBUNDLE_OPEN3D_ML=OFF \
        -DBUILD_GUI=OFF \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        ..
    make python-package -j$NPROC
    make -j$NPROC
    bin/GLInfo
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}:$PWD/lib/python_package"
    python -c "from open3d import *; import open3d; print(open3d)"
    cd ../docs # To Open3D/docs
    python make_docs.py $DOC_ARGS --clean_notebooks --execute_notebooks=always --py_api_rst=never --py_example_rst=never
    python -m pip uninstall --yes open3d
    cd ../build
    set +x # Echo commands off
    echo
    echo "Building Open3D with BUILD_GUI=ON for visualization.{gui,rendering} documentation"
    echo
    set -x # Echo commands on
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=OFF \
        -DBUNDLE_OPEN3D_ML=ON \
        -DBUILD_GUI=ON \
        -DBUILD_WEBRTC=ON \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        ..
    make python-package -j$NPROC
    make -j$NPROC
    bin/GLInfo || echo "Expect failure since HEADLESS_RENDERING=OFF"
    python -c "from open3d import *; import open3d; print(open3d)"
    cd ../docs # To Open3D/docs
    python make_docs.py $DOC_ARGS --py_api_rst=always --py_example_rst=always --execute_notebooks=never --sphinx --doxygen
    set +x # Echo commands off
}

maximize_ubuntu_github_actions_build_space() {
    # https://github.com/easimon/maximize-build-space/blob/main/action.yml
    df -h .                                  # => 26GB
    $SUDO rm -rf /usr/share/dotnet           # ~17GB
    $SUDO rm -rf /usr/local/lib/android      # ~11GB
    $SUDO rm -rf /opt/ghc                    # ~2.7GB
    $SUDO rm -rf /opt/hostedtoolcache/CodeQL # ~5.4GB
    $SUDO docker image prune --all --force   # ~4.5GB
    $SUDO rm -rf "$AGENT_TOOLSDIRECTORY"
    df -h . # => 53GB
}
