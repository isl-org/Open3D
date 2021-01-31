#!/usr/bin/env bash

# The following environment variables are required:
SUDO=${SUDO:=sudo}
UBUNTU_VERSION=${UBUNTU_VERSION:="$(lsb_release -cs 2>/dev/null || true)"} # Empty in macOS

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
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ "$BUILD_CUDA_MODULE" == OFF ]; then
    BUILD_PYTORCH_OPS=OFF # PyTorch Ops requires CUDA + CUDNN to build
fi
BUILD_RPC_INTERFACE=${BUILD_RPC_INTERFACE:-ON}
LOW_MEM_USAGE=${LOW_MEM_USAGE:-OFF}

# Dependency versions:
# CUDA
CUDA_VERSION=("10-1" "10.1")
CUDNN_MAJOR_VERSION=7
CUDNN_VERSION="7.6.5.32-1+cuda10.1"
# ML
TENSORFLOW_VER="2.3.1"
TORCH_CUDA_GLNX_VER="1.6.0+cu101"
TORCH_CPU_GLNX_VER="1.6.0+cpu"
TORCH_MACOS_VER="1.6.0"
# Python
CONDA_BUILD_VER="3.20.0"
PIP_VER="20.2.4"
WHEEL_VER="0.35.1"
STOOLS_VER="50.3.2"
PYTEST_VER="6.0.1"
SCIPY_VER="1.4.1"
YAPF_VER="0.30.0"

# Documentation
SPHINX_VER=3.1.2
SPHINX_RTD_VER=0.5.0
NBSPHINX_VER=0.7.1
MATPLOTLIB_VER=3.3.3

OPEN3D_INSTALL_DIR=~/open3d_install
OPEN3D_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

install_cuda_toolkit() {

    SUDO=${SUDO:-sudo}
    echo "Installing CUDA ${CUDA_VERSION[1]} with apt ..."
    $SUDO apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    $SUDO apt-add-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /"
    $SUDO apt-get install --yes --no-install-recommends \
        "cuda-minimal-build-${CUDA_VERSION[0]}" \
        "cuda-cusolver-dev-${CUDA_VERSION[0]}" \
        "cuda-cusparse-dev-${CUDA_VERSION[0]}" \
        "cuda-curand-dev-${CUDA_VERSION[0]}" \
        "cuda-cufft-dev-${CUDA_VERSION[0]}" \
        "cuda-nvrtc-dev-${CUDA_VERSION[0]}" \
        "cuda-nvtx-${CUDA_VERSION[0]}" \
        libcublas-dev
    if [ "${CUDA_VERSION[1]}" == "10.1" ]; then
        echo "CUDA 10.1 needs CUBLAS 10.2. Symlinks ensure this is found by cmake"
        dpkg -L libcublas10 libcublas-dev | while read -r cufile; do
            if [ -f "$cufile" ] && [ ! -e "${cufile/10.2/10.1}" ]; then
                set -x
                $SUDO ln -s "$cufile" "${cufile/10.2/10.1}"
                set +x
            fi
        done
    fi
    options="$(echo "$@" | tr ' ' '|')"
    if [[ "with-cudnn" =~ ^($options)$ ]]; then
        # The repository method can cause "File has unexpected size" error so
        # we use a tar file copy approach instead. The scripts are taken from
        # CentOS 6 nvidia-docker scripts. The CUDA version and CUDNN version
        # should be the same as the repository method. Ref:
        # https://gitlab.com/nvidia/container-images/cuda/-/blob/2d67fde701915bd88a15038895203c894b36d3dd/dist/10.1/centos6-x86_64/devel/cudnn7/Dockerfile#L9
        $SUDO apt-get install --yes --no-install-recommends curl
        CUDNN_DOWNLOAD_SUM=7eaec8039a2c30ab0bc758d303588767693def6bf49b22485a2c00bf2e136cb3
        curl -fsSL http://developer.download.nvidia.com/compute/redist/cudnn/v7.6.5/cudnn-10.1-linux-x64-v7.6.5.32.tgz -O
        echo "$CUDNN_DOWNLOAD_SUM  cudnn-10.1-linux-x64-v7.6.5.32.tgz" | sha256sum -c -
        $SUDO tar --no-same-owner -xzf cudnn-10.1-linux-x64-v7.6.5.32.tgz -C /usr/local
        rm cudnn-10.1-linux-x64-v7.6.5.32.tgz
        $SUDO ldconfig
        # We may revisit the repository approach in the future.
        # echo "Installing cuDNN ${CUDNN_VERSION} with apt ..."
        # $SUDO apt-add-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"
        # $SUDO apt-get install --yes --no-install-recommends \
        #     "libcudnn${CUDNN_MAJOR_VERSION}=$CUDNN_VERSION" \
        #     "libcudnn${CUDNN_MAJOR_VERSION}-dev=$CUDNN_VERSION"
    fi
    CUDA_TOOLKIT_DIR=/usr/local/cuda-${CUDA_VERSION[1]}
    [ -e /usr/local/cuda ] || $SUDO ln -s "$CUDA_TOOLKIT_DIR" /usr/local/cuda
    set +u # Disable "unbound variable is error" since that gives a false alarm error below:
    export PATH="${CUDA_TOOLKIT_DIR}/bin${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="${CUDA_TOOLKIT_DIR}/extras/CUPTI/lib64:$CUDA_TOOLKIT_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    set -u
    echo PATH="$PATH"
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    # Ensure g++ < 9 is installed for CUDA 10.1
    cpp_version=$(c++ --version 2>/dev/null | grep -o -E '([0-9]+\.)+[0-9]+' | head -1)
    if dpkg --compare-versions "$cpp_version" ge-nl 9; then
        $SUDO apt-get install --yes --no-install-recommends g++-8 gcc-8
        $SUDO update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-8 70 \
            --slave /usr/bin/gcc gcc /usr/bin/gcc-8
        $SUDO update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-8 70 \
            --slave /usr/bin/g++ g++ /usr/bin/g++-8
    fi
    if [[ "purge-cache" =~ ^($options)$ ]]; then
        $SUDO apt-get clean
        $SUDO rm -rf /var/lib/apt/lists/*
    fi
}

install_python_dependencies() {

    echo "Installing Python dependencies"
    options="$(echo "$@" | tr ' ' '|')"
    if [[ "with-conda" =~ ^($options)$ ]]; then
        conda install conda-build="$CONDA_BUILD_VER" -y
    fi
    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER"
    if [[ "with-unit-test" =~ ^($options)$ ]]; then
        python -m pip install -U pytest=="$PYTEST_VER"
        python -m pip install -U scipy=="$SCIPY_VER"
    fi
    if [[ "with-cuda" =~ ^($options)$ ]]; then
        TF_ARCH_NAME=tensorflow-gpu
        TF_ARCH_DISABLE_NAME=tensorflow-cpu
        TORCH_ARCH_GLNX_VER="$TORCH_CUDA_GLNX_VER"
    else
        TF_ARCH_NAME=tensorflow-cpu
        TF_ARCH_DISABLE_NAME=tensorflow-gpu
        TORCH_ARCH_GLNX_VER="$TORCH_CPU_GLNX_VER"
    fi

    echo
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        # TF happily installs both CPU and GPU versions at the same time, so remove the other
        python -m pip uninstall --yes "$TF_ARCH_DISABLE_NAME"
        python -m pip install -U "$TF_ARCH_NAME"=="$TENSORFLOW_VER"
    fi
    if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            python -m pip install -U torch=="$TORCH_ARCH_GLNX_VER" -f https://download.pytorch.org/whl/torch_stable.html
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            python -m pip install -U torch=="$TORCH_MACOS_VER"
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

install_librealsense2() {

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo Installing librealsense
        echo Reference: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
        $SUDO apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE ||
            $SUDO apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

        $SUDO apt-add-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo ${UBUNTU_VERSION} main" -u
        $SUDO apt-get install --yes --no-install-recommends librealsense2-dkms librealsense2-udev-rules librealsense2-dev
        $SUDO apt-get install --yes --no-install-recommends librealsense2-utils
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install librealsense
    else
        echo "Unsupported OS $OSTYPE"
        exit 1
    fi
}

install_azure_kinect_dependencies() {

    echo "Installing Azure Kinect dependencies"

    SUDO=${SUDO:-sudo}
    curl https://packages.microsoft.com/keys/microsoft.asc | $SUDO apt-key add -
    $SUDO apt-add-repository --yes https://packages.microsoft.com/ubuntu/18.04/prod

    # Accept EULA using a workaround
    # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1190#issuecomment-618473882
    echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | $SUDO debconf-set-selections
    echo 'libk4a1.4 libk4a1.4/accept-eula boolean true' | $SUDO debconf-set-selections

    $SUDO apt-get --yes install libk4a1.4 libk4a1.4-dev k4a-tools
}

build_all() {

    mkdir -p build
    cd build

    cmakeOptions=(-DBUILD_SHARED_LIBS="$SHARED"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_LIBREALSENSE=ON
        -DBUILD_CUDA_MODULE="$BUILD_CUDA_MODULE"
        -DCUDA_ARCH=BasicPTX
        -DBUILD_TENSORFLOW_OPS="$BUILD_TENSORFLOW_OPS"
        -DBUILD_PYTORCH_OPS="$BUILD_PYTORCH_OPS"
        -DBUILD_RPC_INTERFACE="$BUILD_RPC_INTERFACE"
        -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR"
        -DPYTHON_EXECUTABLE="$(which python)"
        -DBUILD_UNIT_TESTS=ON
        -DBUILD_BENCHMARKS=ON
        -DBUILD_EXAMPLES=OFF
    )

    echo
    echo Running cmake "${cmakeOptions[@]}" ..
    cmake "${cmakeOptions[@]}" ..
    echo
    echo "build & install Open3D..."
    make VERBOSE=1 -j"$NPROC"
    make install -j"$NPROC"
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
    if [[ "$DEVELOPER_BUILD" != "OFF" ]]; then # Validate input coming from GHA input field
        DEVELOPER_BUILD="ON"
    else
        echo "Building for a new Open3D release"
    fi
    if [[ "build_azure_kinect" =~ ^($options)$ ]]; then
        echo "Azure Kinect enabled in Python wheel."
        BUILD_AZURE_KINECT=ON
    else
        echo "Azure Kinect disabled in Python wheel."
        BUILD_AZURE_KINECT=OFF
    fi
    set -u

    echo
    echo Building with CPU only...
    mkdir -p build
    cd build # PWD=Open3D/build
    cmakeOptions=(-DBUILD_SHARED_LIBS=OFF
        -DDEVELOPER_BUILD="$DEVELOPER_BUILD"
        -DBUILD_AZURE_KINECT="$BUILD_AZURE_KINECT"
        -DBUILD_LIBREALSENSE=ON
        -DBUILD_TENSORFLOW_OPS=ON
        -DBUILD_PYTORCH_OPS=ON
        -DBUILD_RPC_INTERFACE=ON
        -DBUILD_FILAMENT_FROM_SOURCE="$BUILD_FILAMENT_FROM_SOURCE"
        -DBUILD_JUPYTER_EXTENSION=ON
        -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR"
        -DPYTHON_EXECUTABLE="$(which python)"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_UNIT_TESTS=OFF
        -DBUILD_BENCHMARKS=OFF
        -DBUNDLE_OPEN3D_ML="$BUNDLE_OPEN3D_ML"
    )
    cmake -DBUILD_CUDA_MODULE=OFF "${cmakeOptions[@]}" ..
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
        cmake -DBUILD_CUDA_MODULE=ON -DCUDA_ARCH=BasicPTX "${cmakeOptions[@]}" ..
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
    cd .. # PWD=Open3D
}

# Test wheel in blank virtual environment
# Usage: test_wheel wheel_path
test_wheel() {
    wheel_path="$1"
    python -m venv open3d_test.venv
    source open3d_test.venv/bin/activate
    python -m pip install --upgrade pip=="$PIP_VER" wheel=="$WHEEL_VER" \
        setuptools=="$STOOLS_VER"
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
    if [ "$BUILD_PYTORCH_OPS" == ON ]; then
        python -m pip install -r "$OPEN3D_ML_ROOT/requirements-torch.txt"
        python -c \
            "import open3d.ml.torch; print('PyTorch Ops library loaded:', open3d.ml.torch._loaded)"
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == ON ]; then
        python -m pip install -r "$OPEN3D_ML_ROOT/requirements-tensorflow.txt"
        python -c \
            "import open3d.ml.tf.ops; print('TensorFlow Ops library loaded:', open3d.ml.tf.ops)"
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] && [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        echo "importing in the reversed order"
        python -c "import tensorflow as tf; import open3d.ml.torch as o3d"
        echo "importing in the normal order"
        python -c "import open3d.ml.torch as o3d; import tensorflow as tf"
    fi
    deactivate
}

# Run in virtual environment
run_python_tests() {
    source open3d_test.venv/bin/activate
    python -m pip install -U pytest=="$PYTEST_VER"
    python -m pip install -U scipy=="$SCIPY_VER"
    pytest_args=("$OPEN3D_SOURCE_ROOT"/python/test/)
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] || [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        echo Testing ML Ops disabled
        pytest_args+=(--ignore "$OPEN3D_SOURCE_ROOT"/python/test/ml_ops/)
    fi
    python -m pytest "${pytest_args[@]}"
    deactivate
}

# Use: run_unit_tests
run_cpp_unit_tests() {
    unitTestFlags=
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    ./bin/tests "$unitTestFlags"
    echo
}

# test_cpp_example runExample
# Need variable OPEN3D_INSTALL_DIR
test_cpp_example() {

    cd ../docs/_static/C++
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
    make -j"$NPROC" VERBOSE=1
    runExample="$1"
    if [ "$runExample" == ON ]; then
        ./TestVisualizer
    fi
    cd ../../../../build
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
    which python
    python -V
    python -m pip install -U -q \
        "wheel==$WHEEL_VER" \
        "pip==$PIP_VER" \
        "matplotlib==$MATPLOTLIB_VER" \
        "sphinx==$SPHINX_VER" \
        "sphinx-rtd-theme==$SPHINX_RTD_VER" \
        "nbsphinx==$NBSPHINX_VER"
    # m2r needs a patch for sphinx 3
    # https://github.com/sphinx-doc/sphinx/issues/7420
    python -m pip install -U -q "git+https://github.com/intel-isl/m2r@dev#egg=m2r"
    python -m pip install -U -q "yapf==$YAPF_VER"
    echo
    if [[ -d "$1" ]]; then
        OPEN3D_ML_ROOT="$1"
        echo Installing Open3D-ML dependencies from "${OPEN3D_ML_ROOT}"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements.txt"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements-torch.txt"
        python -m pip install -r "${OPEN3D_ML_ROOT}/requirements-tensorflow.txt"
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
    cmakeOptions=(-DDEVELOPER_BUILD="$DEVELOPER_BUILD"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_JUPYTER_EXTENSION=ON
        -DWITH_OPENMP=ON
        -DBUILD_AZURE_KINECT=ON
        -DBUILD_LIBREALSENSE=ON
        -DBUILD_TENSORFLOW_OPS=ON
        -DBUILD_PYTORCH_OPS=ON
        -DBUILD_RPC_INTERFACE=ON
        -DBUNDLE_OPEN3D_ML=ON
    )
    set -x # Echo commands on
    cmake "${cmakeOptions[@]}" \
        -DENABLE_HEADLESS_RENDERING=ON \
        -DBUILD_GUI=OFF \
        ..
    make install-pip-package -j$NPROC
    make -j$NPROC
    bin/GLInfo
    python -c "from open3d import *; import open3d; print(open3d)"
    cd ../docs # To Open3D/docs
    python make_docs.py $DOC_ARGS --clean_notebooks --execute_notebooks=always --pyapi_rst=never
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
        ..
    make install-pip-package -j$NPROC
    make -j$NPROC
    bin/GLInfo || echo "Expect failure since HEADLESS_RENDERING=OFF"
    python -c "from open3d import *; import open3d; print(open3d)"
    cd ../docs # To Open3D/docs
    python make_docs.py $DOC_ARGS --pyapi_rst=always --execute_notebooks=never --sphinx --doxygen
    set +x # Echo commands off
}

install_arm64_dependencies() {
    apt-get update -q -y
    apt-get install -y apt-utils build-essential git wget
    apt-get install -y python3 python3-dev python3-pip python3-virtualenv
    apt-get install -y xorg-dev libglu1-mesa-dev ccache
    apt-get install -y libblas-dev liblapack-dev liblapacke-dev libssl-dev
    apt-get install -y libsdl2-dev libc++-7-dev libc++abi-7-dev libxi-dev
    apt-get install -y libudev-dev autoconf libtool # librealsense
    apt-get install -y clang-7
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 100
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-7 100
    update-alternatives --install /usr/bin/cc cc /usr/bin/clang 100
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 100
    /usr/sbin/update-ccache-symlinks
    echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc
    virtualenv --python=$(which python) ${HOME}/venv
    source ${HOME}/venv/bin/activate
    which python
    python --version
    pip install pytest=="$PYTEST_VER" -U
    pip install wheel=="$WHEEL_VER" -U
    # Get pre-compiled CMake
    wget https://github.com/intel-isl/Open3D/releases/download/v0.11.0/cmake-3.18-aarch64.tar.gz
    tar -xvf cmake-3.18-aarch64.tar.gz
    cp -ar cmake-3.18-aarch64 ${HOME}
    PATH=${HOME}/cmake-3.18-aarch64/bin:$PATH
    which cmake
    cmake --version
}
