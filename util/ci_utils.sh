#!/usr/bin/env bash

# The following environment variables are required:
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

# Dependency versions
CUDA_VERSION=("10-1" "10.1")
CUDNN_MAJOR_VERSION=7
CUDNN_VERSION="7.6.5.32-1+cuda10.1"
TENSORFLOW_VER="2.3.0"
TORCH_CUDA_GLNX_VER="1.6.0+cu101"
TORCH_CPU_GLNX_VER="1.6.0+cpu"
TORCH_MACOS_VER="1.6.0"
YAPF_VER="0.30.0"
PIP_VER="20.2.2"
WHEEL_VER="0.35.1"
PYTEST_VER="6.0.1"
SCIPY_VER="1.4.1" # Needed by Tensorflow 2.3.0

OPEN3D_INSTALL_DIR=~/open3d_install

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
    python -m pip install --upgrade pip=="$PIP_VER"
    python -m pip install -U wheel=="$WHEEL_VER"
    options="$(echo "$@" | tr ' ' '|')"
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

build_all() {

    mkdir -p build
    cd build

    cmakeOptions=(-DBUILD_SHARED_LIBS="$SHARED"
        -DCMAKE_BUILD_TYPE=Release
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

build_wheel() {

    echo "Building Open3D wheel"
    echo
    echo Building with CPU only...
    mkdir -p build
    cd build # PWD=Open3D/build

    # BUILD_FILAMENT_FROM_SOURCE if Linux and old glibc (Ubuntu 18.04)
    BUILD_FILAMENT_FROM_SOURCE=OFF
    #if [[ "$OSTYPE" == linux-gnu* ]]; then
    #    glibc_version=$(ldd --version | grep -o -E '([0-9]+\.)+[0-9]+' | head -1)
    #    if dpkg --compare-versions "$glibc_version" lt 2.31; then
    #        BUILD_FILAMENT_FROM_SOURCE=ON
    #    fi
    #fi

    cmakeOptions=(-DBUILD_SHARED_LIBS=OFF
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
    )
    cmake -DBUILD_CUDA_MODULE=OFF "${cmakeOptions[@]}" ..
    echo
    make VERBOSE=1 -j"$NPROC" pybind open3d_tf_ops open3d_torch_ops

    if [ "$BUILD_CUDA_MODULE" == ON ]; then
        echo
        echo Installing CUDA versions of Tensorflow and PyTorch...
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
    echo "Packaging Open3D wheel..."
    make VERBOSE=1 -j"$NPROC" pip-package
    cd .. # PWD=Open3D
}

install_wheel() {
    echo
    echo "Installing Open3D wheel..."
    python -m pip install open3d -f lib/python_package/pip_package/
}

test_wheel() {
    python -c "import open3d; print('Installed:', open3d)"
    python -c "import open3d; print('CUDA enabled: ', open3d.core.cuda.is_available())"
    if [ "$BUILD_PYTORCH_OPS" == ON ]; then
        python -c \
            "import open3d.ml.torch; print('PyTorch Ops library loaded:', open3d.ml.torch._loaded)"
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == ON ]; then
        python -c \
            "import open3d.ml.tf.ops; print('Tensorflow Ops library loaded:', open3d.ml.tf.ops)"
    fi
}

# Use: run_unit_tests
run_cpp_unit_tests() {
    unitTestFlags=
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    ./bin/tests "$unitTestFlags"
    echo
}

run_python_tests() {
    pytest_args=(../python/test/)
    if [ "$BUILD_PYTORCH_OPS" == "OFF" ] || [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        echo Testing ML Ops disabled
        pytest_args+=(--ignore ../python/test/ml_ops/)
    fi
    python -m pytest "${pytest_args[@]}"
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
