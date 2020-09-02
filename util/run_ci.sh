#!/usr/bin/env bash
#
# The following environment variables are required:
# - SHARED
# - NPROC
# - BUILD_CUDA_MODULE
# - BUILD_TENSORFLOW_OPS
# - BUILD_PYTORCH_OPS
# - BUILD_RPC_INTERFACE
# - LOW_MEM_USAGE
# Optional:
# - BUILD_WHEEL_ONLY

BUILD_WHEEL_ONLY=${BUILD_WHEEL_ONLY:=OFF}
CUDA_VERSION=("10-1" "10.1")
CUDNN_MAJOR_VERSION=7
CUDNN_VERSION="7.6.5.32-1+cuda10.1"
TENSORFLOW_VER="2.3.0"
TORCH_GLNX_VER=("1.6.0+cu101" "1.6.0+cpu")
TORCH_MACOS_VER="1.6.0"
YAPF_VER="0.30.0"

# disable incompatible pytorch configurations
if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
    # we need cudnn for building pytorch ops
    if ! find $(dirname $(which nvcc))/.. -name "libcudnn*"; then
        export BUILD_PYTORCH_OPS=OFF
    fi
    # pytorch 1.6 requires at least python 3.6
    if ! python -c "import sys; sys.exit(0) if sys.version_info.major==3 and sys.version_info.minor > 5 else sys.exit(1)"; then
        export BUILD_PYTORCH_OPS=OFF
    fi
fi

set -euo pipefail

# $1 - name of the job
reportJobStart() {
    rj_ts=$(date +%s)
    ((rj_dt = rj_ts - rj_prevts)) || true
    echo "$rj_ts EndJob $rj_prevj ran for $rj_dt sec (session started $rj_startts)"
    echo "$rj_ts StartJob $1"
    rj_prevj=$1
    rj_prevts=$rj_ts
}
rj_startts=$(date +%s)
rj_prevts=$rj_startts
rj_prevj=ReportInit
echo "$rj_startts StartJob ReportInit"
reportJobFinishSession() {
    rj_ts=$(date +%s)
    ((rj_dt = rj_ts - rj_prevts)) || true
    echo "$rj_ts EndJob $rj_prevj ran for $rj_dt sec (session started $rj_startts)"
    ((rj_dt = rj_ts - rj_startts)) || true
    echo "ReportJobSession: ran for $rj_dt sec"
}
reportRun() {
    reportJobStart "$*"
    echo "path: $(which "$1")"
    "$@"
}

install_cuda_deb() {
    echo "Installing CUDA ${CUDA_VERSION[1]} with apt ..."
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo apt-add-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /"
    sudo apt-get install --yes "cuda-toolkit-${CUDA_VERSION[0]}"
    if [ "${CUDA_VERSION[1]}" == "10.1" ]; then
        echo "CUDA 10.1 needs CUBLAS 10.2. Symlinks ensure this is found by cmake"
        dpkg -L libcublas10 libcublas-dev | while read -r cufile ; do
            if [ -f "$cufile" ] && [ ! -e "${cufile/10.2/10.1}" ] ; then
                set -x
                sudo ln -s "$cufile" "${cufile/10.2/10.1}"
                set +x
            fi
        done
    fi
    set +u  # Disable "unbound variable is error" since that gives a false alarm error below:
    if [[ "with-cudnn" =~ ^($1|$2)$ ]] ; then
        echo "Installing cuDNN ${CUDNN_VERSION} with apt ..."
        sudo apt-add-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /"
        sudo apt-get install --yes \
            "libcudnn${CUDNN_MAJOR_VERSION}=$CUDNN_VERSION" \
            "libcudnn${CUDNN_MAJOR_VERSION}-dev=$CUDNN_VERSION"
    fi
    CUDA_TOOLKIT_DIR=/usr/local/cuda-${CUDA_VERSION[1]}
    export PATH="${CUDA_TOOLKIT_DIR}/bin${PATH:+:$PATH}"
    export LD_LIBRARY_PATH="${CUDA_TOOLKIT_DIR}/extras/CUPTI/lib64:$CUDA_TOOLKIT_DIR/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo PATH="$PATH"
    echo LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    if [[ "purge-cache" =~ ^($1|$2)$ ]] ; then
        sudo apt-get clean
        sudo rm -rf /var/lib/apt/lists/*
    fi
    set -u
}

echo "nproc = $(nproc) NPROC = ${NPROC}"
reportJobStart "installing Python unit test dependencies"
echo "using pip: $(which pip)"
pip install --upgrade pip
pip install -U pytest
pip install -U wheel
pip install scipy
echo

echo "using python: $(which python)"
python --version
echo "using pytest: $(which pytest)"
pytest --version
echo "using cmake: $(which cmake)"
cmake --version

date
if [ "$BUILD_CUDA_MODULE" == "ON" ] && \
    ! nvcc --version | grep -q "release ${CUDA_VERSION[1]}" 2>/dev/null ; then
    reportRun install_cuda_deb with-cudnn purge-cache
    nvcc --version
fi

date
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
    reportRun pip install -U tensorflow=="$TENSORFLOW_VER"
fi
if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
            reportRun pip install -U torch=="${TORCH_GLNX_VER[0]}" -f https://download.pytorch.org/whl/torch_stable.html
        else
            reportRun pip install -U torch=="${TORCH_GLNX_VER[1]}" -f https://download.pytorch.org/whl/torch_stable.html
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        reportRun pip install -U torch=="$TORCH_MACOS_VER"
    else
        echo "unknown OS $OSTYPE"
        exit 1
    fi
fi
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
    reportRun pip install -U yapf=="$YAPF_VER"
fi

python -m pip cache purge || true

mkdir -p build
cd build

runBenchmarks=true
OPEN3D_INSTALL_DIR=~/open3d_install
cmakeOptions="-DBUILD_SHARED_LIBS=${SHARED} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE \
        -DCUDA_ARCH=BasicPTX \
        -DBUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS} \
        -DBUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS} \
        -DBUILD_RPC_INTERFACE=${BUILD_RPC_INTERFACE} \
        -DBUILD_UNIT_TESTS=ON \
        -DBUILD_BENCHMARKS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} \
        -DPYTHON_EXECUTABLE=$(which python)"

echo
echo "Running cmake $cmakeOptions .."
reportRun cmake $cmakeOptions ..
echo

echo "build & install Open3D..."
date
reportRun make VERBOSE=1 -j"$NPROC"
reportRun make install -j"$NPROC"
reportRun make VERBOSE=1 install-pip-package -j"$NPROC"
echo

echo "Building examples iteratively..."
date
reportRun make VERBOSE=1 -j"$NPROC" build-examples-iteratively
echo

# skip unit tests if built with CUDA, unless system contains Nvidia GPUs
if [ "$BUILD_CUDA_MODULE" == "OFF" ] || nvidia-smi -L | grep -q GPU ; then
    echo "try importing Open3D python package"
    reportRun python -c "import open3d; print(open3d)"
    reportRun python -c "import open3d; open3d.pybind.core.kernel.test_mkl_integration()"

    echo "running Open3D unit tests..."
    unitTestFlags=
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    date
    reportRun ./bin/tests "$unitTestFlags"
    echo

    if $runBenchmarks; then
        echo "running Open3D python tests..."
        date
        pytest_args=(../python/test/)
        if [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
            pytest_args+=(--ignore ../python/test/test_tf_op_library.py)
            pytest_args+=(--ignore ../python/test/tf_ops/)
        fi
        reportRun pytest "${pytest_args[@]}"
        echo
    fi
fi

reportJobStart "test build C++ example"
echo "test building a C++ example with installed Open3D..."
date
cd ../docs/_static/C++
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
make VERBOSE=1
if [ "$BUILD_CUDA_MODULE" == "OFF" ]; then
./TestVisualizer
fi
echo

echo "test uninstalling Open3D..."
date
cd ../../../../build
make uninstall

reportJobFinishSession
