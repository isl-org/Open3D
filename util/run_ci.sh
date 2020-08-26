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

TENSORFLOW_VER="2.3.0"
TORCH_GLNX_VER=("1.5.0+cu101" "1.4.0+cpu")
TORCH_MACOS_VER="1.4.0"
YAPF_VER="0.30.0"

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
if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    CUDA_TOOLKIT_DIR=~/cuda
    export PATH="$CUDA_TOOLKIT_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT_DIR/extras/CUPTI/lib64:$CUDA_TOOLKIT_DIR/lib64"
    if ! which nvcc >/dev/null ; then       # If CUDA is not already installed
        reportRun curl -LO https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
        reportRun sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit --toolkitpath="$CUDA_TOOLKIT_DIR" --defaultroot="$CUDA_TOOLKIT_DIR"
    fi
    nvcc --version
fi

date
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
    reportRun pip install -U tensorflow=="$TENSORFLOW_VER"
fi
if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    # disable pytorch build if CUDA is enabled for now until the problem with caffe2 and cudnn is solved
    BUILD_PYTORCH_OPS="OFF"
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

mkdir -p build
cd build

runBenchmarks=true
OPEN3D_INSTALL_DIR=~/open3d_install
cmakeOptions="-DBUILD_SHARED_LIBS=${SHARED} \
        -DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE \
        -DCUDA_ARCH=BasicPTX \
        -DBUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS} \
        -DBUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS} \
        -DBUILD_RPC_INTERFACE=${BUILD_RPC_INTERFACE} \
        -DBUILD_UNIT_TESTS=ON \
        -DBUILD_BENCHMARKS=ON \
        -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} \
        -DPYTHON_EXECUTABLE=$(which python)"

echo
echo "Running cmake" $cmakeOptions ..
reportRun cmake $cmakeOptions ..
echo

echo "build & install Open3D..."
date
reportRun make VERBOSE=1 -j"$NPROC"
reportRun make install -j"$NPROC"
reportRun make VERBOSE=1 install-pip-package -j"$NPROC"
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
make
if [ "$BUILD_CUDA_MODULE" == "OFF" ]; then
./TestVisualizer
fi
echo

echo "test uninstalling Open3D..."
date
cd ../../../../build
make uninstall

reportJobFinishSession
