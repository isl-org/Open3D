#!/usr/bin/env bash
#
# The following environment variables are required:
# - SHARED
# - NPROC
# - BUILD_DEPENDENCY_FROM_SOURCE
# - BUILD_CUDA_MODULE
# - BUILD_TENSORFLOW_OPS
# - BUILD_PYTORCH_OPS
# - LOW_MEM_USAGE

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
    echo "path: $(which $1)"
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
    reportRun curl -LO https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
    reportRun sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit --toolkitpath="$CUDA_TOOLKIT_DIR" --defaultroot="$CUDA_TOOLKIT_DIR"
    export PATH="$CUDA_TOOLKIT_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT_DIR/extras/CUPTI/lib64:$CUDA_TOOLKIT_DIR/lib64"
    nvcc --version
fi

date
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
    reportRun pip install -U tensorflow==2.0.0
fi
if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    # disable pytorch build if CUDA is enabled for now until the problem with caffe2 and cudnn is solved
    BUILD_PYTORCH_OPS="OFF"
fi
if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
            reportRun pip install -U torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        else
            reportRun pip install -U torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        reportRun pip install -U torch==1.4.0
    else
        echo "unknown OS $OSTYPE"
        exit 1
    fi
fi
if [ "$BUILD_TENSORFLOW_OPS" == "ON" -o "$BUILD_PYTORCH_OPS" == "ON" ]; then
    reportRun pip install -U yapf==0.28.0
fi
mkdir build
cd build

runBenchmarks=true
OPEN3D_INSTALL_DIR=~/open3d_install
cmakeOptions="-DBUILD_SHARED_LIBS=${SHARED} \
        -DBUILD_CUDA_MODULE=$BUILD_CUDA_MODULE \
        -DCUDA_ARCH=BasicPTX \
        -DBUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS} \
        -DBUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS} \
        -DBUILD_UNIT_TESTS=ON \
        -DBUILD_BENCHMARKS=ON \
        -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} \
        -DPYTHON_EXECUTABLE=$(which python)"

if [ "$BUILD_DEPENDENCY_FROM_SOURCE" == "ON" ]; then
    cmakeOptions="$cmakeOptions \
        -DBUILD_EIGEN3=ON \
        -DBUILD_FLANN=ON \
        -DBUILD_GLEW=ON \
        -DBUILD_GLFW=ON \
        -DBUILD_JPEG=ON \
        -DBUILD_PNG=ON"
fi

echo
echo "Running cmake" $cmakeOptions ..
reportRun cmake $cmakeOptions ..
echo

echo "build & install Open3D..."
date
reportRun make -j"$NPROC" || make VERBOSE=1
reportRun make install -j"$NPROC"
reportRun make install-pip-package -j"$NPROC" || make VERBOSE=1 install-pip-package
echo

# skip unit tests if built with CUDA
if [ "$BUILD_CUDA_MODULE" == "OFF" ]; then
    echo "running Open3D unit tests..."
    unitTestFlags=
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    date
    reportRun ./bin/tests $unitTestFlags
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
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
make
./TestVisualizer
echo

echo "test uninstalling Open3D..."
date
cd ../../../../build
make uninstall

reportJobFinishSession
