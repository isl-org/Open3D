#!/usr/bin/env bash

set -euo pipefail

# The following environment variables are required:
SHARED=${SHARED:-OFF}
NPROC=${NPROC:-$(getconf _NPROCESSORS_ONLN)}    # POSIX: MacOS + Linux
if [ -z "${BUILD_CUDA_MODULE:+x}" ] ; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        BUILD_CUDA_MODULE=ON
    else
        BUILD_CUDA_MODULE=OFF
    fi
fi
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-ON}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-ON}
BUILD_RPC_INTERFACE=${BUILD_RPC_INTERFACE:-ON}
LOW_MEM_USAGE=${LOW_MEM_USAGE:-OFF}

TENSORFLOW_VER="2.3.0"
TORCH_GLNX_VER=("1.5.0+cu101" "1.5.0+cpu")
TORCH_MACOS_VER="1.5.0"
YAPF_VER="0.30.0"

OPEN3D_INSTALL_DIR=~/open3d_install

rj_startts=${rj_startts:-$(date +%s)}
rj_prevts=${rj_prevts:-$rj_startts}
rj_prevj=${rj_prevj:-ReportInit}

reportJobStart() {
    rj_ts=$(date +%s)
    ((rj_dt = rj_ts - rj_prevts)) || true
    echo "$rj_ts EndJob $rj_prevj ran for $rj_dt sec (session started $rj_startts)"
    echo "$rj_ts StartJob $1"
    rj_prevj=$1
    rj_prevts=$rj_ts
}

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

install_cuda_toolkit() {

    CUDA_TOOLKIT_DIR=~/cuda
    export PATH="$CUDA_TOOLKIT_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_TOOLKIT_DIR/extras/CUPTI/lib64:$CUDA_TOOLKIT_DIR/lib64"
    if ! which nvcc >/dev/null ; then       # If CUDA is not already installed
        reportRun curl -LO https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
        reportRun sh cuda_10.1.243_418.87.00_linux.run --silent --toolkit --toolkitpath="$CUDA_TOOLKIT_DIR" --defaultroot="$CUDA_TOOLKIT_DIR"
    fi
    nvcc --version
}


install_dependencies() {

    python -m pip install --upgrade pip
    python -m pip install -U wheel
    unittestDependencies="$1"
    if [ "$unittestDependencies" == ON ] ; then
        python -m pip install -U pytest
        python -m pip install scipy
    fi
    echo

    date
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
        reportRun python -m pip install -U tensorflow=="$TENSORFLOW_VER"
    fi
    if [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
                reportRun python -m pip install -U torch=="${TORCH_GLNX_VER[0]}" -f https://download.pytorch.org/whl/torch_stable.html
            else
                reportRun python -m pip install -U torch=="${TORCH_GLNX_VER[1]}" -f https://download.pytorch.org/whl/torch_stable.html
            fi
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            reportRun python -m pip install -U torch=="$TORCH_MACOS_VER"
        else
            echo "unknown OS $OSTYPE"
            exit 1
        fi
    fi
    if [ "$BUILD_TENSORFLOW_OPS" == "ON" ] || [ "$BUILD_PYTORCH_OPS" == "ON" ]; then
        reportRun python -m pip install -U yapf=="$YAPF_VER"
    fi

}


build_all() {

    mkdir -p build
    cd build

    cmakeOptions=(-DBUILD_SHARED_LIBS="$SHARED" \
        -DBUILD_CUDA_MODULE="$BUILD_CUDA_MODULE" \
        -DCUDA_ARCH=BasicPTX \
        -DBUILD_TENSORFLOW_OPS="$BUILD_TENSORFLOW_OPS" \
        -DBUILD_PYTORCH_OPS="$BUILD_PYTORCH_OPS" \
        -DBUILD_RPC_INTERFACE="$BUILD_RPC_INTERFACE" \
        -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR" \
        -DPYTHON_EXECUTABLE="$(which python)" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_UNIT_TESTS=ON \
        -DBUILD_BENCHMARKS=ON \
    )

    echo
    echo Running cmake "${cmakeOptions[@]}" ..
    reportRun cmake "${cmakeOptions[@]}" ..
    echo

    echo "build & install Open3D..."
    date
    reportRun make VERBOSE=1 -j"$NPROC"
    reportRun make install -j"$NPROC"
    reportRun make VERBOSE=1 install-pip-package -j"$NPROC"
    echo
}


build_wheel() {

    echo
    echo Building with CPU only...
    date
    mkdir -p build
    cd build         # PWD=Open3D/build
    rebuild_list=(bin lib/Release/*.{a,so}  lib/_build_config.py cpp)

    cmakeOptions=(-DBUILD_SHARED_LIBS=OFF \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_TENSORFLOW_OPS=ON \
        -DBUILD_PYTORCH_OPS=OFF \       # TODO: PyTorch Ops is OFF with CUDA
        -DBUILD_RPC_INTERFACE=ON \
        -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR" \
        -DPYTHON_EXECUTABLE="$(which python)" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF \
    )
    reportRun cmake "${cmakeOptions[@]}" ..
    echo
    reportRun make VERBOSE=1 -j"$NPROC" pybind

    if [ "$BUILD_CUDA_MODULE" == ON ] ; then
        echo
        echo Building with CUDA...
        date
        rm -r "${rebuild_list[@]}"
        cmakeOptions=(-DBUILD_SHARED_LIBS=OFF \
            -DBUILD_CUDA_MODULE=ON \
            -DCUDA_ARCH=BasicPTX \
            -DBUILD_TENSORFLOW_OPS=ON \
            -DBUILD_PYTORCH_OPS=OFF \       # TODO: PyTorch Ops is OFF with CUDA
            -DBUILD_RPC_INTERFACE=ON \
            -DCMAKE_INSTALL_PREFIX="$OPEN3D_INSTALL_DIR" \
            -DPYTHON_EXECUTABLE="$(which python)" \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_UNIT_TESTS=OFF \
            -DBUILD_BENCHMARKS=OFF \
        )
        reportRun cmake "${cmakeOptions[@]}" ..
    fi
    echo
    echo "Building Open3D wheel..."
    date
    reportRun make VERBOSE=1 -j"$NPROC" pip-package
}

install_wheel() {
    echo
    echo "Installing Open3D wheel..."
    date
    reportRun python -m pip install lib/python_package/pip_package/open3d-*.whl
}

test_wheel() {
    reportRun python -c "import open3d; print(open3d)"
    reportRun python -c "import open3d; open3d.pybind.core.kernel.test_mkl_integration()"
    reportRun python -c "import open3d; print('CUDA enabled: ', open3d.__cuda__)"
}

# Use: run_unit_tests
run_unit_tests() {
    unitTestFlags=
    [ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
    reportRun ./bin/tests "$unitTestFlags"
    echo
}

run_benchmarks() {
    pytest_args=(../python/test/)
    if [ "$BUILD_TENSORFLOW_OPS" == "OFF" ]; then
        pytest_args+=(--ignore ../python/test/test_tf_op_library.py)
        pytest_args+=(--ignore ../python/test/tf_ops/)
    fi
    reportRun python -m pytest "${pytest_args[@]}"
}

# test_cpp_example runExample
# Need variable OPEN3D_INSTALL_DIR
test_cpp_example() {

    cd ../docs/_static/C++
    mkdir -p build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} ..
    make -j"$NPROC"
    runExample="$1"
    if [ "$runExample" == ON ]; then
        ./TestVisualizer
    fi
    cd ../../../../build

}

repair_wheel() {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}
