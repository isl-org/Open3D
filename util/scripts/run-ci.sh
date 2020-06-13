#!/usr/bin/env bash
#
# The following environment variables are required:
# - SHARED
# - BUILD_TENSORFLOW_OPS
# - BUILD_DEPENDENCY_FROM_SOURCE
# - NPROC

set -euo pipefail

#$1 - name of the job
reportJobStart() {
    rj_ts=$(date +%s)
    ((rj_dt=rj_ts-rj_prevts)) || true
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
    ((rj_dt=rj_ts-rj_prevts)) || true
    echo "$rj_ts EndJob $rj_prevj ran for $rj_dt sec (session started $rj_startts)"
    ((rj_dt=rj_ts-rj_startts)) || true
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
echo

echo "using python: $(which python)"
python --version
echo "using pytest: $(which pytest)"
pytest --version
echo "using cmake: $(which cmake)"
cmake --version

date
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
    reportRun pip install -U tensorflow==2.0.0
fi
mkdir build
cd build

runBenchmarks=true
OPEN3D_INSTALL_DIR=~/open3d_install
cmakeOptions="-DBUILD_SHARED_LIBS=$SHARED \
        -DBUILD_TENSORFLOW_OPS=$BUILD_TENSORFLOW_OPS \
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
reportRun make -j$NPROC
reportRun make install -j$NPROC
reportRun make install-pip-package -j$NPROC
echo

echo "running Open3D unit tests..."
unitTestFlags=
[ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter=-*Reduce*Sum*"
date
reportRun ./bin/unitTests $unitTestFlags
echo

echo "running Open3D python tests..."
date
# TODO: fix TF op library test.
reportRun pytest ../src/UnitTest/Python/ --ignore="../src/UnitTest/Python/test_tf_op_library.py"
echo

if $runBenchmarks; then
    echo "running Open3D benchmarks..."
    date
    reportRun ./bin/benchmarks
    echo
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

reportJobStart "cleanup"
echo "cleanup the C++ example..."
date
cd ../
rm -rf build

echo "uninstall Open3D..."
date
cd ../../../build
make uninstall

echo "cleanup Open3D..."
date
cd ../
rm -rf build
rm -rf ${OPEN3D_INSTALL_DIR}
echo

reportJobFinishSession
