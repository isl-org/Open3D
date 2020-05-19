#!/usr/bin/env bash
#
# The following environment variables are required:
# - SHARED
# - BUILD_TENSORFLOW_OPS
# - BUILD_DEPENDENCY_FROM_SOURCE
# - NPROC

set -euo pipefail
set -x

#$1 - name of the job
reportJobStart() {
	rj_ts=$(date +%T)
	echo "$rj_ts EndJob $rj_prevj ran $rj_prevts - $rj_ts (session started $rj_startts)"
	echo "$rj_ts StartJob $1"
	rj_prevj=$1
	rj_prevts=$rj_ts
}
rj_startts=$(date +%H%M%S)
rj_prevts=$rj_startts
rj_prevj=ReportInit
echo "$rj_startts StartJob ReportInit"
reportJobFinishSession() {
	rj_ts=$(date +%H%M%S)
	echo "$rj_ts EndJob $rj_prevj ran $rj_prevts - $rj_ts (session started $rj_startts)"
	echo "ReportJobSession: ran $rj_startts - $rj_ts"
}

reportJobStart "installing Python unit test dependencies"
pip install --upgrade pip
pip install -U -q pytest
pip install -U -q wheel
echo

python --version
pytest --version
cmake --version

date
if [ "$BUILD_TENSORFLOW_OPS" == "ON" ]; then
    reportJobStart "install tensorflow"
    pip install -U -q tensorflow==2.0.0
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
        -DBUILD_PNG=ON"
fi

echo
reportJobStart "cmake configure the Open3D project"
echo "Running cmake" $cmakeOptions ..
cmake $cmakeOptions ..
echo

echo "build & install Open3D..."
date
reportJobStart "build -j$NPROC"
make -j$NPROC
reportJobStart "install"
make install -j$NPROC
reportJobStart "install-pip-package"
make install-pip-package -j$NPROC
echo

echo "running Open3D unit tests..."
unitTestFlags=
[ "${LOW_MEM_USAGE-}" = "ON" ] && unitTestFlags="--gtest_filter='-*Reduce*Sum*'"
date
reportJobStart "unitTests"
./bin/unitTests $unitTestFlags
echo

echo "running Open3D python tests..."
date
# TODO: fix TF op library test.
reportJobStart "pytest"
pytest ../src/UnitTest/Python/ --ignore="../src/UnitTest/Python/test_tf_op_library.py"
echo

if $runBenchmarks; then
    echo "running Open3D benchmarks..."
    date
reportJobStart "benchmarks"
    ./bin/benchmarks
    echo
fi

echo "test find_package(Open3D)..."
reportJobStart "other tests"
date
test=$(cmake --find-package \
    -DNAME=Open3D \
    -DCOMPILER_ID=GNU \
    -DLANGUAGE=C \
    -DMODE=EXIST \
    -DCMAKE_PREFIX_PATH="${OPEN3D_INSTALL_DIR}/lib/cmake")
if [ "$test" == "Open3D found." ]; then
    echo "PASSED find_package(Open3D) in specified install path."
else
    echo "FAILED find_package(Open3D) in specified install path."
    exit 1
fi
echo

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

reportJobFinish
