#!/bin/bash

set -e

declare -a build_type=(Debug Release)
declare -a link_type=(STATIC SHARED)
declare -a env_type=(py2 py3 mc2 mc3)

OPEN3D_INSTALL_DIR=~/open3d_install

# display help on the required command line arguments
if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    echo "./build.sh <build_type> <link_type> <env_type>"
    echo
    echo "Required:"
    echo "    Build type:       ${build_type[*]}"
    echo "    Link type:        ${link_type[*]}"
    echo "    Environment type: ${env_type[*]}"
    echo
    exit 1
fi

# display help on the first required argument
if [[ ! " ${build_type[@]} " =~ " $1 " ]]; then
    echo "    options for the 1st argument: ${build_type[*]}"
    echo "    argument provided: '$1'"
    echo
    exit 1
fi

# display help on the second required argument
if [[ ! " ${link_type[@]} " =~ " $2 " ]]; then
    echo "    options for the 2nd argument: ${link_type[*]}"
    echo "    argument provided: '$2'"
    echo
    exit 1
fi

# display help on the third required argument
if [[ ! " ${env_type[@]} " =~ " $3 " ]]; then
    echo "    options for the 3rd argument: ${env_type[*]}"
    echo "    argument provided: '$3'"
    echo
    exit 1
fi
echo

cd open3d
echo

echo "building $2..."
date

# set the library link mode to OFF (STATIC) or ON (SHARED)
SHARED=OFF
if [ "$2" = "STATIC" ]; then
    SHARED="OFF"
elif [ "$2" = "SHARED" ]; then
    SHARED="ON"
fi

# set the python executable
PYTHON=""
if [ "$3" = "py2" ]; then
    PYTHON="/usr/bin/python2"
elif [ "$3" = "py3" ]; then
    PYTHON="/usr/bin/python3"
elif [ "$3" = "mc2" ]; then
    PYTHON="/root/miniconda2/bin/python2"
elif [ "$3" = "mc3" ]; then
    PYTHON="/root/miniconda3/bin/python3"
fi

echo "cmake configure the Open3D project..."
date
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=${1} \
      -DBUILD_SHARED_LIBS=${SHARED} \
      -DPYTHON_EXECUTABLE=${PYTHON} \
      -DBUILD_UNIT_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_DIR} \
      ..
echo

echo "build & install Open3D..."
date
make install -j$(nproc)
echo

if [ "$3" = "py2" ]; then
    echo "building python2 pip package..."
    date

    make pip-package
    #make install-pip-package
elif [ "$3" = "py3" ]; then
    echo "building python3 pip package..."
    date

    make pip-package
    #make install-pip-package
elif [ "$3" = "mc2" ]; then
    echo "building python2 conda package..."
    date

    make conda-package
elif [ "$3" = "mc3" ]; then
    echo "building python3 conda package..."
    date

    make conda-package
fi
date
echo

echo "running the Open3D unit tests..."
date
./bin/tests
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
