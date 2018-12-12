#!/bin/bash

declare -a build_type=(Debug Release)
declare -a link_type=(STATIC SHARED)
declare -a env_type=(py2 py3 mc2 mc3)

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

if [[ ! " ${build_type[@]} " =~ " $1 " ]]; then
    echo "    the first argument must be the build type: ${build_type[*]}"
    echo "    argument provided: '$1'"
    echo
    exit 1
fi

if [[ ! " ${link_type[@]} " =~ " $2 " ]]; then
    echo "    the second argument must be the library link type: ${link_type[*]}"
    echo "    argument provided: '$2'"
    echo
    exit 1
fi

if [[ ! " ${env_type[@]} " =~ " $3 " ]]; then
    echo "    the third argument must be the environment type: ${env_type[*]}"
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
if [ "$3" = "STATIC" ]; then
    SHARED="OFF"
elif [ "$3" = "SHARED" ]; then
    SHARED="ON"
fi

# set the python executable
PYTHON=$3
if [ "$3" = "py2" ]; then
    PYTHON="/usr/bin/python2"
elif [ "$3" = "py3" ]; then
    PYTHON="/usr/bin/python3"
elif [ "$3" = "mc2" ]; then
    PYTHON="/root/miniconda2/bin/python2"
elif [ "$3" = "mc3" ]; then
    PYTHON="/root/miniconda3/bin/python3"
fi

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=${1} \
         -DBUILD_SHARED_LIBS=${SHARED} \
         -DPYTHON_EXECUTABLE=${PYTHON} \
         -DBUILD_UNIT_TESTS=ON
echo
make -j
date
echo

echo "building pip package..."
date
if [ "$3" = "py2" ]; then
    make pip-package
    #make install-pip-package
elif [ "$3" = "py3" ]; then
    make pip-package
    #make install-pip-package
elif [ "$3" = "mc2" ]; then
    make conda-package
elif [ "$3" = "mc3" ]; then
    make conda-package
fi
date
echo

echo "running the unit tests..."
./bin/unitTests

date
echo

echo "cleaning..."
cd ..
rm -rf build
echo
