#!/bin/bash

. ./name.sh

declare -a ubuntu_version=(14.04 16.04 18.04)
declare -a python_version=(py2 py3)
declare -a deps_type=(no_deps with_deps)

if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    echo "./build.sh <Ubuntu_version_nr> <base  or <Python_version_nr> <Type>>"
    echo
    echo "    <Ubuntu version> -- base"
    echo "                    |"
    echo "                     -- <Python version> <Type>"
    echo
    echo "    Ubuntu version: ${ubuntu_version[*]}"
    echo "    Python version: ${python_version[*]}"
    echo "    Type:           ${deps_type[*]}"
    echo
    exit 1
fi

if [[ ! " ${ubuntu_version[@]} " =~ " $1 " ]]; then
    echo "    the first argument must be the Ubuntu version: ${ubuntu_version[*]}"
    echo "    argument provided: $1"
    echo
    exit 1
fi

if [ "$2" = "" ]; then
    echo "    the second argument must be either 'base' or the Python version: ${python_version[*]}"
    echo
    exit 1
fi

if [ "$2" != "base" ]; then
    if [[ ! " ${python_version[@]} " =~ " $2 " ]]; then
        echo "    the second argument must be either 'base' or the Python version: ${python_version[*]}"
        echo "    argument provided: $2"
        echo
        exit 1
    fi

    if [ "$3" = "" ]; then
        echo "    the third argument must be the build type: ${deps_type[*]}"
        echo
        exit 1
    fi

    if [[ ! " ${deps_type[@]} " =~ " $3 " ]]; then
        echo "    the third argument must be the build type: ${deps_type[*]}"
        echo "    argument provided: $3"
        echo
        exit 1
    fi
fi

# build the tag of the image
TAG=${1}
if [ "$2" = "base" ]; then
    TAG=${TAG}-base
else
    TAG=${TAG}-${2}-${3}
fi

# build the Dockerfile name
DOCKERFILE=Dockerfile-${TAG}

# build the container name
CONTAINER_NAME=${NAME}-${TAG}

PYTHON="python"
if [ "$2" = "py2" ]; then
    PYTHON="python2"
elif [ "$2" = "py3" ]; then
    PYTHON="python3"
fi

export NAME
export TAG
export DOCKERFILE
export CONTAINER_NAME
export PYTHON
