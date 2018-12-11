#!/bin/bash

. name.sh

. arguments.sh

if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    echo "./build.sh <Ubuntu_version> <Bundle_type> <Python_version>"
    echo
    echo "Required:"
    echo "    Ubuntu version: ${ubuntu_version[*]}"
    echo "    Bundle type:    ${bundle_type[*]}"
    echo "Optional:"
    echo "    Python version: ${python_version[*]}"
    echo
    exit 1
fi

if [[ ! " ${ubuntu_version[@]} " =~ " $1 " ]]; then
    echo "    the first argument must be the Ubuntu version: ${ubuntu_version[*]}"
    echo "    argument provided: '$1'"
    echo
    exit 1
fi

if [[ ! " ${bundle_type[@]} " =~ " $2 " ]]; then
    echo "    the second argument must be the Bundle type: ${bundle_type[*]}"
    echo "    argument provided: '$2'"
    echo
    exit 1
fi

if [ "$3" != "" ]; then
    if [[ ! " ${python_version[@]} " =~ " $3 " ]]; then
        echo "    the third argument must be the Python version: ${python_version[*]}"
        echo "    argument provided: '$3'"
        echo
        exit 1
    fi
fi

# build the tag of the image
TAG=${1}-${2}
if [ "$3" != "" ]; then
    TAG=${TAG}-${3}
fi

# build image name complete with tag
IMAGE_NAME=${NAME}:${TAG}

# build the Dockerfile name
DOCKERFILE=Dockerfile-${TAG}

# build the container name
CONTAINER_NAME=${NAME}-${TAG}

# set the python executable
PYTHON=$3
if [ "$3" = "py2" ]; then
    PYTHON="python2"
elif [ "$3" = "py3" ]; then
    PYTHON="python3"
fi

export IMAGE_NAME
export DOCKERFILE
export CONTAINER_NAME
export PYTHON
