#!/bin/bash

. name.sh

. arguments.sh

# display help on the required command line arguments
if [ $# -eq 0 ] || [ "${1}" = "--help" ]; then
    echo "./build.sh <ubuntu_version> <bundle_type> <env_type>"
    echo
    echo "Required:"
    echo "    Ubuntu version:   ${ubuntu_version[*]}"
    echo "    Bundle type:      ${bundle_type[*]}"
    echo "    Environment type: ${env_type[*]}"
    echo
    exit 1
fi

# display help on the first required argument
if [[ ! " ${ubuntu_version[@]} " =~ " ${1} " ]]; then
    echo "    options for the the 1st argument: ${ubuntu_version[*]}"
    echo "    argument provided: '${1}'"
    echo
    exit 1
fi

# display help on the second required argument
if [[ ! " ${bundle_type[@]} " =~ " ${2} " ]]; then
    echo "    options for the 2nd argument: ${bundle_type[*]}"
    echo "    argument provided: '${2}'"
    echo
    exit 1
fi

# display help on the third required argument
if [ "${3}" != "" ]; then
    if [[ ! " ${env_type[@]} " =~ " ${3} " ]]; then
        echo "    options for the 3rd argument: ${env_type[*]}"
        echo "    argument provided: '${3}'"
        echo
        exit 1
    fi
fi

# build the tag of the image
TAG=${1}-${2}
if [ "${3}" != "" ]; then
    TAG=${TAG}-${3}
fi

# build image name complete with tag
IMAGE_NAME=${NAME}:${TAG}

# build the Dockerfile name
DOCKERFILE=""
if [ "${3}" = "" ]; then
    DOCKERFILE=Dockerfile-${2}
elif [[ "${3}" =~ "py" ]]; then
    DOCKERFILE=Dockerfile-py
elif [[ "${3}" =~ "mc" ]]; then
    DOCKERFILE=Dockerfile-mc
fi

# build the container name
CONTAINER_NAME=${NAME}-${TAG}

# python version
PYTHON=""

# the miniconda2/3 installer filename
MC_INSTALLER=""

# miniconda2/3 install dir
CONDA_DIR=""

if [ "${3}" = "py2" ]; then
    PYTHON="python"
elif [ "${3}" = "py3" ]; then
    PYTHON="python3"
elif [ "${3}" = "mc2" ]; then
    MC_INSTALLER=Miniconda2-latest-Linux-x86_64.sh
    CONDA_DIR="/root/miniconda2"
elif [ "${3}" = "mc3" ]; then
    MC_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
    CONDA_DIR="/root/miniconda3"
fi

# the host/docker Open3D clone locations
Open3D_HOST=~/${NAME}-${1}
Open3D_DOCK=/root/${NAME}-${1}

export IMAGE_NAME
export DOCKERFILE
export CONTAINER_NAME
export PYTHON
export MC_INSTALLER
export CONDA_DIR
export Open3D_HOST
export Open3D_DOCK
