#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/name.sh
. ${REAL_PATH}/arguments.sh

# display help on the required command line arguments
if [ $# -eq 0 ] || [ "${1}" = "--help" ]; then
    echo "./build.sh <ubuntu_version> <bundle_type> <env_type> <link_type>"
    echo
    echo "Required:"
    echo "    Ubuntu version:   ${ubuntu_version[*]}"
    echo "    Bundle type:      ${bundle_type[*]}"
    echo "Optional:"
    echo "    Environment type: ${env_type[*]}"
    echo "    Link type:        ${link_type[*]}"
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

# display help on the fourth required argument
if [ "${4}" != "" ]; then
    if [[ ! " ${link_type[@]} " =~ " ${4} " ]]; then
        echo "    options for the 4th argument: ${link_type[*]}"
        echo "    argument provided: '${4}'"
        echo
        exit 1
    fi
fi

# the name of the repository where the images will be uploaded to
REPOSITORY=intelvcl

# build the tag of the image
TAG=${1}-${2}
if [ "${3}" != "" ]; then
    TAG=${TAG}-${3}
fi

# build image name complete with tag
IMAGE_NAME=${REPOSITORY}/${NAME}:${TAG}

# check if the image already exists or not
docker image inspect ${IMAGE_NAME} >/dev/null 2>&1
IMAGE_EXISTS=$?

# build the Dockerfile name
DOCKERFILE=""
if [ "${3}" = "" ]; then
    DOCKERFILE=Dockerfile-${2}
elif [[ "${3}" =~ "py" ]]; then
    DOCKERFILE=Dockerfile-py
elif [[ "${3}" =~ "mc" ]]; then
    DOCKERFILE=Dockerfile-mc
fi

# build the container host name
# remove the dot in the TAG/Ubuntu version number
#   in order to use the full hostname in the bash prompt
#   otherwise the text after the dot is not displayed
CONTAINER_HOSTNAME=${NAME}-${TAG//.}

# build the container name
# suffix with the link type in order to avoid container name collisions
CONTAINER_NAME=${NAME}-${TAG//.}
if [ "${4}" != "" ]; then
    CONTAINER_NAME=${NAME}-${TAG//.}-${4}
fi

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

# link type, default STATIC
LINK_TYPE=${4}

export IMAGE_NAME
export IMAGE_EXISTS
export DOCKERFILE
export CONTAINER_HOSTNAME
export CONTAINER_NAME
export PYTHON
export MC_INSTALLER
export CONDA_DIR
export Open3D_HOST
export Open3D_DOCK
export LINK_TYPE
