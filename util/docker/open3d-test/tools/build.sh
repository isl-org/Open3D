#!/bin/bash

. set_variables.sh

# build the images this image depends on
if [ "${3}" != "" ]; then
    ./build.sh ${1} ${2}
fi
if [ "${2}" = "${bundle_type[1]}" ]; then
    ./build.sh ${1} ${bundle_type[0]}
fi

# download miniconda installer once
if [ "${MC_INSTALLER}" != "" ]; then
    if [ ! -f "../setup/${MC_INSTALLER}" ]; then
        echo not found
        wget -P ../setup "https://repo.anaconda.com/miniconda/${MC_INSTALLER}"
    fi
fi

# check if the image already exists or not
docker image inspect ${IMAGE_NAME} >/dev/null 2>&1
IMAGE_EXISTS=$?

# build the image only if not found
if [ 0 -eq ${IMAGE_EXISTS} ]; then
    echo "skipping ${IMAGE_NAME}, already exists."
    exit 0
else
    echo
    echo "building ${IMAGE_NAME}..."
    date
    docker image build \
        --build-arg UBUNTU_VERSION="${1}" \
        --build-arg BUNDLE_TYPE="${2}" \
        --build-arg PYTHON="${PYTHON}" \
        --build-arg MC_INSTALLER="${MC_INSTALLER}" \
        --build-arg CONDA_DIR="${CONDA_DIR}" \
        -t ${IMAGE_NAME} -f ../Dockerfiles/${DOCKERFILE} ..
    date
    echo "done building ${IMAGE_NAME}..."
    echo
fi
