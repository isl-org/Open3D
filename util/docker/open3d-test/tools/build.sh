#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/set_variables.sh

# build the image only if not found locally
if [ 0 -eq ${IMAGE_EXISTS} ]; then
    echo "skipping ${IMAGE_NAME}, already exists."
    exit 0
fi

# build the images this image depends on
if [ "${2}" = "${bundle_type[1]}" ]; then
    ${REAL_PATH}/build.sh ${1} ${bundle_type[0]}
fi
if [ "${3}" != "" ]; then
    ${REAL_PATH}/build.sh ${1} ${2}
fi

# download miniconda installer once
if [ "${MC_INSTALLER}" != "" ]; then
    if [ ! -f "${REAL_PATH}/../setup/${MC_INSTALLER}" ]; then
        echo not found
        wget -P "${REAL_PATH}/../setup" "https://repo.anaconda.com/miniconda/${MC_INSTALLER}"
    fi
fi

echo "building ${IMAGE_NAME}..."
date
docker image build \
    --build-arg REPOSITORY="${REPOSITORY}" \
    --build-arg UBUNTU_VERSION="${1}" \
    --build-arg BUNDLE_TYPE="${2}" \
    --build-arg PYTHON="${PYTHON}" \
    --build-arg MC_INSTALLER="${MC_INSTALLER}" \
    --build-arg CONDA_DIR="${CONDA_DIR}" \
    -t ${IMAGE_NAME} -f "${REAL_PATH}/../Dockerfiles/${DOCKERFILE}" \
    ${REAL_PATH}/..
date
echo "done building ${IMAGE_NAME}..."
echo
