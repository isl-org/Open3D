#!/bin/bash

. set_variables.sh

# download the image only if not found locally
if [ 1 -eq ${IMAGE_EXISTS} ]; then
    echo "downloading ${IMAGE_NAME}..."
    docker pull ${IMAGE_NAME}
    echo
fi

# helps sync the container clock with the host clock
TIMEZONE=$(cat /etc/timezone)

# run the container
docker container run \
    --rm \
    -d \
    -t \
    -e ENV_TYPE="${3}" \
    -e TZ="${TIMEZONE}" \
    -e LINK_TYPE="${LINK_TYPE}" \
    -h ${CONTAINER_NAME} \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}

# attach to the running container, clone once & build Open3D twice
echo "testing ${IMAGE_NAME}..."
date

if [ "${3}" = "py2" ] || [ "${3}" = "py3" ]; then
    docker container exec -it ${CONTAINER_NAME} /bin/bash -c '\
        git clone --recursive https://github.com/IntelVCL/Open3D.git open3d && \
        ./test.sh Release ${LINK_TYPE} $ENV_TYPE'
elif [ "${3}" = "mc2" ] || [ "${3}" = "mc3" ]; then
    # the conda settings in .bashrc only work with interactive shells
    # for this reason we need to explicitly activate conda here
    docker container exec -e CONDA_DIR="${CONDA_DIR}" -it ${CONTAINER_NAME} /bin/bash -c '\
        git clone --recursive https://github.com/IntelVCL/Open3D.git open3d && \
        source ${CONDA_DIR}/bin/activate && \
        ./test.sh Release ${LINK_TYPE} $ENV_TYPE'
fi

# stop the container
docker container stop -t 0 ${CONTAINER_NAME}
