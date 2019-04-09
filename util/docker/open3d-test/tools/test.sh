#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/set_variables.sh

# download the image only if not found locally
if [ 1 -eq ${IMAGE_EXISTS} ]; then
    echo "downloading ${IMAGE_NAME}..."
    date
    docker pull ${IMAGE_NAME}
    echo
fi

# helps sync the container clock with the host clock
TIMEZONE=$(cat /etc/timezone)

echo "running the ${CONTAINER_HOSTNAME} container..."
date
docker container run \
    --rm \
    -d \
    -t \
    -e ENV_TYPE="${3}" \
    -e TZ="${TIMEZONE}" \
    -e LINK_TYPE="${LINK_TYPE}" \
    -h ${CONTAINER_HOSTNAME} \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}
echo

# attach to the running container, clone/build/test Open3D
echo "testing ${IMAGE_NAME}..."
date
echo

if [ "${3}" = "py2" ] || [ "${3}" = "py3" ]; then
    docker container exec -it ${CONTAINER_NAME} /bin/bash -c '\
        git clone --recursive https://github.com/intel-isl/Open3D.git open3d && \
        ./test.sh Release ${LINK_TYPE} $ENV_TYPE'
elif [ "${3}" = "mc2" ] || [ "${3}" = "mc3" ]; then
    # the conda settings in .bashrc only work with interactive shells
    # for this reason we need to explicitly activate conda here
    docker container exec -e CONDA_DIR="${CONDA_DIR}" -it ${CONTAINER_NAME} /bin/bash -c '\
        git clone --recursive https://github.com/intel-isl/Open3D.git open3d && \
        source ${CONDA_DIR}/bin/activate && \
        ./test.sh Release ${LINK_TYPE} $ENV_TYPE'
fi

# docker exec is returning the result of the tests
TEST_RESULT=$?

echo "stopping the ${CONTAINER_HOSTNAME} container..."
date
docker container stop -t 0 ${CONTAINER_NAME} >/dev/null 2>&1
echo

exit ${TEST_RESULT}
