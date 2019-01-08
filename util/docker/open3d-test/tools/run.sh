#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/set_variables.sh

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
    -h ${CONTAINER_NAME} \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}

# attach to the running container, clone once & build Open3D twice
echo "running ${IMAGE_NAME}..."
date
docker container exec -it -w /root ${CONTAINER_NAME} bash -c 'bash'

# stop the container
docker container stop -t 0 ${CONTAINER_NAME} >/dev/null 2>&1
