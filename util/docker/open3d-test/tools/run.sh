#!/bin/bash

# tool used for debugging
# accepts the exact same command line arguments as test.sh

. set_variables.sh

TIMEZONE=$(cat /etc/timezone)

# run the container
docker container run \
    --rm \
    -d \
    -t \
    -e TZ=${TIMEZONE} \
    -h ${CONTAINER_NAME} \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}

# attach to the running container
echo "running ${IMAGE_NAME}..."
docker container exec -it -w /root ${CONTAINER_NAME} bash -c 'bash'

# stop the container
docker container stop ${CONTAINER_NAME}
