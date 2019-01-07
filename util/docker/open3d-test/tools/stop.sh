#!/bin/bash

. set_variables.sh

echo "stopping ${CONTAINER_NAME}..."
docker container stop -t 0 ${CONTAINER_NAME} >/dev/null 2>&1
echo
