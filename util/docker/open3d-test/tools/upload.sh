#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/set_variables.sh

# upload the image only if found locally
if [ 0 -eq ${IMAGE_EXISTS} ]; then
    printf "uploading ${IMAGE_NAME}..."
    docker push ${IMAGE_NAME} >/dev/null 2>&1
    printf "done\n"
else
    echo "image ${IMAGE_NAME} not found"
fi
