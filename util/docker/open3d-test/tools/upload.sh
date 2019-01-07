#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/set_variables.sh

echo "uploading ${IMAGE_NAME}..."
docker push ${IMAGE_NAME}
echo
