#!/bin/bash

. set_variables.sh

# build the images this image depends on
if [ "${3}" != "" ]; then
    ./upload.sh ${1} ${2}
fi
if [ "${2}" = "${bundle_type[1]}" ]; then
    ./upload.sh ${1} ${bundle_type[0]}
fi

echo "uploading ${IMAGE_NAME}..."
docker push ${IMAGE_NAME}
echo
