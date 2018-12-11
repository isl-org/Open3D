#!/bin/bash

. set_variables.sh

# build the images this image depends on
if [ "$3" != "" ]; then
    ./build.sh ${1} ${2}
fi
if [ "$2" = "${bundle_type[1]}" ]; then
    ./build.sh ${1} ${bundle_type[0]} ${3}
fi

# build the image
echo "building $IMAGE_NAME..."
date
docker image build -t $IMAGE_NAME -f ../Dockerfiles/${1}/$DOCKERFILE ..
date
echo "done building $IMAGE_NAME..."
echo
