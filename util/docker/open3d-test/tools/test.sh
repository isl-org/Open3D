#!/bin/bash

. set_variables.sh

# build the image
./build.sh ${1} ${2} ${3}

TIMEZONE=$(cat /etc/timezone)

# run the container
docker container run \
    --rm \
    -d \
    -t \
    -e PYTHON=$PYTHON \
    -e TZ=$TIMEZONE \
    -h $CONTAINER_NAME \
    --name $CONTAINER_NAME \
    $IMAGE_NAME

# attach to the running container, clone & build Open3d
echo "testing $IMAGE_NAME..."
docker container exec -it -w /root $CONTAINER_NAME bash -c '\
    git clone --recurse-submodules https://github.com/IntelVCL/Open3D.git open3d && \
    source /root/miniconda2/bin/activate && \
    ./test.sh Release STATIC ${3} && \
    ./test.sh Release SHARED ${3} && \
    bash'

# stop the container
docker container stop -t 0 $CONTAINER_NAME
