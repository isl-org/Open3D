#!/bin/bash

. set_variables.sh

# build the image
./build.sh ${1} ${2} ${3}

# miniconda needs to be activated before building Open3D
ACTIVATE_CONDA=""
if [ "$3" = "mc2" ]; then
    ACTIVATE_CONDA="source /root/miniconda2/bin/activate"
elif [ "$3" = "mc3" ]; then
    ACTIVATE_CONDA="source /root/miniconda3/bin/activate"
fi

# helps sync the container clock with the host clock
TIMEZONE=$(cat /etc/timezone)

# run the container
docker container run \
    --rm \
    -d \
    -t \
    -e ACTIVATE_CONDA="$ACTIVATE_CONDA" \
    -e ENV_TYPE="$3" \
    -e TZ="$TIMEZONE" \
    -h $CONTAINER_NAME \
    --name $CONTAINER_NAME \
    $IMAGE_NAME

# attach to the running container, clone once & build Open3D twice
echo "testing $IMAGE_NAME..."
date
docker container exec -it -w /root $CONTAINER_NAME bash -c '\
    git clone --recursive https://github.com/IntelVCL/Open3D.git open3d && \
    $ACTIVATE_CONDA && \
    ./test.sh Release STATIC $ENV_TYPE && \
    ./test.sh Release SHARED $ENV_TYPE'

# stop the container
docker container stop -t 0 $CONTAINER_NAME
