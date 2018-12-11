#!/bin/bash

. set_variables.sh

# build the image
./build.sh ${1} ${2} ${3}

# run the container
docker container run \
    --rm \
    -d \
    -t \
    -e PYTHON=$PYTHON \
    -h $CONTAINER_NAME \
    --name $CONTAINER_NAME \
    $NAME:$TAG

# attach to the running container, clone & build Open3d
echo "testing $NAME:$TAG..."
docker container exec -it -w /root $CONTAINER_NAME bash -c './test.sh'

# stop the container
docker container stop -t 0 $CONTAINER_NAME
