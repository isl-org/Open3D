#!/bin/sh

# tool used for debugging
# accepts the exact same command line arguments as test.sh

. ./set_variables.sh

PYTHON="python"
if [ "$2" = "py2" ]; then
    PYTHON="python2"
elif [ "$2" = "py3" ]; then
    PYTHON="python3"
fi

# run the container
docker container run \
    --rm \
    -d \
    -t \
    -e PYTHON=$PYTHON \
    -h $CONTAINER_NAME \
    --name $CONTAINER_NAME \
    $NAME:$TAG

# attach to the running container
echo "running $NAME:$TAG..."
docker container exec -it -w /root $CONTAINER_NAME bash -c 'bash'

# stop the container
docker container stop $CONTAINER_NAME
