#!/bin/sh

# tool used for debugging
# accepts the exact same command line arguments as test.sh

. ./set_variables.sh

docker container run --rm -d -it -h $CONTAINER_NAME --name $CONTAINER_NAME $NAME:$TAG

docker container exec -it $CONTAINER_NAME "bash"

docker container stop $CONTAINER_NAME
