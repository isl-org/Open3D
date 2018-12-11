#!/bin/bash

. ./name.sh

echo "stopping containers..."
echo

for ubuntu in 14.04 16.04 18.04; do
    for python in py2 py3; do
        for deps in no_deps with_deps; do
            # build the tag of the image
            TAG=${ubuntu}-${python}-${deps}
            # build the container name
            CONTAINER_NAME=${NAME}-${TAG}

            echo "stopping $CONTAINER_NAME..."
            docker container stop -t 0 $CONTAINER_NAME
            echo
        done
    done
done

docker image ls
echo
