#!/bin/bash

. name.sh

. arguments.sh

echo "stopping containers..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for python in ${python_version[@]}; do
        for deps in ${deps_type[@]}; do
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
