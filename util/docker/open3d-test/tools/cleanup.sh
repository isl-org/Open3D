#!/bin/bash

. name.sh

. arguments.sh

echo "cleaning up images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    # build the tag of the image
    TAG=${ubuntu}-base
    echo "removing $NAME:${TAG}..."
    docker image rm $NAME:${TAG}
    echo

    for python in ${python_version[@]}; do
        for deps in ${deps_type[@]}; do
            # build the tag of the image
            TAG=${ubuntu}-${python}-${deps}
            echo "removing $NAME:${TAG}..."
            docker image rm $NAME:${TAG}
            echo
        done
    done
done

docker image ls
echo
