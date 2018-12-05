#!/bin/sh

. ./name.sh

echo "cleaning up images..."
echo

for ubuntu in 16.04 18.04; do
    # build the tag of the image
    TAG=${ubuntu}-base
    echo "removing $NAME:${TAG}..."
    docker image rm $NAME:${TAG}
    echo

    for python in py2 py3; do
        for deps in no_deps with_deps; do
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
