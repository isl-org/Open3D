#!/bin/sh

echo "cleaning up images..."
echo

# use the name of the upper level directory as the image name
NAME=$(bash -c 'basename $(cd .. ; pwd)')

for ubuntu in 18.04; do
    echo "removing $NAME:${ubuntu}-base..."
    docker image rm $NAME:${ubuntu}-base
    echo

    for python in py2 py3; do
        for deps in no_deps with_deps; do
            echo "removing $NAME:${ubuntu}-${python}-${deps}..."
            docker image rm $NAME:${ubuntu}-${python}-${deps}
            echo
        done
    done
done

docker image ls
echo