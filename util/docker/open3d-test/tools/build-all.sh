#!/bin/bash

. arguments.sh

echo "building all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    ./build.sh $ubuntu base
    echo

    for python in ${python_version[@]}; do
        for deps in ${deps_type[@]}; do
            ./build.sh $ubuntu $python $deps
            echo
        done
    done
done

# display images in order to check image size
docker image ls
echo
