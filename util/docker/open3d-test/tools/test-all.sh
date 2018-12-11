#!/bin/bash

. arguments.sh

echo "testing all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    ./build.sh $ubuntu base
    echo

    for python in ${python_version[@]}; do
        for deps in ${deps_type[@]}; do
            ./test.sh $ubuntu $python $deps
            echo
        done
    done
done

# display images in order to check image size
docker image ls
echo
