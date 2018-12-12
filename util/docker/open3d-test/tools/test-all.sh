#!/bin/bash

. arguments.sh

echo "testing all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for deps in ${bundle_type[@]}; do
        for python in ${env_type[@]}; do
            ./test.sh $ubuntu $deps $python
            echo
        done
    done
done

# display images in order to check image size
docker image ls
echo
