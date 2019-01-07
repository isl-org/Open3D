#!/bin/bash

. arguments.sh

echo "testing all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        for env in ${env_type[@]}; do
            for link in ${link_type[@]}; do
                ./test.sh $ubuntu $bundle $env $link
                echo
            done
        done
    done
done

# display images in order to check image size
docker image ls
echo
