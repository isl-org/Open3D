#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/arguments.sh

echo "building all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        for env in ${env_type[@]}; do
            ${REAL_PATH}/build.sh $ubuntu $bundle $env
            echo
        done
    done
done

# display images in order to check image size
docker image ls
echo
