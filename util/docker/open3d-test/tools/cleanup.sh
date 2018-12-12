#!/bin/bash

. arguments.sh

echo "cleaning up images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for deps in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${deps}

        echo "removing $IMAGE_NAME..."
        docker image rm $IMAGE_NAME
        echo

        for python in ${env_type[@]}; do
            . set_variables.sh ${ubuntu} ${deps} ${python}

            echo "removing $IMAGE_NAME..."
            docker image rm $IMAGE_NAME
            echo
        done
    done
done

docker image ls
echo
