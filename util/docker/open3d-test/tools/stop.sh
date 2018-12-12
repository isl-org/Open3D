#!/bin/bash

. arguments.sh

echo "stopping containers..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for deps in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${deps}

        echo "stopping $CONTAINER_NAME..."
        docker container stop -t 0 $CONTAINER_NAME
        echo

        for python in ${env_type[@]}; do
            . set_variables.sh ${ubuntu} ${deps} ${python}

            echo "stopping $CONTAINER_NAME..."
            docker container stop -t 0 $CONTAINER_NAME
            echo
        done
    done
done


docker image ls
echo
