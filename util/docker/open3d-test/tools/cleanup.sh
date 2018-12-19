#!/bin/bash

. arguments.sh

echo "cleaning up images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${bundle}

        echo "removing ${IMAGE_NAME}..."
        docker image rm ${IMAGE_NAME} >/dev/null 2>&1

        for env in ${env_type[@]}; do
            . set_variables.sh ${ubuntu} ${bundle} ${env}

            echo "removing ${IMAGE_NAME}..."
            docker image rm ${IMAGE_NAME} >/dev/null 2>&1
        done
    done
done

echo
