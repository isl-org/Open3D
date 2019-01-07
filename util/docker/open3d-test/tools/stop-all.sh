#!/bin/bash

. arguments.sh

echo "stopping containers..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${bundle}

        echo "stopping ${CONTAINER_NAME}..."
        docker container stop -t 0 ${CONTAINER_NAME} >/dev/null 2>&1

        for env in ${env_type[@]}; do
            . set_variables.sh ${ubuntu} ${bundle} ${env}

            echo "stopping ${CONTAINER_NAME}..."
            docker container stop -t 0 ${CONTAINER_NAME} >/dev/null 2>&1
        done
    done
done

echo
