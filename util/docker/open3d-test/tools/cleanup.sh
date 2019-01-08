#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/arguments.sh

echo "cleaning up images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        . ${REAL_PATH}/set_variables.sh ${ubuntu} ${bundle}

        # remove the image only if found locally
        if [ 0 -eq ${IMAGE_EXISTS} ]; then
            printf "removing ${IMAGE_NAME}..."
            docker image rm ${IMAGE_NAME} >/dev/null 2>&1
            printf "done\n"
        fi

        for env in ${env_type[@]}; do
            . ${REAL_PATH}/set_variables.sh ${ubuntu} ${bundle} ${env}

            # remove the image only if found locally
            if [ 0 -eq ${IMAGE_EXISTS} ]; then
                printf "removing ${IMAGE_NAME}..."
                docker image rm ${IMAGE_NAME} >/dev/null 2>&1
                printf "done\n"
            fi
        done
    done
done
echo
