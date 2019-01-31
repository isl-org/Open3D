#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/arguments.sh

echo "uploading all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        ${REAL_PATH}/upload.sh ${ubuntu} ${bundle}

        for env in ${env_type[@]}; do
            ${REAL_PATH}/upload.sh ${ubuntu} ${bundle} ${env}
        done
    done
done
echo
