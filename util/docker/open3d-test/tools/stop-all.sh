#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/arguments.sh

echo "stopping containers..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        ${REAL_PATH}/stop.sh ${ubuntu} ${bundle}

        for env in ${env_type[@]}; do
            for link in ${link_type[@]}; do
                ${REAL_PATH}/stop.sh ${ubuntu} ${bundle} ${env} ${link}
            done
        done
    done
done

echo
