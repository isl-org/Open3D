#!/bin/bash

REAL_PATH=$(dirname $(realpath ${0}))

. ${REAL_PATH}/arguments.sh

echo "testing all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        for env in ${env_type[@]}; do
            for link in ${link_type[@]}; do
                ${REAL_PATH}/test.sh $ubuntu $bundle $env $link
                echo
            done
        done
    done
done
