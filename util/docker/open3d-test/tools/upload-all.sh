#!/bin/bash

. arguments.sh

echo "uploading all images..."
echo

for ubuntu in ${ubuntu_version[@]}; do
    for bundle in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${bundle}

        echo "uploading ${IMAGE_NAME}..."
        docker push ${IMAGE_NAME}
        echo

        for env in ${env_type[@]}; do
            . set_variables.sh ${ubuntu} ${bundle} ${env}

            echo "uploading ${IMAGE_NAME}..."
            docker push ${IMAGE_NAME}
            echo
        done
    done
done

echo
