#!/bin/bash

. name.sh

. arguments.sh

echo "cleaning up images..."
echo

# for ubuntu in ${ubuntu_version[@]}; do
#     for deps in ${bundle_type[@]}; do
#         # build the tag of the image
#         TAG=${ubuntu}-${deps}
#         echo "removing $NAME:${TAG}..."
#         docker image rm $NAME:${TAG}
#         echo

#         for python in ${python_version[@]}; do
#             # build the tag of the image
#             TAG=${ubuntu}-${deps}-${python}
#             echo "removing $NAME:${TAG}..."
#             docker image rm $NAME:${TAG}
#             echo
#         done
#     done
# done

for ubuntu in ${ubuntu_version[@]}; do
    for deps in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${deps}

        echo "removing $IMAGE_NAME..."
        docker image rm $IMAGE_NAME
        echo

        for python in ${python_version[@]}; do
            . set_variables.sh ${ubuntu} ${deps} ${python}

            echo "removing $IMAGE_NAME..."
            docker image rm $IMAGE_NAME
            echo
        done
    done
done

docker image ls
echo
