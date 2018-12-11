#!/bin/bash

. name.sh

. arguments.sh

echo "stopping containers..."
echo

# for ubuntu in ${ubuntu_version[@]}; do
#     for deps in ${bundle_type[@]}; do
#         # build the tag of the image
#         TAG=${ubuntu}-${deps}
#         # build the container name
#         CONTAINER_NAME=${NAME}-${TAG}

#         echo "stopping $CONTAINER_NAME..."
#         docker container stop -t 0 $CONTAINER_NAME
#         echo

#         for python in ${python_version[@]}; do
#             # build the tag of the image
#             TAG=${ubuntu}-${deps}-${python}
#             # build the container name
#             CONTAINER_NAME=${NAME}-${TAG}

#             echo "stopping $CONTAINER_NAME..."
#             docker container stop -t 0 $CONTAINER_NAME
#             echo
#         done
#     done
# done

for ubuntu in ${ubuntu_version[@]}; do
    for deps in ${bundle_type[@]}; do
        . set_variables.sh ${ubuntu} ${deps}

        echo "stopping $CONTAINER_NAME..."
        docker container stop -t 0 $CONTAINER_NAME
        echo

        for python in ${python_version[@]}; do
            . set_variables.sh ${ubuntu} ${deps} ${python}

            echo "stopping $CONTAINER_NAME..."
            docker container stop -t 0 $CONTAINER_NAME
            echo
        done
    done
done


docker image ls
echo
