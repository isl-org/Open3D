#!/bin/sh

# $1 must be the Ubuntu version:
# - 14.04
# - 16.04
# - 18.04

# $2 must be the python version:
# - py2
# - py3

# $3 must be:
# - no_deps
# - with_deps

if [ $# -lt 3 ]; then
    echo "./test.sh <Ubuntu_version_nr> <Python_version_nr> <Type>"
    echo
    echo "    Ubuntu version nr: 14.04/16.04/18.04"
    echo "    Python version nr: py2/py3"
    echo "    Type:              no_deps/with_deps"
    echo

    exit 0
fi

# get the name of the upper level directory
NAME=$(bash -c 'basename $(cd .. ; pwd)')
TAG=${1}-${2}-${3}
CONTAINER_NAME=${NAME}-${TAG}
DOCKERFILE=Dockerfile-$TAG

# stop the container if it's already running
docker container stop -t 0 $CONTAINER_NAME

# delete the previous container
docker image rm $NAME:$TAG

# build the image
docker image build -t $NAME:$TAG -f ../Dockerfiles/${1}/$DOCKERFILE ..

# run the container
docker container run \
       --rm \
       -d \
       -t \
       -h $CONTAINER_NAME \
       --name $CONTAINER_NAME \
       $NAME:$TAG

# attach to the container, clone & build & install Open3d
docker container exec -it -w /root $CONTAINER_NAME bash -c '\
    echo && \
    git clone https://github.com/IntelVCL/Open3D.git open3d && \
    cd open3d && \
    echo && \
    echo building... && \
    mkdir -p build && \
    cd build && \
    cmake .. -DBUILD_UNIT_TESTS=ON \
             -DCMAKE_BUILD_TYPE=Release && \
    echo && \
    make -j && \
    echo && \
    echo running the unit tests... && \
    ./bin/unitTests'

# stop the container
docker container stop -t 0 $CONTAINER_NAME

# display images in order to check image size
docker image ls
