#!/bin/sh

# get the name of the upper level directory
NAME=$(bash -c 'basename $(cd .. ; pwd)')

# stop the container if it's already running
docker container stop -t 0 $NAME

# delete the previous container
docker image rm $NAME:latest

# build the image
docker image build -t $NAME -f ../Dockerfile_with_deps ..

# run the container
docker container run \
       --rm \
       -d \
       -t \
       -h $NAME \
       --name $NAME \
       $NAME

# attach to the container, clone & build & install Open3d
docker container exec -it -w /root $NAME bash -c '\
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
docker container stop -t 0 $NAME

# display images in order to check image size
docker image ls
