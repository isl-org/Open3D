#!/bin/sh

. ./set_variables.sh

# build the image
./build.sh ${1} ${2} ${3}

# run the container
docker container run \
       --rm \
       -d \
       -t \
       -h $CONTAINER_NAME \
       --name $CONTAINER_NAME \
       $NAME:$TAG

# attach to the container, clone & build & install Open3d
echo "testing $NAME:$TAG..."
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
