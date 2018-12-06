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

PYTHON=""
if [ "$2" != "py2" ]; then
    PYTHON="2"
fi
if [ "$2" != "py3" ]; then
    PYTHON="3"
fi

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
    cmake .. -DPYTHON_EXECUTABLE=/usr/bin/python${PYTHON} \
             -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_UNIT_TESTS=ON && \
    echo && \
    make -j && \
    echo && \
    echo running the unit tests... && \
    ./bin/unitTests'

# stop the container
docker container stop -t 0 $CONTAINER_NAME
