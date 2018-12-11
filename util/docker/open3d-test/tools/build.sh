#!/bin/bash

. set_variables.sh

if [ "$3" = "${deps_type[1]}" ]; then
    ./build.sh ${1} ${2} no_deps
fi

if [ "$3" = "${deps_type[0]}" ]; then
    ./build.sh ${1} base
fi

# build the image
echo "building $NAME:$TAG..."
date
docker image build -t $NAME:$TAG -f ../Dockerfiles/${1}/$DOCKERFILE ..
date
echo "done building $NAME:$TAG..."
echo
