#!/bin/sh

. ./set_variables.sh

if [ "$3" = "with_deps" ]; then
    ./build.sh ${1} ${2} no_deps
fi

if [ "$3" = "no_deps" ]; then
    ./build.sh ${1} base
fi

# build the image
echo "building $NAME:$TAG..."
date
docker image build -t $NAME:$TAG -f ../Dockerfiles/${1}/$DOCKERFILE ..
date
echo "done building $NAME:$TAG..."
echo
