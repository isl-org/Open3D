#!/bin/sh

. ./set_variables.sh

# build the image
echo "building $NAME:$TAG..."
docker image build -t $NAME:$TAG -f ../Dockerfiles/${1}/$DOCKERFILE ..
echo
