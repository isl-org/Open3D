#!/bin/sh

# $1 must be the Ubuntu version:
# - 14.04
# - 16.04
# - 18.04

if [ $# -eq 0 ]; then
    echo "./build-base.sh <Ubuntu_version_nr>"
    echo
    echo "    Ubuntu version nr: 14.04/16.04/18.04"
    echo

    exit 0
fi

# get the name of the upper level directory
NAME=$(bash -c 'basename $(cd .. ; pwd)')
TAG=${1}-base
DOCKERFILE=Dockerfile-${TAG}

# delete the previous image
docker image rm $NAME:$TAG

# build the image
docker image build -t $NAME:$TAG -f ../Dockerfiles/${1}/$DOCKERFILE ..

# display images in order to check image size
docker image ls
