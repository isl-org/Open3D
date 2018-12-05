#!/bin/sh

if [ $# -eq 0 ] || [ "$1" = "--help" ]; then
    echo "./build.sh <Ubuntu_version_nr> <base  or <Python_version_nr> <Type>>"
    echo
    echo "    <Ubuntu version> -- base"
    echo "                    |"
    echo "                     -- <Python version> <Type>"
    echo
    echo "    Ubuntu version: 14.04/16.04/18.04"
    echo "    Python version: py2/py3"
    echo "    Type:           no_deps/with_deps"
    echo
    exit 1
fi

if [ "$1" != "14.04" ] && [ "$1" != "16.04" ] && [ "$1" != "18.04" ]; then
    echo "    the first argument must be the Ubuntu version: 14.04/16.04/18.04"
    echo "    argument provided: $1"
    echo
    exit 1
fi

if [ "$2" != "base" ] && [ "$2" != "py2" ] && [ "$2" != "py3" ]; then
    echo "    the second argument must be either 'base' or the Python version: py2/py3"
    echo "    argument provided: $2"
    echo
    exit 1
fi

if [ "$2" != "base" ]; then
    if [ "$3" != "no_deps" ] && [ "$3" != "with_deps" ]; then
        echo "    the third argument must be the build type: no_deps/with_deps"
        echo "    argument provided: $3"
        echo
        exit 1
    fi
fi

# use the name of the upper level directory as the image name
NAME=$(bash -c 'basename $(cd .. ; pwd)')

# build the tag of the image
TAG=${1}
if [ "$2" = "base" ]; then
    TAG=${TAG}-base
else
    TAG=${TAG}-${2}-${3}
fi

# build the Dockerfile name
DOCKERFILE=Dockerfile-${TAG}

# build the image
echo "building $NAME:$TAG..."
docker image build -t $NAME:$TAG -f ../Dockerfiles/${1}/$DOCKERFILE ..
echo

# # display images in order to check image size
# docker image ls
# echo
