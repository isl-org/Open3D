#!/bin/sh

echo "building all images..."
echo

./build.sh 18.04 base
echo

./build.sh 18.04 py2 no_deps
echo

./build.sh 18.04 py2 with_deps
echo

./build.sh 18.04 py3 no_deps
echo

./build.sh 18.04 py3 with_deps
echo

# display images in order to check image size
docker image ls
echo
