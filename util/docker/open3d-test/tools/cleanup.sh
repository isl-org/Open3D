#!/bin/sh

echo "cleaning up images..."
echo

docker image rm open3d-test:18.04-py3-with_deps
echo

docker image rm open3d-test:18.04-py3-no_deps
echo

docker image rm open3d-test:18.04-py2-with_deps
echo

docker image rm open3d-test:18.04-py2-no_deps
echo

docker image rm open3d-test:18.04-base
echo

docker image ls
echo